@group(0) @binding(0) var<uniform> uniforms: State;
@group(0) @binding(1) var<uniform> uniforms_old: State; // retarded
@group(0) @binding(2) var texture_noise: texture_storage_2d<rgba32uint, read>;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var texture_rt_prev: texture_2d<f32>;
@group(1) @binding(2) var texture_geo_pack_0: texture_storage_2d<rgba32uint, read_write>;
@group(1) @binding(3) var texture_geo_pack_1: texture_storage_2d<rgba32uint, read_write>;
@group(1) @binding(4) var texture_geo_pack_0_old: texture_2d<u32>;
@group(1) @binding(5) var texture_geo_pack_1_old: texture_2d<u32>;

@group(2) @binding(0) var<storage> triangles: array<Triangle>;

const SAMPLE_DIRECT: bool = true;
const SAMPLES_PER_PIXEL: i32 = 1;
const ADDITIONAL_BOUNCES_PER_RAY: i32 = 4;
// 0 -> no accumulation
// 1 -> average of all frames
// 2 -> svgf
const ACCUMULATION_MODE: i32 = 2;
const ACCUMULATION_RATIO: f32 = 0.2; // α

const USE_FIXED_PIPELINE: bool = true;

fn collect_geo_u(coords: vec2<u32>) -> GeometryElement { return collect_geo_ts2d(coords, texture_geo_pack_0, texture_geo_pack_1); }

struct Material {
    albedo: vec3<f32>,
    emittance: vec3<f32>,
    // 0 -> diffuse, 1 -> perfect mirror, 2 -> dielectric, 3 -> glossy (NYI)
    mat_type: u32,
    glossiness: f32,
    index: f32,
}

var<private> materials: array<Material, 10> = array<Material, 10>(
    Material(vec3<f32>(1.)  , vec3<f32>(12.), 0u, 0., 1. ),         // 0 light
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 1u, 1., 1. ),         // 1 mirror
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 2u, 0., 1.5),         // 2 glass
    Material(vec3<f32>(0.75), vec3<f32>(0.) , 0u, 0., 1. ),         // 3 white
    Material(vec3<f32>(0.75, 0.25, 0.25), vec3<f32>(0.), 0u, 0., 1.), // 4 red
    Material(vec3<f32>(0.25, 0.25, 0.75), vec3<f32>(0.), 0u, 0., 1.), // 5 blue
    Material(vec3<f32>(1., 0., 0.), vec3<f32>(12., 0., 0.), 0u, 0., 1.), // 6 light (red)
    Material(vec3<f32>(0., 1., 0.), vec3<f32>(0., 12., 0.), 0u, 0., 1.), // 7 light (green)
    Material(vec3<f32>(0., 0., 1.), vec3<f32>(0., 0., 12.), 0u, 0., 1.), // 8 light (blue)
    Material(vec3<f32>(1., 1., 1.), vec3<f32>(0.5, 0.5, 0.5), 0u, 0., 1.),
);

const NUM_EMISSIVE: u32 = 6u;
var<private> emissive_triangles: array<u32, 6> = array<u32, 6>(0u, 1u, 2u, 3u, 4u, 5u);

fn intersect_stuff(ray: Ray, out_intersection: ptr<function, Intersection>) -> u32 {
    var object_index = 0u;

    for (var i = 0u; i < arrayLength(&triangles); i++) {
        if triangle_intersect(triangles[i], ray, (*out_intersection).distance, out_intersection) {
            object_index = i + 1u;
        }
    }

    return object_index;
}

fn sample_direct_lighting(intersection: Intersection) -> vec3<f32> {
    let going_in = dot(intersection.wo, intersection.normal) > 0.;
    let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

    let triangle_selection = i32(trunc(rand() * f32(NUM_EMISSIVE - 1)));

    let tri_idx = emissive_triangles[triangle_selection];
    let tri = triangles[tri_idx];
    let sample = triangle_sample(tri);
    let material = materials[tri.material];

    let vector_towards_light = sample.position - intersection.position;
    let square_distance = dot(vector_towards_light, vector_towards_light);
    let distance_to_light = sqrt(square_distance);
    let wi = vector_towards_light / distance_to_light;

    let hitcheck_ray = Ray(sample.position, wi);
    var hitcheck_intersection: Intersection;
    if intersect_stuff(hitcheck_ray, &hitcheck_intersection) != 0u
        && abs(hitcheck_intersection.distance - distance_to_light) > 0.01 {
        return vec3<f32>(0.);
    }

    /*let nld = wi;
    let area = triangle_area(tri);
    let direction_to_light = (tri.vertices[0] + tri.vertices[1] + tri.vertices[2]).xyz / 3. - intersection.position;
    let distance_to_light_sq = dot(direction_to_light, direction_to_light);

    let cos_a_max = sqrt(1. - clamp(area * area / distance_to_light_sq, 0., 1.));
    let weight = 2. * (1. - cos_a_max);
    return material.emittance * material.albedo * weight * clamp(dot(nld, intersection.normal), 0., 1.);*/

    let brdf = FRAC_1_PI * 0.5;
    //let power_heuristic = (sample.pdf * sample.pdf) / (sample.pdf * sample.pdf + brdf * brdf);

    let p = abs(dot(sample.normal, -wi)) / square_distance;
    return material.emittance / PI * abs(dot(intersection.normal, vector_towards_light)) * p * triangle_area(tri);
}

fn get_wi_and_weight(intersection: Intersection, out_was_specular: ptr<function, bool>) -> vec4<f32> {
    let material = materials[intersection.material_idx];
    let going_in = dot(intersection.wo, intersection.normal) > 0.;
    let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

    var wi: vec3<f32>;
    var cos_brdf_over_wi_pdf: f32;
    var was_specular = false;

    switch material.mat_type {
        case 0u: {
            let brdf = FRAC_1_PI * 0.5;

            var sample_probability: f32;

            let importance_sample = sample_cos_hemisphere_3d(&sample_probability);
            wi = rodrigues_fast(importance_sample, oriented_normal);

            /*let sample = sample_sphere_3d(&sample_probability);
            wi = select(sample, -sample, dot(oriented_normal, sample) < 0.);
            sample_probability *= 2.;*/

            cos_brdf_over_wi_pdf = dot(wi, oriented_normal) * brdf / sample_probability;
        }

        case 1u: {
            was_specular = true;
            cos_brdf_over_wi_pdf = 1.;

            wi = reflect(intersection.wo, oriented_normal);
        }

        case 2u: {
            was_specular = true;
            cos_brdf_over_wi_pdf = 1.;

            let transmittant_index = select(1., material.index, going_in);
            let incident_index     = select(material.index, 1., going_in);

            var refraction: vec3<f32>;
            var p_reflection: f32;
            if refract(
                intersection.wo, oriented_normal,
                incident_index, transmittant_index,
                &p_reflection, &refraction
            ) {
                let generated = rand();

                if generated > p_reflection {
                    wi = refraction;
                } else {
                    wi = reflect(intersection.wo, oriented_normal);
                }
            } else {
                wi = reflect(intersection.wo, oriented_normal);
            }
        }

        default: {}
    }

    *out_was_specular = was_specular;
    return vec4<f32>(wi, cos_brdf_over_wi_pdf);
}

struct PixelSample {
    rt: vec3<f32>,
    normal: vec3<f32>,
    position: vec3<f32>,
}

fn pixel_sample_add(lhs: PixelSample, rhs: PixelSample) -> PixelSample {
    return PixelSample(
        lhs.rt + rhs.rt,
        lhs.normal + rhs.normal,
        lhs.position + rhs.position,
    );
}

fn pixel_sample_div(lhs: PixelSample, divisor: f32) -> PixelSample {
    return PixelSample(
        lhs.rt / divisor,
        lhs.normal / divisor,
        lhs.position / divisor,
    );
}

fn make_ray(pixel: vec2<u32>) -> Ray {
    let screen_dims = vec2<f32>(f32(uniforms.width), f32(uniforms.height));
    let pixel_corr = vec2<f32>(f32(pixel.x), screen_dims.y - f32(pixel.y));
    
    // the 1.5 fixes the fov issue and i have no clue why
    let ndc_pixel = ((pixel_corr / screen_dims) * 2. - 1.) * 1.5;

    let coord = uniforms.inverse_transform * vec4<f32>(ndc_pixel, 0., 1.);
    let ray_dir = normalize((coord.xyz / coord.w) - uniforms.camera_position);

    return Ray(uniforms.camera_position, ray_dir);
}

fn new_cs(pixel: vec2<u32>, dimensions: vec2<u32>, in_geo_sample: GeometryElement, out_geo: ptr<function, GeometryElement>) -> vec3<f32> {
    var light = vec3<f32>(0.);
    var direct_illum = vec3<f32>(0.);
    var attenuation = vec3<f32>(1.);
    var have_diffuse_geo = true;

    var intersection: Intersection;
    var geo_sample: GeometryElement;

    if USE_FIXED_PIPELINE {
        geo_sample = in_geo_sample;
        *out_geo = in_geo_sample;
        intersection = Intersection(
            length(geo_sample.position - uniforms.camera_position),
            geo_sample.position,
            geo_sample.normal,
            -normalize(geo_sample.position - uniforms.camera_position),
            triangles[geo_sample.object_index - 1].material,
        );
    } else {
        let ray = make_ray(pixel);

        intersection = dummy_intersection(ray);
        let object_index = intersect_stuff(ray, &intersection);

        if object_index == 0u {
            *out_geo = GeometryElement (
                /* albedo   */ vec3<f32>(1.),
                /* variance */ 0.,
                /* normal   */ -ray.direction,
                /* depth    */ 1.,
                /* position */ vec3<f32>(INF, INF, INF),
                /* distance */ INF,
                /* index    */ 0u,
            );

            return get_skybox_ray(ray.direction);
        }

        let geo = GeometryElement (
            /* albedo   */ materials[intersection.material_idx].albedo,
            /* variance */ 0., // TODO
            /* normal   */ intersection.normal,
            /* depth    */ intersection.distance,
            /* position */ intersection.position,
            /* distance */ length(intersection.position),
            /* index    */ object_index,
        );

        *out_geo = geo;
        geo_sample = geo;
    }

    // return geo_sample.albedo;

    var intersection_object_index = geo_sample.object_index;

    // testing
    /*let ray = make_ray(pixel);
    intersection = dummy_intersection(ray);
    intersect_stuff(ray, &intersection);*/

    for (var depth = 0; depth < ADDITIONAL_BOUNCES_PER_RAY; depth++) {
        let material = materials[intersection.material_idx];

        var was_specular = false;
        let wi_and_weight = get_wi_and_weight(intersection, &was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;

        /*if depth == 0 {
            have_diffuse_geo = !was_specular;
        }*/

        if !have_diffuse_geo && !was_specular {
            have_diffuse_geo = true;
            *out_geo = GeometryElement(
                /* albedo   */ materials[intersection.material_idx].albedo,
                /* variance */ geo_sample.variance,
                /* normal   */ intersection.normal,
                /* depth    */ geo_sample.depth,
                // /* position */ intersection.position,
                /* position */ geo_sample.position, // preserve this for reprojection
                // /* distance */ length(intersection.position),
                /* distance */ geo_sample.distance_from_origin,
                /* index    */ intersection_object_index,
            );
        }

        light += attenuation * material.emittance;

        if depth == 0 {
            direct_illum = light;
        }

        if depth == 0 {
            attenuation *= cos_brdf_over_wi_pdf;
        } else {
            attenuation *= material.albedo * cos_brdf_over_wi_pdf;
        }

        if !was_specular && SAMPLE_DIRECT {
            light += material.albedo * attenuation * sample_direct_lighting(intersection);
        }

        let offset = intersection.normal * 0.009 * select(1., -1., dot(wi, intersection.normal) < 0.);
        let ray = Ray(intersection.position + offset, wi);
        var new_intersection: Intersection = dummy_intersection(ray);
        intersection_object_index = intersect_stuff(ray, &new_intersection);
        if intersection_object_index == 0u {
            //let sky_intersection = sky(ray.direction);
            //light += materials[sky_intersection.material_idx].emittance * attenuation;
            let skybox_sample = get_skybox_ray(ray.direction);
            light += attenuation * skybox_sample;
            return light;
        }
        intersection = new_intersection;
    }

    light += attenuation * materials[intersection.material_idx].emittance;

    return light;
}

fn get_previous(pixel: vec2<i32>, geo_at_pixel: GeometryElement) -> vec3<f32> {
    let pos_old = (uniforms_old.camera_transform * vec4<f32>(geo_at_pixel.position, 1.));
    let uv_old = ((pos_old.xyz / pos_old.w).xy / 2. + vec2<f32>(0.5));
    let uv_corrected = vec2<f32>(uv_old.x, 1. - uv_old.y);

    let fractional_pixel_old = uv_corrected * vec2<f32>(f32(uniforms.width), f32(uniforms.height));
    let rounded_pixel_old = select(round(fractional_pixel_old), trunc(fractional_pixel_old), USE_FIXED_PIPELINE);
    let pixel_old = vec2<i32>(rounded_pixel_old);

    let old_geo = collect_geo_t2d(vec2<u32>(pixel_old), texture_geo_pack_0_old, texture_geo_pack_1_old);

    let same_face = old_geo.object_index == geo_at_pixel.object_index;
    let similarity_normal = dot(old_geo.normal, geo_at_pixel.normal);
    let similarity_albedo = abs(dot(normalize(old_geo.albedo), normalize(geo_at_pixel.albedo)));
    let similarity_location = abs(old_geo.distance_from_origin - geo_at_pixel.distance_from_origin);

    let similarity_score = 
        select(-1., 1., same_face) +
        select(-2., similarity_normal, similarity_normal >= 0.75) +
        select(-1., similarity_albedo, similarity_albedo >= 0.75) +
        select(-1., (0.2 - similarity_location) / 0.2, similarity_location < 0.2) +
        0.;

    let invalidated = similarity_score < 1.5;
    let rt_old = textureLoad(texture_rt_prev, pixel_old, 0).xyz;

    return select(rt_old, vec3<f32>(0.), invalidated);
}

fn trace(pixel: vec2<u32>, geo_sample: GeometryElement, out_geo: ptr<function, GeometryElement>) -> vec3<f32> {
    var ret_sum: vec3<f32>;
    var out_geo_tmp: GeometryElement;
    for (var i = 0; i < SAMPLES_PER_PIXEL; i++) {
        let cur_sample = new_cs(pixel, vec2<u32>(uniforms.width, uniforms.height), geo_sample, &out_geo_tmp);
        ret_sum += cur_sample;
    }
    ret_sum /= f32(SAMPLES_PER_PIXEL);

    *out_geo = out_geo_tmp;
    return ret_sum;
}

fn accumulate(pixel: vec2<u32>, color: vec3<f32>, geo_sample: GeometryElement) -> vec3<f32> {
    var ret: vec3<f32>;

    // let rt_old = textureLoad(texture_rt_prev, pixel, 0).xyz;
    let rt_old = get_previous(vec2<i32>(pixel), geo_sample);

    switch ACCUMULATION_MODE {
        case 0: { ret = color ; }
        case 1: {
            let prev_weight = f32(uniforms.frame_no) / f32(uniforms.frame_no + 1);
            let new_weight = 1. / f32(uniforms.frame_no + 1);

            ret = rt_old * prev_weight + color * new_weight;
        }
        case 2: {
            let prev_weight = (1. - ACCUMULATION_RATIO);
            let new_weight = ACCUMULATION_RATIO;

            ret = rt_old * prev_weight + color * new_weight;
        }
        default: {}
    }

    return ret;
}

@compute @workgroup_size(8, 8) fn cs_main(
    @builtin(global_invocation_id)   global_id: vec3<u32>,
    @builtin(workgroup_id)           workgroup_id: vec3<u32>,
    @builtin(local_invocation_id)    local_id:  vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let pixel = global_id.xy;
    let texture_dimensions = textureDimensions(texture_rt);

    setup_rng(global_id.xy, texture_dimensions, local_idx);

    let geo_sample = collect_geo_u(pixel);

    if geo_sample.object_index == 0u {
        let ray = make_ray(pixel);

        // all geo is trash, basically
        let out_geo = GeometryElement(
            /* albedo       */ vec3<f32>(1.),
            /* variance     */ 0.,
            /* normal       */ -ray.direction,
            /* depth        */ 1.,
            /* position     */ vec3<f32>(INF),
            /* distance     */ INF,
            /* object_index */ 0,
        );

        let packed_geo = pack_geo(out_geo);

        //let asdasd = get_skybox_uv(vec2<f32>(pixel) / vec2<f32>(f32(texture_dimensions.x), f32(texture_dimensions.y)));
        let asdasd = get_skybox_ray(ray.direction);

        textureStore(texture_rt, pixel, vec4<f32>(asdasd, 1.));
        textureStore(texture_geo_pack_0, pixel, packed_geo.pack_0);
        textureStore(texture_geo_pack_1, pixel, packed_geo.pack_1);
    } else {
        var out_geo: GeometryElement;
        let cur_luminance = trace(pixel, geo_sample, &out_geo);

        let rt_to_write = accumulate(pixel, cur_luminance, out_geo);
        //let rt_to_write = accumulate(pixel, cur_luminance, geo_sample);
        textureStore(texture_rt, pixel, vec4<f32>(rt_to_write, 1.));

        let packed_geo = pack_geo(out_geo);
        textureStore(texture_geo_pack_0, pixel, packed_geo.pack_0);
        textureStore(texture_geo_pack_1, pixel, packed_geo.pack_1);
    }
}
