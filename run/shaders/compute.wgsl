@group(0) @binding(0) var<uniform> uniforms: State;
@group(0) @binding(1) var<uniform> uniforms_old: State; // retarded
@group(0) @binding(2) var texture_noise: texture_storage_2d<rgba32uint, read>;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var texture_rt_prev: texture_2d<f32>;
@group(1) @binding(2) var texture_geo_pack_0: texture_storage_2d<rgba32uint, read_write>;
@group(1) @binding(3) var texture_geo_pack_1: texture_storage_2d<rgba32uint, read_write>;
// @group(1) @binding(4) var texture_geo_pack_0_old: texture_2d<u32->;
// @group(1) @binding(5) var texture_geo_pack_1_old: texture_2d<u32->;

@group(2) @binding(0) var<storage> triangles: array<Triangle>;

const SAMPLE_DIRECT: bool = true;
const SAMPLES_PER_PIXEL: i32 = 1;
const ADDITIONAL_BOUNCES_PER_RAY: i32 = 3;
// 0 -> no accumulation
// 1 -> average of all frames
// 2 -> svgf
const ACCUMULATION_MODE: i32 = 2;
const ACCUMULATION_RATIO: f32 = 0.2; // α

fn collect_geo_u(coords: vec2<u32>) -> GeometryElement {
    let sample_pack_0 = textureLoad(texture_geo_pack_0, coords);
    let sample_pack_1 = textureLoad(texture_geo_pack_1, coords);

    return unpack_geo(PackedGeometry(sample_pack_0, sample_pack_1));
}

struct Material {
    albedo: vec3<f32>,
    emittance: vec3<f32>,
    // 0 -> diffuse, 1 -> perfect mirror, 2 -> dielectric, 3 -> glossy (NYI)
    mat_type: u32,
    glossiness: f32,
    index: f32,
}

var<private> materials: array<Material, 9> = array<Material, 9>(
    Material(vec3<f32>(1.)  , vec3<f32>(12.), 0u, 0., 1. ),         // 0 light
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 1u, 1., 1. ),         // 1 mirror
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 2u, 0., 1.5),         // 2 glass
    Material(vec3<f32>(0.75), vec3<f32>(0.) , 0u, 0., 1. ),         // 3 white
    Material(vec3<f32>(0.75, 0.25, 0.25), vec3<f32>(0.), 0u, 0., 1.), // 4 red
    Material(vec3<f32>(0.25, 0.25, 0.75), vec3<f32>(0.), 0u, 0., 1.), // 5 blue
    Material(vec3<f32>(1., 0., 0.), vec3<f32>(12., 0., 0.), 0u, 0., 1.), // 6 light (red)
    Material(vec3<f32>(0., 1., 0.), vec3<f32>(0., 12., 0.), 0u, 0., 1.), // 7 light (green)
    Material(vec3<f32>(0., 0., 1.), vec3<f32>(0., 0., 12.), 0u, 0., 1.), // 8 light (blue)
);

const NUM_EMISSIVE: u32 = 6u;
var<private> emissive_triangles: array<u32, 6> = array<u32, 6>(0u, 1u, 2u, 3u, 4u, 5u);

fn intersect_stuff(ray: Ray, out_intersection: ptr<function, Intersection>) -> bool {
    var intersected = false;

    for (var i = 0u; i < arrayLength(&triangles); i++) {
        if triangle_intersect(triangles[i], ray, (*out_intersection).distance, out_intersection) {
            intersected = true;
        }
    }

    return intersected;
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
    if intersect_stuff(hitcheck_ray, &hitcheck_intersection)
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

fn new_cs(pixel: vec2<u32>, dimensions: vec2<u32>, geo_sample: GeometryElement) -> PixelSample {
    var sample_info: PixelSample;

    sample_info.normal = geo_sample.normal;
    sample_info.position = geo_sample.position;

    var light = vec3<f32>(0.);
    var direct_illum = vec3<f32>(0.);
    var attenuation = vec3<f32>(1.);
    var have_diffuse_geo = false;

    var intersection = Intersection(
        length(geo_sample.position - uniforms.camera_position),
        geo_sample.position,
        geo_sample.normal,
        -normalize(geo_sample.position - uniforms.camera_position),
        triangles[geo_sample.object_index].material,
    );

    for (var depth = 0; depth < ADDITIONAL_BOUNCES_PER_RAY; depth++) {
        let material = materials[intersection.material_idx];

        var was_specular = false;
        let wi_and_weight = get_wi_and_weight(intersection, &was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;

        if depth == 0 {
            have_diffuse_geo = !was_specular;
        }

        if !have_diffuse_geo && !was_specular {
            have_diffuse_geo = true;
            sample_info.normal = intersection.normal;
            sample_info.position = intersection.position;
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
        if !intersect_stuff(ray, &new_intersection) {
            break;
        }
        intersection = new_intersection;
    }

    light += attenuation * materials[intersection.material_idx].emittance;

    sample_info.rt = light;

    return sample_info;
}

fn shitty_gauss_variance(pixel: vec2<i32>) -> f32 {
    var kernel = array<f32, 9>(
        1., 2., 1.,
        2., 4., 2.,
        1., 2., 1.,
    );

    var kern_sum = 0.;
    var val_sum = 0.;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let coords = pixel + vec2<i32>(x, y);
            let kern_value = kernel[(x + 1) + (y + 1) * 3];
            kern_sum += kern_value;
            val_sum += collect_geo_i(coords).variance;
        }
    }

    return val_sum / kern_sum;
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

    var sample_sum: PixelSample;
    for (var i = 0; i < SAMPLES_PER_PIXEL; i++) {
        sample_sum = pixel_sample_add(sample_sum, new_cs(pixel, texture_dimensions, geo_sample));
    }
    let sample = pixel_sample_div(sample_sum, f32(SAMPLES_PER_PIXEL));

    var rt_to_write: vec3<f32>;

    let pos_old = (uniforms_old.camera_transform * vec4<f32>(geo_sample.position, 1.));
    let uv_old = ((pos_old.xyz / pos_old.w).xy / 2. + vec2<f32>(0.5));
    let uv_corrected = vec2<f32>(uv_old.x, 1. - uv_old.y);
    let pixel_old = vec2<i32>(round(uv_corrected * vec2<f32>(texture_dimensions)));
    let rt_old = textureLoad(texture_rt_prev, pixel_old, 0).xyz;

    switch ACCUMULATION_MODE {
        case 0: { rt_to_write = sample.rt; }
        case 1: {
            let prev_weight = f32(uniforms.frame_no) / f32(uniforms.frame_no + 1);
            let new_weight = 1. / f32(uniforms.frame_no + 1);

            // let prev_rt = textureLoad(texture_rt_prev, pixel, 0).xyz;
            rt_to_write = rt_old * prev_weight + sample.rt * new_weight;
        }
        case 2: {
            let prev_weight = (1. - ACCUMULATION_RATIO);
            let new_weight = ACCUMULATION_RATIO;

            // let prev_rt = textureLoad(texture_rt_prev, pixel, 0).xyz;
            rt_to_write = rt_old * prev_weight + sample.rt * new_weight;
        }
        default: {}
    }

    var out_geo = geo_sample;

    /*let prev = textureLoad(texture_rt_prev, pixel, 0);
    let integrated = rt_to_write;
    out_geo.variance = sqrt(abs(dot(prev, prev) - dot(integrated, integrated)));*/
    //out_geo.variance = length(pixel_old) / 512.;
    //let pos_new = (uniforms.camera_transform * vec4<f32>(sample.position, 1.));
    //let pos_delta = /*uniforms.inverse_transform * */ (pos_old - pos_new);
    //out_geo.variance = length(pos_delta.xyz / pos_delta.w);

    //out_geo.normal = sample.normal;
    out_geo.normal = rt_old;
    out_geo.position = sample.position;
    out_geo.distance_from_origin = length(sample.position);
    out_geo.variance = 1.;

    let packed_geo = pack_geo(out_geo);

    textureStore(texture_rt, pixel, vec4<f32>(rt_to_write, 1.));
    //textureStore(geo_texture_pack_normal_depth, pixel, vec4<f32>(sample.normal, geo_sample.depth));
    textureStore(texture_geo_pack_0, pixel, packed_geo.pack_0);
    textureStore(texture_geo_pack_1, pixel, packed_geo.pack_1);
}
