@group(0) @binding(1) var texture_noise: texture_storage_2d<rgba32uint, read>;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var<storage, read_write> geometry_buffer: array<GeometryElement>;
@group(1) @binding(2) var texture_denoise_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(3) var texture_denoise_1: texture_storage_2d<rgba8unorm, read_write>;

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
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 2u, 0., 1.7),         // 2 glass
    Material(vec3<f32>(0.75), vec3<f32>(0.) , 0u, 0., 1. ),         // 3 white
    Material(vec3<f32>(0.75, 0.25, 0.25), vec3<f32>(0.), 0u, 0., 1.), // 4 red
    Material(vec3<f32>(0.25, 0.25, 0.75), vec3<f32>(0.), 0u, 0., 1.), // 5 blue
    Material(vec3<f32>(1., 0., 0.), vec3<f32>(12., 0., 0.), 0u, 0., 1.), // 6 light (red)
    Material(vec3<f32>(0., 1., 0.), vec3<f32>(0., 12., 0.), 0u, 0., 1.), // 7 light (green)
    Material(vec3<f32>(0., 0., 1.), vec3<f32>(0., 0., 12.), 0u, 0., 1.), // 8 light (blyue)
);

const NUM_SPHERES: u32 = 2u;
var<private> spheres: array<Sphere, NUM_SPHERES> = array<Sphere, 2>(
    Sphere(vec3<f32>(-1.75 , -2.5 + 0.9      , 17.5  ), .9     , 1u), // mirror
    Sphere(vec3<f32>(1.75  , -2.5 + 0.9 + 0.2, 16.5  ), .9     , 2u), // glass
    //Sphere(vec3<f32>(0.    , 42.499          , 15.   ), 40.    , 0u), // light
    /*
    Sphere(vec3<f32>(0.    , 0.              , -5000.), 4980.  , 3u), // front wall
    Sphere(vec3<f32>(0.    , 0.              , 5000. ), 4980.  , 3u), // backwall
    Sphere(vec3<f32>(5000. , 0.              , 0.    ), 4996.5 , 5u), // right wall
    Sphere(vec3<f32>(0.    , 5000.           , 5.    ), 4997.5 , 3u), // ceiling
    Sphere(vec3<f32>(-5000., 0.              , 0.    ), 4996.5 , 4u), // left wall
    Sphere(vec3<f32>(0.    , -5000.          , 5.    ), 4997.5 , 3u), // floor
    */
);

const CBL: vec3<f32> = vec3<f32>(-3.5, -2.5, -20.);
const CTR: vec3<f32> = vec3<f32>(3.5, 2.5, 20.);

const NUM_TRIANGLES = 18u;
var<private> triangles: array<Triangle, NUM_TRIANGLES> = array<Triangle, NUM_TRIANGLES>(
    // front wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),

    // back wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),

    // right wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 5u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 5u),

    // ceiling
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),

    // left wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 4u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 4u),

    // floor
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CTR.z)), vec2<f32>(0.), vec2<f32>(1.), 3u),

    // light 1
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-3., 2.4, 15.), vec3<f32>(-1., 2.4, 15.), vec3<f32>(-1., 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 6u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1., 2.4, 11.25), vec3<f32>(-3., 2.4, 11.25), vec3<f32>(-3., 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 6u),

    // light 2
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1., 2.4, 15.), vec3<f32>(3., 2.4, 15.), vec3<f32>(3., 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 7u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(3., 2.4, 11.25), vec3<f32>(1., 2.4, 11.25), vec3<f32>(1., 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 7u),

    // light 3
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 8.25)), vec2<f32>(0.), vec2<f32>(1.), 8u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 12.)), vec2<f32>(0.), vec2<f32>(1.), 8u),

    // light 2
    //Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 0u),
    //Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 0u),
);

const NUM_EMISSIVE: u32 = 6u;
var<private> emissive_triangles: array<u32, NUM_EMISSIVE> = array<u32, NUM_EMISSIVE>(12u, 13u, 14u, 15u, 16u, 17u);

fn intersect_stuff(ray: Ray, out_intersection: ptr<function, Intersection>) -> bool {
    var intersection: Intersection = *out_intersection;

    let tri = Triangle(
        array<vec3<f32>, 3>(
            vec3<f32>(2., -1., 14.),
            vec3<f32>(2.5, -1., 14.),
            vec3<f32>(2.5, -0.5, 14.),
        ),
        vec2<f32>(0.), vec2<f32>(1.),
        4u,
    );

    var intersected = false;

    for (var i = 0u; i < NUM_SPHERES; i++) {
        var cur: Intersection;
        if sphere_intersect(spheres[i], ray, intersection.distance, &cur) {
            intersected = true;
            intersection = pick_intersection(intersection, cur);
        }
    }

    for (var i = 0u; i < NUM_TRIANGLES; i++) {
        var cur: Intersection;
        if triangle_intersect(triangles[i], ray, intersection.distance, &cur) {
            intersected = true;
            intersection = pick_intersection(intersection, cur);
        }
    }

    //intersected = intersected | triangle_intersect(tri, ray, intersection.distance, &intersection);

    *out_intersection = intersection;

    return intersected;
}

fn sample_direct_lighting(intersection: Intersection) -> vec3<f32> {
    let material = materials[intersection.material_idx];
    
    if material.mat_type != 0u {
        return vec3<f32>(0.);
    }

    var sum = vec3<f32>(0.);

    let going_in = dot(intersection.wo, intersection.normal) > 0.;
    let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

    for (var i = 0u; i < NUM_EMISSIVE; i++) {
        let tri_idx = emissive_triangles[i];
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
            continue;
        }

        let brdf = FRAC_1_PI * 0.5;
        //let power_heuristic = (sample.pdf * sample.pdf) / (sample.pdf * sample.pdf + brdf * brdf);

        let p = abs(dot(sample.normal, wi)) / dot(vector_towards_light, vector_towards_light);
        sum += material.emittance * abs(dot(intersection.normal, vector_towards_light)) * p / triangle_area(tri);
    }

    return sum;
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
            var sample_probability: f32;
            let importance_sample = sample_cos_hemisphere_3d(&sample_probability);
            //wi = intersection.refl_to_surface * importance_sample;
            wi = rodrigues_fast(importance_sample, oriented_normal);

            let brdf = FRAC_1_PI * 0.5;
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
    albedo: vec3<f32>,
    normal: vec3<f32>,
    position: vec3<f32>,
    depth: f32,
    direct_illum: vec3<f32>,
}

fn pixel_sample_add(lhs: PixelSample, rhs: PixelSample) -> PixelSample {
    return PixelSample(
        lhs.rt + rhs.rt,
        lhs.albedo + rhs.albedo,
        lhs.normal + rhs.normal,
        lhs.position + rhs.position,
        lhs.depth + rhs.depth,
        lhs.direct_illum + rhs.direct_illum,
    );
}

fn pixel_sample_div(lhs: PixelSample, divisor: f32) -> PixelSample {
    return PixelSample(
        lhs.rt / divisor,
        lhs.albedo / divisor,
        lhs.normal / divisor,
        lhs.position / divisor,
        lhs.depth / divisor,
        lhs.direct_illum / divisor,
    );
}

fn radiance(initial_intersection: Intersection) -> vec3<f32> {
    var attenuation = vec3<f32>(1.);
    var light = vec3<f32>(0.);

    var ray: Ray;
    {
        var _was_specular: bool;
        let wi_and_weight = get_wi_and_weight(initial_intersection, &_was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;
        let offset = initial_intersection.normal * 0.0009 * select(1., -1., dot(wi, initial_intersection.normal) < 0.);
        ray = Ray(initial_intersection.position + offset, wi);
    }

    for (var i = 0u; i < 4u; i++) {
        var intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &intersection) {
            break;
        }
        let material = materials[intersection.material_idx];

        // let explicit_lighting = sample_direct_lighting(intersection);

        let going_in = dot(ray.direction, intersection.normal) < 0.;
        let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

        var _was_specular: bool;
        let wi_and_weight = get_wi_and_weight(intersection, &_was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;

        let cur_attenuation = material.albedo * cos_brdf_over_wi_pdf;

        ray = Ray(intersection.position + intersection.normal * 0.0009 * select(1., -1., dot(wi, intersection.normal) < 0.), wi);

        light = light + (material.emittance * attenuation);
        attenuation = cur_attenuation * attenuation;
    }

    return light;
}

fn just_geo(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    var sample_info: PixelSample;

    let camera = PinpointCamera(FRAC_PI_4);
    var ray = pinpoint_generate_ray(camera, pixel, dimensions, uniforms.camera_position);

    for (var depth = 0; depth < 5; depth++) {
        var intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &intersection) {
            break;
        }
        let material = materials[intersection.material_idx];

        var was_specular = false;
        let wi_and_weight = get_wi_and_weight(intersection, &was_specular);
        let wi = wi_and_weight.xyz;

        if !was_specular {
            sample_info.normal = intersection.normal;
            sample_info.position = intersection.position;
            sample_info.depth = intersection.distance;
            break;
        }

        if depth == 0 {
            sample_info.albedo = material.albedo;
        }

        let offset = intersection.normal * 0.0009 * select(1., -1., dot(wi, intersection.normal) < 0.);
        ray = Ray(intersection.position + offset, wi);
    }

    return sample_info;
}

fn geo_and_rt(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    var sample_info: PixelSample;
    var hit_diffuse = false;

    let camera = PinpointCamera(FRAC_PI_4);
    var ray = pinpoint_generate_ray(camera, pixel, dimensions, uniforms.camera_position);

    var light = vec3<f32>(0.);
    var attenuation = vec3<f32>(1.);

    for (var depth = 0; depth < 5; depth++) {
        var intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &intersection) {
            break;
        }
        let material = materials[intersection.material_idx];

        var was_specular = false;
        let wi_and_weight = get_wi_and_weight(intersection, &was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;

        if !was_specular && !hit_diffuse {
            sample_info.normal = intersection.normal;
            sample_info.position = intersection.position;
            sample_info.depth = intersection.distance;
            hit_diffuse = true;
        }

        light += attenuation * material.emittance;

        if !was_specular && depth == 0 {
            //sample_info.direct_illum = sample_direct_lighting(intersection) * cos_brdf_over_wi_pdf;
            // light += attenuation * sample_direct_lighting(intersection);
        }

        if depth == 0 {
            sample_info.albedo = material.albedo;
            attenuation *= cos_brdf_over_wi_pdf;
        } else {
            attenuation *= material.albedo * cos_brdf_over_wi_pdf;
        }

        // let offset = intersection_oriented_normal(intersection) * 0.0009;
        let offset = intersection.normal * 0.0009 * select(1., -1., dot(wi, intersection.normal) < 0.);
        ray = Ray(intersection.position + offset, wi);
    }

    sample_info.rt = light;

    return sample_info;
}

fn a_trous(
    tex_coords: vec2<i32>, tex_dims: vec2<i32>, step_scale: i32,
    tex_from: texture_storage_2d<rgba8unorm, read_write>,
) -> vec3<f32> {
    /* abc
       bbc
       ccc */
    // js for testing stuff:
    // let g=f=>{let a=[];for(let y=-2;y<=2;y++){let b=[];for(let x=-2;x<=2;x++){b.push(f(x, y))}a.push(b)}return a}
    // let min = (x,y)=> x < y ? x : y;
    // let max = (x,y)=> x < y ? y : x;
    // let clamp = (v,lo,hi) => max(min(v, hi), lo);
    // let abs = v => v < 0 ? -v : v;
    // g((x,y)=>['a','b','c'][2 - clamp(2 - abs(x), 0, 2 - abs(y))])
    var kernel = array<f32, 3>(1./16., 1./4., 3./8.);

    let center_rt = textureLoad(tex_from, tex_coords).xyz;
    let center_geo = geometry_buffer[gb_idx_i(tex_coords)];

    var sum = vec3<f32>(0.);
    var kernel_sum = 0.;

    let σ_rt = 0.5;
    let σ_n  = 0.5;
    let σ_p  = 0.7;

    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            //let kernel_weight = kernel[(x + 2) + ((y + 2) * 5)];
            let kernel_weight = kernel[2 - clamp(2 - abs(x), 0, 2 - abs(y))];

            let offset = vec2<i32>(x, y) * step_scale;
            let cur_coords = clamp(tex_coords + offset, vec2<i32>(0), tex_dims);

            let sample_rt = textureLoad(tex_from, cur_coords).xyz;
            let dist_rt = distance(center_rt, sample_rt);
            let weight_rt = min(exp(-dist_rt / (σ_rt * σ_rt)), 1.);

            let sample_normal = ge_normal(geometry_buffer[gb_idx_i(cur_coords)]);
            let dist_normal = distance(ge_normal(center_geo), sample_normal);
            let weight_normal = min(exp(-dist_normal / (σ_n * σ_n)), 1.);

            /*let sample_pos = ge_position(geometry_buffer[gb_idx_i(cur_coords)]);
            let dist_pos = distance(ge_position(center_geo), sample_pos);
            let weight_pos = min(exp(-dist_pos / (σ_p * σ_p)), 1.);*/

            let sample_distance = ge_origin_distance(geometry_buffer[gb_idx_i(cur_coords)]);
            let dist_distance = abs(sample_distance - ge_origin_distance(center_geo));
            let weight_distance = min(exp(-dist_distance / (σ_p * σ_p)), 1.);

            let weight = kernel_weight * weight_rt * weight_normal * weight_distance;

            sum += weight * sample_rt.xyz;
            kernel_sum += weight;
        }
    }

    return sum / kernel_sum;
}

fn denoise_from_rt(pixel: vec2<u32>, texture_dimensions: vec2<u32>, stride: i32) {
    storageBarrier();
    textureStore(texture_denoise_0, pixel, vec4<f32>(a_trous(vec2<i32>(pixel), vec2<i32>(texture_dimensions), stride, texture_rt), 1.));
}

fn denoise_from_d0(pixel: vec2<u32>, texture_dimensions: vec2<u32>, stride: i32) {
    storageBarrier();
    textureStore(texture_denoise_1, pixel, vec4<f32>(a_trous(vec2<i32>(pixel), vec2<i32>(texture_dimensions), stride, texture_denoise_0), 1.));

}

fn denoise_from_d1(pixel: vec2<u32>, texture_dimensions: vec2<u32>, stride: i32) {
    storageBarrier();
    textureStore(texture_denoise_0, pixel, vec4<f32>(a_trous(vec2<i32>(pixel), vec2<i32>(texture_dimensions), stride, texture_denoise_1), 1.));
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

    let is_rt_pixel = (pixel.x % 2u) == 0u && (pixel.y % 2u) == 0u;

    var sample_sum: PixelSample;
    let samples = 8;
    for (var i = 0; i < samples; i++) {
        let sample = geo_and_rt(pixel, texture_dimensions);
        sample_sum = pixel_sample_add(sample_sum, sample);
        /*if is_rt_pixel {
            let sample = geo_and_rt(pixel, texture_dimensions);
            sample_sum = pixel_sample_add(sample_sum, sample);
        } else {
            let sample = just_geo(pixel, texture_dimensions);
            sample_sum = pixel_sample_add(sample_sum, sample);
        }*/
    }
    let sample = pixel_sample_div(sample_sum, f32(samples));

    /*if is_rt_pixel {
        textureStore(texture_rt, pixel + vec2<u32>(0u, 0u), vec4<f32>(sample.rt, 1.));
        textureStore(texture_rt, pixel + vec2<u32>(1u, 0u), vec4<f32>(sample.rt, 1.));
        textureStore(texture_rt, pixel + vec2<u32>(0u, 1u), vec4<f32>(sample.rt, 1.));
        textureStore(texture_rt, pixel + vec2<u32>(1u, 1u), vec4<f32>(sample.rt, 1.));
    }*/
    textureStore(texture_rt, pixel, vec4<f32>(sample.rt, 1.));

    geometry_buffer[gb_idx_u(pixel)].normal_and_depth = vec4<f32>(sample.normal, sample.depth);
    geometry_buffer[gb_idx_u(pixel)].albedo_and_origin_dist = vec4<f32>(sample.albedo, length(sample.position));
    geometry_buffer[gb_idx_u(pixel)].direct_illum = sample.direct_illum;

    denoise_from_rt(pixel, texture_dimensions, 1);
    denoise_from_d0(pixel, texture_dimensions, 2);
    denoise_from_d1(pixel, texture_dimensions, 3);
}
