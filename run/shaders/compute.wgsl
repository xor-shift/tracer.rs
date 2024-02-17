struct MainUniform {
    width: u32,                 // 00..03
    height: u32,                // 04..07
    frame_no: u32,              // 08..0B
    current_instant: f32,       // 0C..0F
    seed_0: u32,                // 10..13
    seed_1: u32,                // 14..17
    seed_2: u32,                // 18..1B
    seed_3: u32,                // 1C..1F
    visualisation_mode: i32,    // 20..23
    camera_position: vec3<f32>, // 30..3B
}

@group(0) @binding(0) var<uniform> uniforms: MainUniform;
@group(0) @binding(1) var texture_noise: texture_storage_2d<rgba32uint, read>;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var geo_texture_albedo: texture_2d<f32>;
@group(1) @binding(2) var geo_texture_pack_normal_depth: texture_2d<f32>;
@group(1) @binding(3) var geo_texture_pack_pos_dist: texture_2d<f32>;
@group(1) @binding(4) var geo_texture_object_index: texture_2d<u32>;
@group(1) @binding(5) var texture_denoise_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(6) var texture_denoise_1: texture_storage_2d<rgba8unorm, read_write>;

@group(2) @binding(0) var<storage> triangles: array<Triangle>;

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
    Material(vec3<f32>(0., 0., 1.), vec3<f32>(0., 0., 12.), 0u, 0., 1.), // 8 light (blyue)
);

const CBL: vec3<f32> = vec3<f32>(-3.5, -3.5, -20.);
const CTR: vec3<f32> = vec3<f32>(3.5, 2.5, 20.);

/*
const NUM_TRIANGLES = 30u;
var<private> triangles: array<Triangle, NUM_TRIANGLES> = array<Triangle, NUM_TRIANGLES>(
    // light 1
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-3., 2.4, 15.), vec3<f32>(-1., 2.4, 15.), vec3<f32>(-1., 2.4, 11.25)), 6u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1., 2.4, 11.25), vec3<f32>(-3., 2.4, 11.25), vec3<f32>(-3., 2.4, 15.)), 6u),

    // light 2
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1., 2.4, 15.), vec3<f32>(3., 2.4, 15.), vec3<f32>(3., 2.4, 11.25)), 7u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(3., 2.4, 11.25), vec3<f32>(1., 2.4, 11.25), vec3<f32>(1., 2.4, 15.)), 7u),

    // light 3
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 8.25)), 8u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 12.)), 8u),

    // light 2
    //Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 11.25)), 0u),
    //Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 15.)), 0u),

    // mirror prism (bounding box: [-2.65, -2.5, 16.6], [-0.85, -0.7, 18.4])
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 16.6),
        vec3<f32>(-2.65, -2.5, 18.4),
        vec3<f32>(-0.85, -2.5, 18.4),
    ), 1u), // bottom 1
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 18.4),
        vec3<f32>(-0.85, -2.5, 16.6),
        vec3<f32>(-2.65, -2.5, 16.6),
    ), 1u), // bottom 2
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 18.4),
        vec3<f32>(-2.65, -2.5, 16.6),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.),
    ), 1u), // west
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 16.6),
        vec3<f32>(-0.85, -2.5, 16.6),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.),
    ), 1u), // south
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 16.6),
        vec3<f32>(-0.85, -2.5, 18.4),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.),
    ), 1u), // east
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 18.4),
        vec3<f32>(-2.65, -2.5, 18.4),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.),
    ), 1u), // north

    // glass prism (bounding box: [0.85, -2.3, 15.6], [2.65, -0.5, 17.4])
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-2.65, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-0.85, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
    ), 2u), // bottom 1
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-0.85, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-2.65, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
    ), 2u), // bottom 2
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-2.65, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.) + vec3<f32>(3.5, 0.2, -1.) + vec3<f32>(0., -0.3, 0.),
    ), 2u), // west
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-2.65, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-0.85, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.) + vec3<f32>(3.5, 0.2, -1.) + vec3<f32>(0., -0.3, 0.),
    ), 2u), // south
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 16.6) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-0.85, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.) + vec3<f32>(3.5, 0.2, -1.) + vec3<f32>(0., -0.3, 0.),
    ), 2u), // east
    Triangle(array<vec3<f32>, 3>(
        vec3<f32>(-0.85, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>(-2.65, -2.5, 18.4) + vec3<f32>(3.5, 0.2, -1.),
        vec3<f32>((-2.65 + -0.85) / 2., -0.7, (16.6 + 18.4) / 2.) + vec3<f32>(3.5, 0.2, -1.) + vec3<f32>(0., -0.3, 0.),
    ), 2u), // north

    // front wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z)), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), 3u),

    // back wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z)), 3u),

    // right wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), 5u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CTR.x, CTR.y, CTR.z)), 5u),

    // ceiling
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CTR.z), vec3<f32>(CTR.x, CTR.y, CBL.z)), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), 3u),

    // left wall
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CTR.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CBL.x, CBL.y, CBL.z)), 4u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CBL.z), vec3<f32>(CBL.x, CTR.y, CTR.z)), 4u),

    // floor
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CBL.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CTR.z), vec3<f32>(CTR.x, CBL.y, CBL.z)), 3u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(CTR.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CBL.z), vec3<f32>(CBL.x, CBL.y, CTR.z)), 3u),
);
*/

const NUM_EMISSIVE: u32 = 2u;
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
    direct_illum: vec3<f32>,
}

fn pixel_sample_add(lhs: PixelSample, rhs: PixelSample) -> PixelSample {
    return PixelSample(
        lhs.rt + rhs.rt,
        lhs.direct_illum + rhs.direct_illum,
    );
}

fn pixel_sample_div(lhs: PixelSample, divisor: f32) -> PixelSample {
    return PixelSample(
        lhs.rt / divisor,
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

fn new_cs(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    let geo_sample = collect_geo_u(pixel);
    var sample_info: PixelSample;

    var light = vec3<f32>(0.);
    var attenuation = vec3<f32>(1.);
    var hit_diffuse = false;

    var intersection = Intersection(
        length(geo_sample.position - uniforms.camera_position),
        geo_sample.position,
        geo_sample.normal,
        -normalize(geo_sample.position - uniforms.camera_position),
        triangles[geo_sample.object_index].material,
    );

    //intersection.position += intersection_oriented_normal(intersection) * 1;

    for (var depth = 0; depth < 4; depth++) {
        let material = materials[intersection.material_idx];

        var was_specular = false;
        let wi_and_weight = get_wi_and_weight(intersection, &was_specular);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;

        light += attenuation * material.emittance;

        if depth >= 1 && !hit_diffuse && !was_specular {
            hit_diffuse = true;
            light += material.albedo * attenuation * sample_direct_lighting(intersection);
        }

        if depth == 0 {
            attenuation *= cos_brdf_over_wi_pdf;
        } else {
            attenuation *= material.albedo * cos_brdf_over_wi_pdf;
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

fn check_discrapency(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    let geo_sample = collect_geo_u(pixel);

    var initial_intersection = Intersection(
        length(geo_sample.position - uniforms.camera_position),
        geo_sample.position,
        geo_sample.normal,
        -normalize(geo_sample.position - uniforms.camera_position),
        triangles[geo_sample.object_index].material,
    );
    
    let camera = PinpointCamera(FRAC_PI_4);
    //var ray = pinpoint_generate_ray(camera, pixel, dimensions, uniforms.camera_position);
    let direction = normalize(geo_sample.position - uniforms.camera_position);
    var ray = Ray(uniforms.camera_position, direction);

    var intersection: Intersection = dummy_intersection(ray);
    if !intersect_stuff(ray, &intersection) {
        return PixelSample(vec3<f32>(1., 0., 0.), vec3<f32>(0.));
    }

    //return PixelSample(vec3<f32>(initial_intersection.distance - intersection.distance), vec3<f32>(0.));
    //return PixelSample(vec3<f32>(length(initial_intersection.position - intersection.position) * 10.), vec3<f32>(0.));
    //return PixelSample(vec3<f32>(length(initial_intersection.normal - intersection.normal)), vec3<f32>(0.));
    //return PixelSample(vec3<f32>(length(initial_intersection.wo - intersection.wo)), vec3<f32>(0.));
    return PixelSample(vec3<f32>(f32(initial_intersection.material_idx - intersection.material_idx)), vec3<f32>(0.));
}

fn geo_and_rt(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    let geo_sample = collect_geo_u(pixel);
    let direction = normalize(geo_sample.position - uniforms.camera_position);

    var sample_info: PixelSample;
    var hit_diffuse = false;    
    var ray = Ray(uniforms.camera_position, direction);

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

        light += attenuation * material.emittance;

        if depth == 0 {
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
    let center_geo = collect_geo_i(tex_coords);

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

            let sample_normal = collect_geo_i(cur_coords).normal;
            let dist_normal = distance(center_geo.normal, sample_normal);
            let weight_normal = min(exp(-dist_normal / (σ_n * σ_n)), 1.);

            /*let sample_pos = ge_position(geometry_buffer[gb_idx_i(cur_coords)]);
            let dist_pos = distance(ge_position(center_geo), sample_pos);
            let weight_pos = min(exp(-dist_pos / (σ_p * σ_p)), 1.);*/

            let sample_distance = collect_geo_i(cur_coords).distance_from_origin;
            let dist_distance = abs(sample_distance - center_geo.distance_from_origin);
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

    var sample_sum: PixelSample;
    let samples = 1;
    for (var i = 0; i < samples; i++) {
        //let sample = check_discrapency(pixel, texture_dimensions);
        //let sample = geo_and_rt(pixel, texture_dimensions);
        let sample = new_cs(pixel, texture_dimensions);
        sample_sum = pixel_sample_add(sample_sum, sample);
    }
    let sample = pixel_sample_div(sample_sum, f32(samples));

    textureStore(texture_rt, pixel, vec4<f32>(sample.rt, 1.));

    let geo_sample = collect_geo_u(pixel);
    let tri = triangles[geo_sample.object_index];
    let material = materials[tri.material];

    //textureStore(texture_rt, pixel, vec4<f32>(tri.vertices[0].xyz, 1.));
    //textureStore(texture_rt, pixel, vec4<f32>(material.emittance, 1.));

    denoise_from_rt(pixel, texture_dimensions, 1);
    denoise_from_d0(pixel, texture_dimensions, 2);
    denoise_from_d1(pixel, texture_dimensions, 3);
}
