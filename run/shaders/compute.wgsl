@group(0) @binding(0) var<uniform> uniforms: State;
@group(0) @binding(1) var texture_noise: texture_storage_2d<rgba32uint, read>;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var geo_texture_albedo: texture_2d<f32>;
@group(1) @binding(2) var geo_texture_pack_normal_depth: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(3) var geo_texture_pack_pos_dist: texture_storage_2d<rgba32float, read_write>;
@group(1) @binding(4) var geo_texture_object_index: texture_2d<u32>;
//@group(1) @binding(5) var texture_denoise_0: texture_storage_2d<rgba8unorm, read_write>;
//@group(1) @binding(6) var texture_denoise_1: texture_storage_2d<rgba8unorm, read_write>;

@group(2) @binding(0) var<storage> triangles: array<Triangle>;

fn collect_geo_u(coords: vec2<u32>) -> GeometryElement {
    let sample_albedo = textureLoad(geo_texture_albedo, coords, 0);
    let sample_normal_depth = textureLoad(geo_texture_pack_normal_depth, coords);
    let sample_pos_dist = textureLoad(geo_texture_pack_pos_dist, coords);
    let sample_object_index = textureLoad(geo_texture_object_index, coords, 0);

    return GeometryElement (
        sample_albedo.xyz,
        sample_normal_depth.xyz,
        sample_normal_depth.w,
        sample_pos_dist.xyz,
        sample_pos_dist.w,
        sample_object_index.r,
    );
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
    var attenuation = vec3<f32>(1.);
    var hit_diffuse = false;
    var have_diffuse_geo = false;

    var intersection = Intersection(
        length(geo_sample.position - uniforms.camera_position),
        geo_sample.position,
        geo_sample.normal,
        -normalize(geo_sample.position - uniforms.camera_position),
        triangles[geo_sample.object_index].material,
    );

    for (var depth = 0; depth < 4; depth++) {
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
    let samples = 40;
    for (var i = 0; i < samples; i++) {
        sample_sum = pixel_sample_add(sample_sum, new_cs(pixel, texture_dimensions, geo_sample));
    }
    let sample = pixel_sample_div(sample_sum, f32(samples));

    textureStore(texture_rt, pixel, vec4<f32>(sample.rt, 1.));
    textureStore(geo_texture_pack_normal_depth, pixel, vec4<f32>(sample.normal, geo_sample.depth));
    textureStore(geo_texture_pack_pos_dist, pixel, vec4<f32>(sample.position, length(geo_sample.position)));
}
