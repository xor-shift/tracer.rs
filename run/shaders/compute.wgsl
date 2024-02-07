@group(0) @binding(0) var<uniform> uniforms: MainUniform;
@group(0) @binding(1) var texture_noise: texture_storage_2d<rgba32uint, read>;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var<storage, read_write> geometry_buffer: array<GeometryElement>;

struct Material {
    albedo: vec3<f32>,
    emittance: vec3<f32>,
    // 0 -> diffuse, 1 -> perfect mirror, 2 -> dielectric, 3 -> glossy (NYI)
    mat_type: u32,
    glossiness: f32,
    index: f32,
}

var<private> materials: array<Material, 9> = array<Material, 9>(
    Material(vec3<f32>(0.)  , vec3<f32>(12.), 0u, 0., 1. ),         // 0 light
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 1u, 1., 1. ),         // 1 mirror
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 2u, 0., 1.7),         // 2 glass
    Material(vec3<f32>(0.75), vec3<f32>(0.) , 0u, 0., 1. ),         // 3 white
    Material(vec3<f32>(0.75, 0.25, 0.25), vec3<f32>(0.), 0u, 0., 1.), // 4 red
    Material(vec3<f32>(0.25, 0.25, 0.75), vec3<f32>(0.), 0u, 0., 1.), // 5 blue
    Material(vec3<f32>(0.), vec3<f32>(12., 0., 0.), 0u, 0., 1.), // 6 light (red)
    Material(vec3<f32>(0.), vec3<f32>(0., 12., 0.), 0u, 0., 1.), // 7 light (green)
    Material(vec3<f32>(0.), vec3<f32>(0., 0., 12.), 0u, 0., 1.), // 8 light (blyue)
);

const NUM_SPHERES: u32 = 2u;
var<private> spheres: array<Sphere, NUM_SPHERES> = array<Sphere, 2>(
    Sphere(vec3<f32>(-1.75 , -2.5 + 0.9      , 17.5  ), .9     , 3u), // mirror
    Sphere(vec3<f32>(1.75  , -2.5 + 0.9 + 0.2, 16.5  ), .9     , 3u), // glass
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

const NUM_TRIANGLES = 14u;
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

    /*
    // light 1
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-3., 2.4, 15.), vec3<f32>(-1., 2.4, 15.), vec3<f32>(-1., 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 6u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1., 2.4, 11.25), vec3<f32>(-3., 2.4, 11.25), vec3<f32>(-3., 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 6u),

    // light 2
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1., 2.4, 15.), vec3<f32>(3., 2.4, 15.), vec3<f32>(3., 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 7u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(3., 2.4, 11.25), vec3<f32>(1., 2.4, 11.25), vec3<f32>(1., 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 7u),

    // light 3
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 12.), vec3<f32>(1.25, 2.4, 8.25)), vec2<f32>(0.), vec2<f32>(1.), 8u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 8.25), vec3<f32>(-1.25, 2.4, 12.)), vec2<f32>(0.), vec2<f32>(1.), 8u),
    */

    // light 2
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 0u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 0u),
);

const NUM_EMISSIVE: u32 = 2u;
var<private> emissive_triangles: array<u32, NUM_EMISSIVE> = array<u32, NUM_EMISSIVE>(12u, 13u/*, 14u, 15u, 16u, 17u*/);

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
        sum += abs(dot(intersection.normal, vector_towards_light)) * p * (1. / sample.pdf);
    }

    return sum;
}

fn get_wi_and_weight(intersection: Intersection) -> vec4<f32> {
    let material = materials[intersection.material_idx];
    let going_in = dot(intersection.wo, intersection.normal) > 0.;
    let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

    var wi: vec3<f32>;
    var cos_brdf_over_wi_pdf: f32;

    if material.mat_type == 0u {
        /*var sample_probability: f32;
        let sphere_sample = sample_sphere_3d(&sample_probability);
        wi = select(sphere_sample, -sphere_sample, dot(sphere_sample, oriented_normal) < 0.);
        sample_probability *= 2.;*/

        var sample_probability: f32;
        let importance_sample = sample_cos_hemisphere_3d(&sample_probability);
        wi = intersection.refl_to_surface * importance_sample;

        let brdf = FRAC_1_PI * 0.5;
        cos_brdf_over_wi_pdf = dot(wi, oriented_normal) * brdf / sample_probability;
    } else if material.mat_type == 1u {
        wi = reflect(intersection.wo, oriented_normal);
        cos_brdf_over_wi_pdf = 1.;
    } else if material.mat_type == 2u {
        cos_brdf_over_wi_pdf = 1.;

        let transmittant_index = select(1., material.index, going_in);
        let incident_index     = select(material.index, 1., going_in);

        var refraction: vec3<f32>;
        var p_refraction: f32;
        if refract(
            intersection.wo, oriented_normal,
            incident_index, transmittant_index,
            &p_refraction, &refraction
        ) {
            let generated = rand();

            if generated > p_refraction {
                wi = refraction;
            } else {
                wi = reflect(intersection.wo, oriented_normal);
            }
        } else {
            wi = reflect(intersection.wo, oriented_normal);
        }
    }

    return vec4<f32>(wi, cos_brdf_over_wi_pdf);
}

struct PixelSample {
    rt: vec3<f32>,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    position: vec3<f32>,
    depth: f32,
}

fn pixel_sample_add(lhs: PixelSample, rhs: PixelSample) -> PixelSample {
    return PixelSample(
        lhs.rt + rhs.rt,
        lhs.albedo + rhs.albedo,
        lhs.normal + rhs.normal,
        lhs.position + rhs.position,
        lhs.depth + rhs.depth,
    );
}

fn pixel_sample_div(lhs: PixelSample, divisor: f32) -> PixelSample {
    return PixelSample(
        lhs.rt / divisor,
        lhs.albedo / divisor,
        lhs.normal / divisor,
        lhs.position / divisor,
        lhs.depth / divisor,
    );
}

fn sample_pixel(camera: PinpointCamera, pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    //var ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0., 0., uniforms.current_instant * 2.));
    var ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0.));

    var attenuation = vec3<f32>(1.);
    var light = vec3<f32>(0.);

    var sample_info: PixelSample;

    for (var i = 0u; i < 4u; i++) {
        var intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &intersection) {
            break;
        }
        let material = materials[intersection.material_idx];

        if i == 0u {
            sample_info.albedo = material.albedo;
            sample_info.normal = intersection.normal;
            sample_info.position = intersection.position;
            sample_info.depth = intersection.distance;
        }

        let explicit_lighting = sample_direct_lighting(intersection);

        let going_in = dot(ray.direction, intersection.normal) < 0.;
        let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

        let wi_and_weight = get_wi_and_weight(intersection);
        let wi = wi_and_weight.xyz;
        let cos_brdf_over_wi_pdf = wi_and_weight.w;

        let cur_attenuation = material.albedo * cos_brdf_over_wi_pdf;

        ray = Ray(intersection.position + intersection.normal * 0.0009 * select(1., -1., dot(wi, intersection.normal) < 0.), wi);

        light = light + (material.emittance * attenuation;
        attenuation = cur_attenuation * attenuation;
    }

    sample_info.rt = light;

    return sample_info;
}

/*fn sample_pixel_new(camera: PinpointCamera, pixel: vec2<u32>, dimensions: vec2<u32>) -> array<vec3<f32>, 3> {
    var ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0.));

    var intersection: Intersection = dummy_intersection(ray);
    if !intersect_stuff(ray, &intersection) {
        return array<vec3<f32>, 3>(vec3<f32>(0.), vec3<f32>(0.), vec3<f32>(0.));
    }

    let material = materials[intersection.material_idx];

    let explicit_lighting = sample_direct_lighting(intersection);

    return array<vec3<f32>, 3>(explicit_lighting * material.albedo, intersection.normal, intersection.position);
}*/

fn actual_cs(pixel: vec2<u32>, dimensions: vec2<u32>) -> PixelSample {
    let camera = PinpointCamera(FRAC_PI_4);

    let samples = 4u;
    var res: PixelSample;
    for (var i = 0u; i < samples; i++) {
        let tmp = sample_pixel(camera, pixel, dimensions);
        res = pixel_sample_add(res, tmp);
    }
    res = pixel_sample_div(res, f32(samples));

    return res;
}
 
@compute @workgroup_size(8, 8) fn cs_main(
    @builtin(global_invocation_id)   global_id: vec3<u32>,
    @builtin(workgroup_id)           workgroup_id: vec3<u32>,
    @builtin(local_invocation_id)    local_id:  vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let pixel = global_id.xy;
    let texture_dimensions = textureDimensions(texture_rt);

    setup_rng(workgroup_id.xy, texture_dimensions, local_idx);

    let sample = actual_cs(pixel, texture_dimensions);

    textureStore(texture_rt, pixel, vec4<f32>(sample.rt, 1.));
    geometry_buffer[gb_idx_u(pixel)].normal_and_depth = vec4<f32>(sample.normal, sample.depth);
    geometry_buffer[gb_idx_u(pixel)].albedo = vec4<f32>(sample.albedo, 1.);
    geometry_buffer[gb_idx_u(pixel)].position = vec4<f32>(sample.position, 1.);
}
