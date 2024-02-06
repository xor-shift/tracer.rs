@group(0) @binding(0) var<uniform> uniforms: MainUniform;
@group(0) @binding(1) var texture_noise: texture_storage_2d<rgba32uint, read>;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var texture_normal: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(2) var texture_pos: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(3) var<storage, read_write> geometry_buffer: array<GeometryElement>;

struct Material {
    albedo: vec3<f32>,
    emittance: vec3<f32>,
    // 0 -> diffuse/specular, 1 -> dielectric
    mat_type: u32,
    reflectivity: f32,
    index: f32,
}

// PT modelled after https://youtu.be/FewqoJjHR0A?t=1112

var<private> materials: array<Material, 6> = array<Material, 6>(
    Material(vec3<f32>(0.)  , vec3<f32>(12.), 0u, 0.  , 1. ),         // 0 light
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 0u, 1., 1. ),         // 1 mirror
    Material(vec3<f32>(0.99), vec3<f32>(0.) , 1u, 0.  , 1.7),         // 2 glass
    Material(vec3<f32>(0.75), vec3<f32>(0.) , 0u, 0.  , 1. ),         // 3 white
    Material(vec3<f32>(0.75, 0.25, 0.25), vec3<f32>(0.), 0u, 0., 1.), // 4 red
    Material(vec3<f32>(0.25, 0.25, 0.75), vec3<f32>(0.), 0u, 0., 1.), // 5 blue
);

var<private> spheres: array<Sphere, 2> = array<Sphere, 2>(
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

var<private> triangles: array<Triangle, 14> = array<Triangle, 14>(
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

    // light
    Triangle(array<vec3<f32>, 3>(vec3<f32>(-1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 15.), vec3<f32>(1.25, 2.4, 11.25)), vec2<f32>(0.), vec2<f32>(1.), 0u),
    Triangle(array<vec3<f32>, 3>(vec3<f32>(1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 11.25), vec3<f32>(-1.25, 2.4, 15.)), vec2<f32>(0.), vec2<f32>(1.), 0u),
);

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

    for (var i = 0u; i < 2u; i++) {
        var cur: Intersection;
        if sphere_intersect(spheres[i], ray, intersection.distance, &cur) {
            intersected = true;
            intersection = pick_intersection(intersection, cur);
        }
    }

    for (var i = 0u; i < 14u; i++) {
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

fn sample_pixel(camera: PinpointCamera, pixel: vec2<u32>, dimensions: vec2<u32>) -> array<vec3<f32>, 3> {
    //var ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0., 0., uniforms.current_instant * 2.));
    var ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0.));

    var attenuation = vec3<f32>(1.);
    var light = vec3<f32>(0.);

    // TODO: refactor this
    // i am writing this while severely sleepy
    var normal: vec3<f32>;
    var pos: vec3<f32>;

    for (var i = 0u; i < 4u; i++) {
        var intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &intersection) {
            break;
        }

        if i == 0u {
            normal = intersection.normal;
            pos = intersection.position;
        }

        let material = materials[intersection.material_idx];

        let going_in = dot(ray.direction, intersection.normal) < 0.;
        let oriented_normal = select(-intersection.normal, intersection.normal, going_in);

        var wi: vec3<f32>;
        var cos_brdf_over_wi_pdf: f32;
        
        if material.mat_type == 0u {
            let refl_sample = rand();
            if refl_sample <= material.reflectivity {
                wi = reflect(intersection.wo, oriented_normal);
                cos_brdf_over_wi_pdf = material.reflectivity;
            } else {
                let sphere_sample = sample_sphere_3d();
                wi = select(sphere_sample, -sphere_sample, dot(sphere_sample, intersection.normal) < 0.);
                let wi_pdf = FRAC_1_PI * 0.5 * (1. - material.reflectivity);
                let brdf = FRAC_1_PI * 0.5;
                cos_brdf_over_wi_pdf = dot(wi, intersection.normal) * brdf / wi_pdf;
            }
        } else if material.mat_type == 1u {
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

        ray = Ray(intersection.position + intersection.normal * 0.0009 * select(1., -1., dot(wi, intersection.normal) < 0.), wi);
        light = light + material.emittance * attenuation;

        let cur_attenuation = material.albedo * cos_brdf_over_wi_pdf;
        attenuation = cur_attenuation * attenuation;
    }

    return array<vec3<f32>, 3>(light, normal, pos);
}

fn actual_cs(pixel: vec2<u32>, dimensions: vec2<u32>) -> array<vec3<f32>, 3> {
    let camera = PinpointCamera(FRAC_PI_4);

    let samples = 40u;
    var res = array<vec3<f32>, 3>(vec3<f32>(0.), vec3<f32>(0.), vec3<f32>(0.));
    for (var i = 0u; i < samples; i++) {
        let tmp = sample_pixel(camera, pixel, dimensions);
        res[0] += tmp[0];
        res[1] += tmp[1];
        res[2] += tmp[2];
    }
    res[0] /= f32(samples);
    res[1] /= f32(samples);
    res[2] /= f32(samples);

    //let w_prev = previous_value * f32(uniforms.frame_no) / f32(uniforms.frame_no + 1u);
    //let w_new  = vec4<f32>(res / f32(uniforms.frame_no + 1u), 1.);

    //return array<vec3<f32>, 3>(w_prev + w_new, vec3<f32>(0.), vec3<f32>(0.));
    
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

    let out_color = actual_cs(pixel, texture_dimensions);

    //geometry_buffer[gb_idx_u(pixel)].pos = out_color[2];
    geometry_buffer[gb_idx_u(pixel)].normal_and_depth = vec4<f32>(out_color[1], 1.);
    geometry_buffer[gb_idx_u(pixel)].albedo = vec4<f32>(out_color[0], 1.);

    textureStore(texture_rt, vec2<i32>(pixel), vec4<f32>(out_color[0], 1.));
    textureStore(texture_normal, vec2<i32>(pixel), vec4<f32>(out_color[1], 1.));
    textureStore(texture_pos, vec2<i32>(pixel), vec4<f32>(out_color[2], 1.));
}
