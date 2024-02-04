struct MainUniform {
    // at frame no 0, texture 1 should be used and texture 0 should be drawn on
    frame_no: u32,
    current_instant: f32,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
};

@group(0) @binding(0) var<uniform> uniforms: MainUniform;
@group(1) @binding(0) var texture_0: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var texture_1: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(2) var texture_noise: texture_storage_2d<rgba32uint, read>;

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

var<private> spheres: array<Sphere, 9> = array<Sphere, 9>(
    Sphere(vec3<f32>(-1.75 , -2.5 + 0.9      , 17.5  ), .9     , 1u), // mirror
    Sphere(vec3<f32>(1.75  , -2.5 + 0.9 + 0.2, 16.5  ), .9     , 2u), // glass
    Sphere(vec3<f32>(0.    , 42.499          , 15.   ), 40.    , 0u), // light
    Sphere(vec3<f32>(0.    , 0.              , -5000.), 4980.  , 3u), // front wall
    Sphere(vec3<f32>(0.    , 0.              , 5000. ), 4980.  , 3u), // backwall
    Sphere(vec3<f32>(5000. , 0.              , 0.    ), 4996.5 , 5u), // right wall
    Sphere(vec3<f32>(0.    , 5000.           , 5.    ), 4997.5 , 3u), // ceiling
    Sphere(vec3<f32>(-5000., 0.              , 0.    ), 4996.5 , 4u), // left wall
    Sphere(vec3<f32>(0.    , -5000.          , 5.    ), 4997.5 , 3u), // floor
);

fn intersect_stuff(ray: Ray, out_intersection: ptr<function, Intersection>) -> bool {
    var intersection: Intersection = *out_intersection;

    var intersected = false;
    for (var i = 0u; i < 9u; i++) {
        var cur: Intersection;
        if sphere_intersect(spheres[i], ray, intersection.distance, &cur) {
            intersected = true;
            intersection = pick_intersection(intersection, cur);
        }
    }

    *out_intersection = intersection;

    return intersected;
}

fn sample_pixel(camera: PinpointCamera, pixel: vec2<u32>, dimensions: vec2<u32>) -> vec3<f32> {
    //var ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0., 0., uniforms.current_instant * 2.));
    var ray = pinpoint_generate_ray(camera, pixel, dimensions, vec3<f32>(0.));

    var attenuation = vec3<f32>(1.);
    var light = vec3<f32>(0.);

    for (var i = 0u; i < 4u; i++) {
        var intersection: Intersection = dummy_intersection(ray);
        if !intersect_stuff(ray, &intersection) {
            break;
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

    return light;
}

fn actual_cs(pixel: vec2<u32>, dimensions: vec2<u32>, previous_value: vec4<f32>) -> vec4<f32> {
    let camera = PinpointCamera(FRAC_PI_4);

    let samples = 6u;
    var res = vec3<f32>(0.);
    for (var i = 0u; i < samples; i++) {
        res += sample_pixel(camera, pixel, dimensions);
    }
    res /= f32(samples);

    let w_prev = previous_value * f32(uniforms.frame_no) / f32(uniforms.frame_no + 1u);
    let w_new  = vec4<f32>(res / f32(uniforms.frame_no + 1u), 1.);

    return w_prev + w_new;
}
 
@compute @workgroup_size(8, 8) fn cs_main(
    @builtin(global_invocation_id)   global_id: vec3<u32>,
    @builtin(workgroup_id)           workgroup_id: vec3<u32>,
    @builtin(local_invocation_id)    local_id:  vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let pixel = global_id.xy;
    let texture_selection = select(0u, 1u, uniforms.frame_no % 2u == 0u);
    let texture_dimensions = select(textureDimensions(texture_0), textureDimensions(texture_1), texture_selection == 0u);

    setup_rng(workgroup_id.xy, texture_dimensions, local_idx);
    // workgroupBarrier();

    /*rng.state[0] = rng_next(&rng);
    rng.state[1] = rng_next(&rng);
    rng.state[2] = rng_next(&rng);
    rng.state[3] = rng_next(&rng);*/

    var previous_value: vec4<f32>;
    if texture_selection == 0u {
        previous_value = textureLoad(texture_1, vec2<i32>(pixel));
    } else {
        previous_value = textureLoad(texture_0, vec2<i32>(pixel));
    }

    let out_color = actual_cs(pixel, texture_dimensions, previous_value);

    if texture_selection == 0u {
        textureStore(texture_0, vec2<i32>(pixel), out_color);
    } else {
        textureStore(texture_1, vec2<i32>(pixel), out_color);
    }
}
