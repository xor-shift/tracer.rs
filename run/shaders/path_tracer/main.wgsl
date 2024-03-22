@group(0) @binding(0) var<uniform> state: State;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var texture_geo: texture_storage_2d_array<rgba32uint, read_write>;
@group(1) @binding(2) var texture_rt_old: texture_2d<f32>;
@group(1) @binding(3) var texture_geo_old: texture_2d_array<u32>;

const QM_NULL = Material(0, vec4<f32>(0.));
const QM_RED = Material(1, vec4<f32>(0.75, 0., 0., 0.));
const QM_BLUE = Material(1, vec4<f32>(0., 0., 0.75, 0.));
const QM_WHITE = Material(1, vec4<f32>(0.75, 0.75, 0.75, 0.));
const QM_LIGHT = Material(2, vec4<f32>(12., 12., 12., 0.));

const QPM_NULL  = vec2<u32>(0x00000000u, 0x00000000u);
const QPM_RED   = vec2<u32>(0x01C00000u, 0x00000000u);
const QPM_BLUE  = vec2<u32>(0x010000C0u, 0x00000000u);
const QPM_WHITE = vec2<u32>(0x01C0C0C0u, 0x00000000u);
const QPM_LIGHT = vec2<u32>(0x02010101u, 0xFFFF0000u);

var<private> cornell_grid_front: array<Material, 64> = array<Material, 64>(
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,

    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_RED , QM_NULL , QM_NULL , QM_RED,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,

    QM_NULL, QM_NULL , QM_NULL , QM_NULL,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_NULL, QM_NULL , QM_NULL , QM_NULL,
);

/*
var<private> cornell_grid_front_packed: array<vec2<u32>, 64> = array<vec2<u32>, 64>(
    QPM_NULL, QPM_WHITE, QPM_WHITE, QPM_NULL,
    QPM_RED , QPM_NULL , QPM_NULL , QPM_BLUE,
    QPM_RED , QPM_NULL , QPM_NULL , QPM_BLUE,
    QPM_NULL, QPM_WHITE, QPM_WHITE, QPM_NULL,
    
    QPM_NULL, QPM_WHITE, QPM_WHITE, QPM_NULL,
    QPM_RED , QPM_NULL , QPM_NULL , QPM_BLUE,
    QPM_RED , QPM_NULL , QPM_NULL , QPM_BLUE,
    QPM_NULL, QPM_WHITE, QPM_WHITE, QPM_NULL,

    QPM_NULL, QPM_WHITE, QPM_WHITE, QPM_NULL,
    QPM_RED , QPM_NULL , QPM_NULL , QPM_BLUE,
    QPM_RED , QPM_NULL , QPM_NULL , QPM_RED,
    QPM_NULL, QPM_WHITE, QPM_WHITE, QPM_NULL,

    QPM_NULL, QPM_NULL , QPM_NULL , QPM_NULL,
    QPM_NULL, QPM_WHITE, QPM_WHITE, QPM_NULL,
    QPM_NULL, QPM_WHITE, QPM_WHITE, QPM_NULL,
    QPM_NULL, QPM_NULL , QPM_NULL , QPM_NULL,
);
*/

var<private> cornell_grid_back: array<Material, 64> = array<Material, 64>(
    QM_NULL, QM_NULL , QM_NULL , QM_NULL,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_NULL, QM_NULL , QM_NULL , QM_NULL,

    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,

    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_RED , QM_NULL , QM_NULL , QM_BLUE,
    QM_NULL, QM_WHITE, QM_WHITE, QM_NULL,
);

var<private> color_grid: array<Material, 64> = array<Material, 64>(
    Material(1, vec4<f32>(0., 0. , 0., 0.)), Material(1, vec4<f32>(.33, 0. , 0., 0.)), Material(1, vec4<f32>(.66, 0. , 0., 0.)), Material(1, vec4<f32>(1., 0. , 0., 0.)),
    Material(1, vec4<f32>(0., .33, 0., 0.)), QM_NULL                                 , QM_NULL                                 , Material(1, vec4<f32>(1., .33, 0., 0.)),
    Material(1, vec4<f32>(0., .66, 0., 0.)), QM_NULL                                 , QM_NULL                                 , Material(1, vec4<f32>(1., .66, 0., 0.)),
    Material(1, vec4<f32>(0., 1. , 0., 0.)), Material(1, vec4<f32>(.33, 1. , 0., 0.)), Material(1, vec4<f32>(.66, 1. , 0., 0.)), Material(1, vec4<f32>(1., 1. , 0., 0.)),

    Material(1, vec4<f32>(0., 0. , .33, 0.)), QM_NULL, QM_NULL, Material(1, vec4<f32>(1., 0. , .33, 0.)),
    QM_NULL                                 , QM_NULL, QM_NULL, QM_NULL,
    QM_NULL                                 , QM_NULL, QM_NULL, QM_NULL,
    Material(1, vec4<f32>(0., 1. , .33, 0.)), QM_NULL, QM_NULL, Material(1, vec4<f32>(1., 1. , .33, 0.)),

    Material(1, vec4<f32>(0., 0. , .66, 0.)), QM_NULL, QM_NULL, Material(1, vec4<f32>(1., 0. , .66, 0.)),
    QM_NULL                                 , QM_NULL, QM_NULL, QM_NULL,
    QM_NULL                                 , QM_NULL, QM_NULL, QM_NULL,
    Material(1, vec4<f32>(0., 1. , .66, 0.)), QM_NULL, QM_NULL, Material(1, vec4<f32>(1., 1. , .66, 0.)),

    Material(1, vec4<f32>(0., 0. , 1., 0.)), Material(1, vec4<f32>(.33, 0. , 1., 0.)), Material(1, vec4<f32>(.66, 0. , 1., 0.)), Material(1, vec4<f32>(1., 0. , 1., 0.)),
    Material(1, vec4<f32>(0., .33, 1., 0.)), QM_NULL                                 , QM_NULL                                 , Material(1, vec4<f32>(1., .33, 1., 0.)),
    Material(1, vec4<f32>(0., .66, 1., 0.)), QM_NULL                                 , QM_NULL                                 , Material(1, vec4<f32>(1., .66, 1., 0.)),
    Material(1, vec4<f32>(0., 1. , 1., 0.)), Material(1, vec4<f32>(.33, 1. , 1., 0.)), Material(1, vec4<f32>(.66, 1. , 1., 0.)), Material(1, vec4<f32>(1., 1. , 1., 0.)),
);

fn intersect_grid(ray: Ray, inout_intersection: ptr<function, Intersection>, grid: ptr<private, array<Material, 64>>, map_bounds_min: vec3<f32>, map_bounds_max: vec3<f32>) -> bool {
    var ret_intersection = *inout_intersection;
    var intersected = false;

    let per_step_delta = (map_bounds_max - map_bounds_min) / 4.;

    for (var z_step = 0; z_step < 4; z_step++) {
        for (var y_step = 0; y_step < 4; y_step++) {
            for (var x_step = 0; x_step < 4; x_step++) {
                let cur_step = vec3<i32>(x_step, y_step, z_step);
                let cur_index = x_step + y_step * 4 + z_step * 16;
                
                let cur_min = vec3<f32>(cur_step) * per_step_delta * 0.999 + map_bounds_min;
                let cur_max = vec3<f32>(cur_step + vec3<i32>(1)) * per_step_delta * 1.001 + map_bounds_min;

                let cur_data = (*grid)[cur_index];

                if cur_data.mat_type == 0 {
                    continue;
                }

                let cur_box = Box(
                    /* min */ cur_min,
                    /* max */ cur_max,
                    /* mat */ cur_data,
                );

                let cur_intersected = box_intersect(cur_box, ray, &ret_intersection);
                intersected = intersected || cur_intersected;
            }
        }
    }

    if intersected {
        *inout_intersection = ret_intersection;
    }

    return intersected;
}

@compute @workgroup_size(8, 8, 1) fn cs_main(
    @builtin(global_invocation_id)   global_id: vec3<u32>,
    @builtin(workgroup_id)           workgroup_id: vec3<u32>,
    @builtin(local_invocation_id)    local_id:  vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let pixel = global_id.xy;

    let ray = ray_from_pixel(pixel, state);
    var intersection = intersecton_new_dummy();
    var intersected = false;

    intersected = intersected || intersect_grid(ray, &intersection, &color_grid, vec3<f32>(-1, -2.5, 10.), vec3<f32>(1., -.5, 12.));
    intersected = intersected || intersect_grid(ray, &intersection, &cornell_grid_back, vec3<f32>(-6., -6., -6.), vec3<f32>(6., 6., 6.));
    intersected = intersected || intersect_grid(ray, &intersection, &cornell_grid_front, vec3<f32>(-6., -6., 6.), vec3<f32>(6., 6., 18.));

    if intersected {
        textureStore(texture_rt, pixel, vec4<f32>(intersection.material.data.xyz * dot(intersection.normal, -ray.direction), 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(vec3<f32>(dot(intersection.normal, -ray.direction)), 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(abs(intersection.normal), 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(intersection.material.data.xyz, 1.));
    } else {
        textureStore(texture_rt, pixel, vec4<f32>(vec2<f32>(local_id.xy) / vec2<f32>(8., 8.), 0., 1.));
    }
}
