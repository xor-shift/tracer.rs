@group(0) @binding(0) var<uniform> state: State;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var texture_geo: texture_storage_2d_array<rgba32uint, read_write>;
@group(1) @binding(2) var texture_rt_old: texture_2d<f32>;
@group(1) @binding(3) var texture_geo_old: texture_2d_array<u32>;

struct HitMissNode {
    path_pack: vec2<u32>,
    link_hit: u32,
    link_miss: u32,
    material_pack: vec2<u32>,
}

struct SceneTree {
    outer_extents: vec3<u32>,
    padding: u32,
    nodes: array<HitMissNode>,
}

@group(2) @binding(0) var<storage> scene_tree: SceneTree;

/*
fn intersect_grid(ray: Ray, inout_intersection: ptr<function, Intersection>) -> bool {
    var ret_intersection = *inout_intersection;
    var intersected = false;

    let size_z = arrayLength(&chunk.cubes) / (chunk.size_x * chunk.size_y);
    let size = vec3<u32>(chunk.size_x, chunk.size_y, size_z);

    let chunk_bound = chunk.max - chunk.min;
    let per_step_delta = chunk_bound / vec3<f32>(size);

    for (var z_step = 0u; z_step < size.z; z_step++) {
        for (var y_step = 0u; y_step < size.y; y_step++) {
            for (var x_step = 0u; x_step < size.x; x_step++) {
                let cur_step = vec3<i32>(i32(x_step), i32(y_step), i32(z_step));
                let cur_index = x_step + y_step * size.x + z_step * (size.y * size.x);
                
                let cur_min = vec3<f32>(cur_step) * per_step_delta * 0.999 + chunk.min;
                let cur_max = vec3<f32>(cur_step + vec3<i32>(1)) * per_step_delta * 1.001 + chunk.min;

                let cur_packed_material = chunk.cubes[cur_index];
                let cur_material = material_unpack(cur_packed_material);

                if cur_material.mat_type == 0 {
                    continue;
                }

                let cur_box = Box(
                    /* min */ cur_min,
                    /* max */ cur_max,
                    /* mat */ cur_material,
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
*/

fn compute_extent(original: array<vec3<f32>, 2>, path_pack: vec2<u32>) -> array<vec3<f32>, 2> {
    var ret = original;
    var working_pack = path_pack;

    loop {
        let cur = working_pack[0] & 0xFu;
        
        let temp = working_pack[1] & 0xFu;
        working_pack[0] >>= 4u;
        working_pack[1] >>= 4u;
        working_pack[0] |= temp << 28u;

        if (cur & 0x8) == 0 {
            break;
        }

        let oct_size = (ret[1] - ret[0]) / 2;

        let x_p = (cur & 1) != 0;
        let y_p = (cur & 2) != 0;
        let z_p = (cur & 4) != 0;

        let new_min = vec3<f32>(
            select(ret[0].x, ret[0].x + oct_size.x, x_p),
            select(ret[0].y, ret[0].y + oct_size.y, y_p),
            select(ret[0].z, ret[0].z + oct_size.z, z_p),
        );

        ret = array<vec3<f32>, 2>(new_min, new_min + oct_size);
    }

    return ret;
}

struct Statistics {
    traversal_count: u32,
    intersection_count: u32,
}

fn intersect_scene(ray: Ray, inout_intersection: ptr<function, Intersection>, out_statistics: ptr<function, Statistics>) -> bool {
    let SENTINEL_NODE: u32 = 4294967295u;
    var next_node: u32 = 0;

    let global_extent = array<vec3<f32>, 2>(vec3<f32>(0., 0., 0.), vec3<f32>(scene_tree.outer_extents));

    var stats = Statistics(
        /* traversal_count    */ 0,
        /* intersection_count */ 0,
    );

    var intersected = false;
    loop {
        if next_node == SENTINEL_NODE {
            break;
        }

        stats.traversal_count++;

        let node = scene_tree.nodes[next_node];
        let node_extent = compute_extent(global_extent, node.path_pack);

        stats.intersection_count++;
        var cur_intersected = false;
        if node.link_hit == node.link_miss {
            let cur_box = Box(
                /* min */ node_extent[0],
                /* max */ node_extent[1],
                /* mat */ material_unpack(node.material_pack),
            );

            cur_intersected = box_intersect(cur_box, ray, inout_intersection);
            intersected = intersected || cur_intersected;
        } else {
            let cur_box = Box(
                /* min */ node_extent[0],
                /* max */ node_extent[1],
                /* mat */ Material(
                    /* mat type */ 1,
                    /* mat data */ vec4<f32>(1., 1., 1., 0.),
                ),
            );

            var temp_intersection = intersecton_new_dummy();
            cur_intersected = box_intersect_pt0(cur_box, ray, &temp_intersection);
        }


        next_node = select(node.link_miss, node.link_hit, cur_intersected);

    }

    *out_statistics = stats;

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

    //intersected = intersected || intersect_grid(ray, &intersection, &color_grid, vec3<f32>(-1, -2.5, 10.), vec3<f32>(1., -.5, 12.));
    //intersected = intersected || intersect_grid(ray, &intersection, &cornell_grid_back, vec3<f32>(-6., -6., -6.), vec3<f32>(6., 6., 6.));
    //intersected = intersected || intersect_grid(ray, &intersection, &cornell_grid_front, vec3<f32>(-6., -6., 6.), vec3<f32>(6., 6., 18.));
    //intersected = intersected || intersect_grid(ray, &intersection);

    var stats: Statistics;
    intersected = intersect_scene(ray, &intersection, &stats);

    //textureStore(texture_rt, pixel, vec4<f32>(f32(stats.intersection_count) / 40, f32(stats.traversal_count) / 40, 0., 1.));

    if intersected {
        textureStore(texture_rt, pixel, vec4<f32>(intersection.material.data.xyz * dot(intersection.normal, -ray.direction), 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(vec3<f32>(dot(intersection.normal, -ray.direction)), 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(abs(intersection.normal), 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(intersection.material.data.xyz, 1.));
    } else {
        textureStore(texture_rt, pixel, vec4<f32>(vec2<f32>(local_id.xy) / vec2<f32>(8., 8.), 0., 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(0., 0., 0., 1.));
    }
}
