@group(0) @binding(0) var<uniform> state: State;

@group(1) @binding(0) var texture_rt: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(1) var texture_geo: texture_storage_2d_array<rgba32uint, read_write>;
@group(1) @binding(2) var texture_rt_old: texture_2d<f32>;
@group(1) @binding(3) var texture_geo_old: texture_2d_array<u32>;

struct ThreadedNode {
    path_pack: vec2<u32>,
    pack_links_material: vec2<u32>,
}

const SENTINEL_NODE: u32 = 4294967295u;

fn threaded_node_is_leaf(that: ThreadedNode) -> bool { return (that.pack_links_material[0] >> 31u) == 0u; }

fn threaded_node_material(that: ThreadedNode) -> Material { return material_unpack(that.pack_links_material & vec2<u32>(0x7FFFFFFF, 0xFFFFFFFF)); }

fn threaded_node_miss_offset(that: ThreadedNode) -> u32 { return that.pack_links_material[1]; }

fn threaded_node_miss_link(that: ThreadedNode, current_index: u32) -> u32 {
    if threaded_node_is_leaf(that) {
        return current_index + 1;
    } else if threaded_node_miss_offset(that) == SENTINEL_NODE {
        return SENTINEL_NODE;
    } else {
        return threaded_node_miss_offset(that) + current_index;
    }
}

struct Chunk {
    location: vec3<i32>,
    pool_pointer: u32,
    size: vec3<u32>,
    node_count: u32,
}

@group(2) @binding(0) var<storage> g_chunks: array<Chunk>;
@group(2) @binding(1) var<storage> g_node_pool: array<ThreadedNode>;

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

fn select_tree(v: vec3<f32>) -> u32 {
    let abs_v = abs(v);
    
    let a = abs_v.x > abs_v.y;
    let b = abs_v.x > abs_v.z;
    let c = abs_v.z > abs_v.y;
    let largest_axis = u32(!a && !b) + u32(!b && c) * 2u;
    let corrected_largest_axis = select(largest_axis, largest_axis + 7, v[largest_axis] < 0.);

    let is_corner = (abs_v.x + abs_v.y + abs_v.z) > 1.5;
    let corner_if_corner = u32(abs(i32(v.y <= 0.) * 10 - (i32(v.x > 0.) + i32(v.z > 0.) * 2))) + 3;

    return select(corrected_largest_axis, corner_if_corner, is_corner);
}

fn intersect_scene(ray: Ray, inout_intersection: ptr<function, Intersection>, out_statistics: ptr<function, Statistics>) -> bool {
    var stats = Statistics(
        /* traversal_count    */ 0,
        /* intersection_count */ 0,
    );

    var intersected = false;

    for (var i = 0u; i < arrayLength(&g_chunks); i++) {
        let chunk = g_chunks[i];

        var next_node: u32 = 0;

        let chunk_extents = array<vec3<f32>, 2>(vec3<f32>(chunk.location), vec3<f32>(chunk.location) + vec3<f32>(chunk.size));

        loop {
            if next_node >= chunk.node_count {
                break;
            }

            stats.traversal_count++;

            let node = g_node_pool[chunk.pool_pointer + next_node];
            let node_extent = compute_extent(chunk_extents, node.path_pack);

            stats.intersection_count++;
            var cur_intersected = false;
            if threaded_node_is_leaf(node) {
                let cur_box = Box(
                    /* min */ node_extent[0],
                    /* max */ node_extent[1],
                    /* mat */ threaded_node_material(node),
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
                cur_intersected = cur_intersected && temp_intersection.t < (*inout_intersection).t;
            }

            let miss_link = threaded_node_miss_link(node, next_node);
            next_node = select(miss_link, next_node + 1, cur_intersected);
        }
    }

    *out_statistics = stats;

    return intersected;
}

fn jet(v_: f32, vmin: f32, vmax: f32) -> vec3<f32> {
   var c = vec3<f32>(1.);

   var v = v_; 
   if v < vmin { v = vmin; }
   if v > vmax { v = vmax; }
   let dv = vmax - vmin;

   if (v < (vmin + 0.25 * dv)) {
      c.r = 0.;
      c.g = 4. * (v - vmin) / dv;
   } else if (v < (vmin + 0.5 * dv)) {
      c.r = 0.;
      c.b = 1. + 4. * (vmin + 0.25 * dv - v) / dv;
   } else if (v < (vmin + 0.75 * dv)) {
      c.r = 4. * (v - vmin - 0.5 * dv) / dv;
      c.b = 0.;
   } else {
      c.g = 1. + 4. * (vmin + 0.75 * dv - v) / dv;
      c.b = 0.;
   }

   return c;
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

    var stats: Statistics;
    intersected = intersected || intersect_scene(ray, &intersection, &stats);

    var color_to_store = vec3<f32>(0., 0., 0.);

    switch state.visualisation_mode {
        /* PathTrace              */ case 0: { color_to_store = intersection.material.data.xyz * dot(intersection.normal, -ray.direction); }
        /* RealIntersectionTests  */ case 1: { color_to_store = jet(f32(stats.intersection_count), 0., 1024.); }
        /* BoundIntersectionTests */ case 2: { color_to_store = jet(f32(stats.traversal_count), 0., 1024.); }
        /* Denoise0               */ case 3: {}
        /* Denoise1               */ case 4: {}
        /* PathTraceAlbedo        */ case 5: {}
        /* Denoise0Albedo         */ case 6: {}
        /* Denoise1Albedo         */ case 7: {}
        /* Normal                 */ case 8: {}
        /* AbsNormal              */ case 9: {}
        /* DistFromOrigin         */ case 10: {}

        default: {}
    }

    textureStore(texture_rt, pixel, vec4<f32>(color_to_store, 1.));

    /*if intersected {
        textureStore(texture_rt, pixel, vec4<f32>(intersection.material.data.xyz * dot(intersection.normal, -ray.direction), 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(vec3<f32>(dot(intersection.normal, -ray.direction)), 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(abs(intersection.normal), 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(intersection.material.data.xyz, 1.));
    } else {
        textureStore(texture_rt, pixel, vec4<f32>(vec2<f32>(local_id.xy) / vec2<f32>(8., 8.), 0., 1.));
        //textureStore(texture_rt, pixel, vec4<f32>(0., 0., 0., 1.));
    }*/
}
