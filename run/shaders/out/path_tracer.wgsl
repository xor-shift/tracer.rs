const PI: f32 = 3.14159265358979323846264338327950288; // π
const FRAC_PI_2: f32 = 1.57079632679489661923132169163975144; // π/2
const FRAC_PI_3: f32 = 1.04719755119659774615421446109316763; // π/3
const FRAC_PI_4: f32 = 0.785398163397448309615660845819875721; // π/4
const FRAC_PI_6: f32 = 0.39269908169872415480783042290993786; // π/6
const FRAC_1_PI: f32 = 0.318309886183790671537767526745028724; // 1/π
const FRAC_1_SQRT_PI: f32 = 0.564189583547756286948079451560772586; // 1/√π
const FRAC_2_PI: f32 = 0.636619772367581343075535053490057448; // 2/π
const FRAC_2_SQRT_PI: f32 = 1.12837916709551257389615890312154517; // 2/√π
const PHI: f32 = 1.618033988749894848204586834365638118; // φ
const SQRT_2: f32 = 1.41421356237309504880168872420969808; // √2
const FRAC_1_SQRT_2: f32 = 0.707106781186547524400844362104849039; // 1/√2
const SQRT_3: f32 = 1.732050807568877293527446341505872367; // √3
const FRAC_1_SQRT_3: f32 = 0.577350269189625764509148780501957456; // 1/√3

const MAT3x3_IDENTITY: mat3x3<f32> = mat3x3<f32>(1., 0., 0., 0., 1., 0., 0., 0., 1.);

const INF: f32 = 999999999999999999999.;
struct Geometry {
    normal: vec3<f32>,
    position: vec3<f32>,
}

fn _geometry_normal_to_spherical(normal: vec3<f32>) -> vec2<f32> {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;

    let r = 1.; // sqrt(x*x + y*y + z*z)

    let θ = select(
        acos(z / r),
        FRAC_PI_2,
        x == 0. && z == 0.
    );

    let φ = select(
        sign(y) * acos(x / sqrt(x * x + y * y)),
        -FRAC_PI_2,
        x == 0. && y == 0.
    );

    return vec2<f32>(θ, φ);
}

fn _geometry_normal_from_spherical(spherical: vec2<f32>) -> vec3<f32> {
    let r = 1.;
    let θ = spherical.x;
    let φ = spherical.y;

    let x = r * sin(θ) * cos(φ);
    let y = r * sin(θ) * sin(φ);
    let z = r * cos(θ);

    return vec3<f32>(x, y, z);
}

fn geometry_pack(geometry: Geometry) -> PackedGeometry {
    return PackedGeometry(
        /* pack_0 */ vec4<u32>(0),
        /* pack_1 */ vec4<u32>(0),
    );
}

struct PackedGeometry {
    pack_0: vec4<u32>,
    pack_1: vec4<u32>,
}
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    direction_reciprocal: vec3<f32>,
}

fn ray_new(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
    return Ray(origin, direction, 1. / direction);
}

fn ray_from_pixel(pixel: vec2<u32>, state: State) -> Ray {
    let dimensions = vec2<f32>(state.dimensions);
    let pixel_corr = vec2<f32>(f32(pixel.x), dimensions.y - f32(pixel.y));
    
    // the 1.5 fixes the fov issue and i have no clue why
    let ndc_pixel = ((pixel_corr / dimensions) * 2. - 1.) * 1.5;

    let coord = state.inverse_transform * vec4<f32>(ndc_pixel, 0., 1.);
    let ray_dir = normalize((coord.xyz / coord.w) - state.camera_position);

    return ray_new(state.camera_position, ray_dir);
}

struct Material {
    mat_type: u32, // determines how the data is interpreted (min 0, max 255)
    /*
        all values are in the range [0, 1) but fields may have scale factors

        | symbol | name      | scale | resolution (bits) | type  |
        +--------+-----------+-------+-------------------+-------+
        | *      | unused    | N/A   | N/A               | N/A   |
        | R      | red       | 1     | 8                 | unorm |
        | G      | green     | 1     | 8                 | unorm |
        | B      | blue      | 1     | 8                 | unorm |
        | g      | gloss     | NYD   | 16                | unorm |
        | I      | r. idx    | NYD   | 16                | unorm |
        | i      | intensity | 100   | 16                | unorm |

        materials:
        0 -> **** air
        1 -> RGB* diffuse
        2 -> RGBi light
        3 -> RGB* perfect mirror
        4 -> RGBI glass
        5 -> RGBG glossy

        brute force thing for white lights of arbitrary intensitites:
        function f(target) {
            for (let unorm8 = 0; unorm8 < 256; unorm8++) {
                let unorm8_f32 = unorm8 / 255;

                let optim_scaled = target / unorm8_f32;
                let optim_unorm = optim_scaled / 100;
                let optim_unorm16 = optim_unorm * 65535;

                //
            }
        }
    */
    data: vec4<f32>,
}

// packs a material for storage
fn material_pack(material: Material) -> vec2<u32> {
    let first_quad = (pack4x8unorm(material.data) & 0x00FFFFFFu) | ((material.mat_type & 0x000000FFu) << 24u);
    let second_quad = pack2x16unorm(material.data.ba) & 0xFFFF0000u;

    return vec2<u32>(first_quad, second_quad);
}

// unpacks a packed material
fn material_from_pack(pack: vec2<u32>) -> Material {
    let mat_type = (pack[0] >> 24u) & 0xFFu;
    let mat_data_rgb = unpack4x8unorm(pack[0]).rgb;
    let mat_data_a = unpack2x16unorm(pack[1]).x;

    return Material(
        /* mat_type */ mat_type,
        /* data     */ vec4<f32>(mat_data_rgb, mat_data_a),
    );
}

struct Intersection {
    material: Material,

    wo: vec3<f32>,
    t: f32,

    gloabl_pt: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
}

fn intersecton_new_dummy() -> Intersection {
    return Intersection(
        /* material  */ Material (
            /* typ */ 0,
            /* dat */ vec4<f32>(0.),
        ),

        /* wo        */ vec3<f32>(0.),
        /* t         */ 99999999.,

        /* global_pt */ vec3<f32>(0.),
        /* normal    */ vec3<f32>(0.),
        /* uv        */ vec2<f32>(0.),
    );
}
struct State {
    camera_transform: mat4x4<f32>,
    inverse_transform: mat4x4<f32>,
    frame_seed: vec4<u32>,
    camera_position: vec3<f32>,
    frame_no: u32,
    dimensions: vec2<u32>,
    current_instant: f32,
    visualisation_mode: i32,
}
struct Box {
    min: vec3<f32>,
    max: vec3<f32>,

    material: Material,
}

// https://tavianator.com/2011/ray_box.html
fn _box_intersect_impl_baseline(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    var tmin = 0.;
    var tmax = *inout_distance;

    for (var i = 0; i < 3; i++) {
        let t1 = (box.min[i] - ray.origin[i]) * ray.direction_reciprocal[i];
        let t2 = (box.max[i] - ray.origin[i]) * ray.direction_reciprocal[i];

        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    }

    let intersected = tmin < tmax;

    if intersected {
        *inout_distance = tmin;
    }

    return intersected;
}

// https://tavianator.com/2015/ray_box_nan.html
fn _box_intersect_impl_exclusive(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    var tmin = 0.;
    var tmax = *inout_distance;

    for (var i = 0; i < 3; i++) {
        let t1 = (box.min[i] - ray.origin[i]) * ray.direction_reciprocal[i];
        let t2 = (box.max[i] - ray.origin[i]) * ray.direction_reciprocal[i];

        tmin = max(tmin, min(min(t1, t2), tmax));
        tmax = min(tmax, max(max(t1, t2), tmin));
    }

    let intersected = tmin < tmax;

    if intersected {
        *inout_distance = tmin;
    }

    return intersected;
}

// https://tavianator.com/2022/ray_box_boundary.html#boundaries
fn _box_intersect_impl_inclusive(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    var tmin = 0.;
    var tmax = *inout_distance;

    for (var i = 0; i < 3; i++) {
        let t1 = (box.min[i] - ray.origin[i]) * ray.direction_reciprocal[i];
        let t2 = (box.max[i] - ray.origin[i]) * ray.direction_reciprocal[i];

        tmin = min(max(t1, tmin), max(t2, tmin));
        tmax = max(min(t1, tmax), min(t2, tmax));
    }

    let intersected = tmin <= tmax;

    if intersected {
        *inout_distance = tmin;
    }

    return intersected;
}

// https://tavianator.com/2022/ray_box_boundary.html#boundaries
fn _box_intersect_impl_signs(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    var tmin = 0.;
    var tmax = *inout_distance;

    for (var i = 0; i < 3; i++) {
        let sign_bit = select(0, 1, ray.direction_reciprocal[i] < 0);

        let bmin = select(box.min[i], box.max[i], sign_bit == 1);
        let bmax = select(box.min[i], box.max[i], sign_bit == 0);

        let dmin = (bmin - ray.origin[i]) * ray.direction_reciprocal[i];
        let dmax = (bmax - ray.origin[i]) * ray.direction_reciprocal[i];

        tmin = max(dmin, tmin);
        tmax = min(dmax, tmax);
    }

    let intersected = tmin <= tmax;

    if intersected {
        *inout_distance = tmin;
    }

    return intersected;
}

fn _box_intersect(box: Box, ray: Ray, inout_distance: ptr<function, f32>) -> bool {
    return _box_intersect_impl_inclusive(box, ray, inout_distance);
}

fn _box_normal_naive(box: Box, global_pt: vec3<f32>) -> vec3<f32> {
    /*

    +---+ (4, 3)
    |   x (4, 2)
    y   | (0, 1)
    +---+

    global -> local     -> per-axis norm -> fudge       -> trunc
    (4, 2) -> (2, .5)   -> (1, .3)       -> (1.1, .4)   -> (1, 0)
    (0, 1) -> (-2, -.5) -> (-1, -.3)     -> (-1.1, -.4) -> (-1, 0)

    */

    let center = (box.min + box.max) * 0.5;
    let local_pt = global_pt - center;

    let half_side_lengths = (box.max - box.min) * .5;
    let norm_local_pt = local_pt / half_side_lengths;
    let fudged = norm_local_pt * 1.00001;

    return normalize(trunc(fudged));
}

fn _box_normal(box: Box, global_pt: vec3<f32>) -> vec3<f32> {
    return _box_normal_naive(box, global_pt);
}

fn box_intersect(box: Box, ray: Ray, inout_intersection: ptr<function, Intersection>) -> bool {
    var t = (*inout_intersection).t;
    if !_box_intersect(box, ray, &t) {
        return false;
    }

    let global_pt = ray.origin + t * ray.direction;
    let normal = _box_normal(box, global_pt);

    *inout_intersection = Intersection (
        /* material  */ box.material,

        /* wo        */ -ray.direction,
        /* t         */ t,

        /* global_pt */ global_pt,
        /* normal    */ normal,
        /* uv        */ vec2<f32>(0.),
    );

    return true;
}
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
