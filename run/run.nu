def merge_shaders [out: string, files: list] {
    cat $files | save -f $out
}

let compute_files = [
    "shaders/inc/common.wgsl"
    "shaders/inc/geometry.wgsl"
    "shaders/inc/constants.wgsl"

    "shaders/inc/rng.wgsl"
    "shaders/inc/dist.wgsl"

    "shaders/inc/ray.wgsl"
    "shaders/inc/pinpoint.wgsl"

    "shaders/inc/shapes/sphere.wgsl"
    "shaders/inc/shapes/triangle.wgsl"

    "shaders/compute.wgsl"
]

let visualiser_files = [
    "shaders/inc/common.wgsl"
    "shaders/inc/geometry.wgsl"
    "shaders/inc/constants.wgsl"

    "shaders/main.wgsl"
]

let rasteriser_files = [
    "shaders/inc/common.wgsl"
    "shaders/inc/constants.wgsl"

    "shaders/rasteriser.wgsl"
]

merge_shaders "shaders/out/compute.wgsl" $compute_files
merge_shaders "shaders/out/main.wgsl" $visualiser_files
merge_shaders "shaders/out/rasteriser.wgsl" $rasteriser_files

clear
RUST_BACKTRACE=1 cargo run
