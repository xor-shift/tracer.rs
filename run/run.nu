def merge_shaders [out: string, files: list] {
    cat $files | save -f $out
}

let rasteriser_files = [
    "shaders/inc/common.wgsl"
    "shaders/inc/constants.wgsl"
    "shaders/inc/geometry.wgsl"
    "shaders/inc/state.wgsl"

    "shaders/rasteriser.wgsl"
]

let compute_files = [
    "shaders/inc/common.wgsl"
    "shaders/inc/geometry.wgsl"
    "shaders/inc/constants.wgsl"
    "shaders/inc/state.wgsl"

    "shaders/inc/rng.wgsl"
    "shaders/inc/dist.wgsl"

    "shaders/inc/ray.wgsl"
    "shaders/inc/pinpoint.wgsl"

    "shaders/inc/shapes/sphere.wgsl"
    "shaders/inc/shapes/triangle.wgsl"

    "shaders/inc/sky.wgsl"

    "shaders/compute.wgsl"
]

let denoiser_files = [
    "shaders/inc/common.wgsl"
    "shaders/inc/geometry.wgsl"
    "shaders/inc/constants.wgsl"

    "shaders/denoiser.wgsl"
]

let visualiser_files = [
    "shaders/inc/common.wgsl"
    "shaders/inc/geometry.wgsl"
    "shaders/inc/constants.wgsl"

    "shaders/main.wgsl"
]

merge_shaders "shaders/out/rasteriser.wgsl" $rasteriser_files
merge_shaders "shaders/out/compute.wgsl" $compute_files
merge_shaders "shaders/out/denoiser.wgsl" $denoiser_files
merge_shaders "shaders/out/main.wgsl" $visualiser_files

clear
RUST_BACKTRACE=1 cargo run
