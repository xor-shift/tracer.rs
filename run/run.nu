def merge_shaders [out: string, files: list] {
    cat ...$files | save -f $out
}

def merge_shader [name: string] {
    let base_directory = $"shaders/($name)/"
    let shader_files = (ls $base_directory | get name)
    let common_files = (ls shaders/common/ | get name)
    let all_files = $common_files | append $shader_files

    merge_shaders $"shaders/out/($name).wgsl" $all_files
}

merge_shader path_tracer
merge_shader denoiser
merge_shader visualiser

clear
RUST_BACKTRACE=1 cargo run
