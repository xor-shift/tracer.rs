def merge_shaders [out: string, files: list] {
    cat ...$files | save -f $out
}

def merge_shader [name: string] {
    let shader_files = (ls $"../src/shaders/($name)/" | get name)
    let common_files = (ls ../src/shaders/common/ | get name)
    let all_files = $common_files | append $shader_files

    merge_shaders $"shaders/($name).wgsl" $all_files
}

merge_shader path_tracer
merge_shader denoiser
merge_shader visualiser

clear
RUST_BACKTRACE=1 cargo run
