#pragma description MagicaVoxel voxel files

//#pragma MIME application/octet-stream
#pragma endian little
//#pragma pattern_limit 1024000

#include <std/array.pat>
#include <std/core.pat>
#include <std/mem.pat>
#include <std/string.pat>

enum chunk_type : u8 {
    MAIN,
    PACK,
    SIZE,
    XYZI,
    RGBA,
    nTRN,
    nGRP,
    nSHP,
    MATL,
    LAYR,
    rOBJ,
    rCAM,
    NOTE,
    IMAP,
    
    UNKNOWN = 0xff,
};

fn get_chunk_type(auto id) {
    match (id) {
        ("MAIN"): return chunk_type::MAIN;
        ("PACK"): return chunk_type::PACK;
        ("SIZE"): return chunk_type::SIZE;
        ("XYZI"): return chunk_type::XYZI;
        ("RGBA"): return chunk_type::RGBA;
        ("nTRN"): return chunk_type::nTRN;
        ("nGRP"): return chunk_type::nGRP;
        ("nSHP"): return chunk_type::nSHP;
        ("MATL"): return chunk_type::MATL;
        ("LAYR"): return chunk_type::LAYR;
        ("rOBJ"): return chunk_type::rOBJ;
        ("rCAM"): return chunk_type::rCAM;
        ("NOTE"): return chunk_type::NOTE;
        ("IMAP"): return chunk_type::IMAP;
    }

    return chunk_type::UNKNOWN;
};

fn get_chunk_name(chunk_type type) {
    match (type) {
        (chunk_type::MAIN): return "main";
        (chunk_type::PACK): return "pack";
        (chunk_type::SIZE): return "size";
        (chunk_type::XYZI): return "xyz+index voxel data";
        (chunk_type::RGBA): return "rgba palette";
        (chunk_type::nTRN): return "transform node";
        (chunk_type::nGRP): return "group node";
        (chunk_type::nSHP): return "shape node";
        (chunk_type::MATL): return "material";
        (chunk_type::LAYR): return "layer";
        (chunk_type::rOBJ): return "render object";
        (chunk_type::rCAM): return "render camera";
        (chunk_type::NOTE): return "palette notes";
        (chunk_type::IMAP): return "index map";
    }
    return "unknown chunk";
};

str color_chunk_type = "ff0000";
str color_id0 = "0000ff";
str color_id1 = "000055";
str color_count = "ff00ff";
str color_bytecount0 = "aa00aa";
str color_bytecount1 = "550055";
str color_kv_key = "00ff00";
str color_kv_value = "00aa00";
str color_arbitrary = "555555";

struct pascal_string {
    u32 length [[color(color_bytecount0)]];
    char characters[length] [[color(color_kv_value)]];
} [[value(1)]];

struct dictionary_entry {
    pascal_string field_name [[color(color_kv_key)]];
    pascal_string field_value [[color(color_kv_value)]];
};

struct dictionary {
    u32 num_entries [[color(color_count)]];
    dictionary_entry dictionary[num_entries];
};

bitfield rotation {
    index_first_row : 2;
    index_second_row: 2;
    sign_0: 1;
    sign_1: 1;
    sign_2: 1;
};

struct model {
    s32 model_id [[color(color_id0)]];
    dictionary attributes;
};

namespace chunk {

struct main {};

struct pack { /* TODO */ };

struct size {
    s32 size[3];
};

struct xyzi {
    u32 num_voxels [[color(color_count)]];
    u8 data[4 * num_voxels] [[color(color_arbitrary)]];
};

struct rgba {
    u32 data[256];
} [[hex::visualize("bitmap", this.data, 16, 16)]];

struct node_transform {
    s32 node_id [[color(color_id0)]];
    dictionary attributes;
    s32 child_node_id [[color(color_id0)]];
    s32 reserved_id [[color(color_id1)]]; // must be == -1
    s32 layer_id [[color(color_id0)]];
    u32 num_frames [[color(color_count)]]; // must be > 0
    
    dictionary frame_attributes[num_frames];
};

struct node_group {
    s32 node_id [[color(color_id0)]];
    dictionary attributes;
    u32 num_children [[color(color_count)]];
    
    s32 children[num_children];
};

struct node_shape {
    s32 node_id [[color(color_id0)]];
    dictionary attributes;
    u32 num_models [[color(color_count)]];
    
    model models[num_models];
};

struct material {
    u32 material_id [[color(color_id0)]];
    dictionary dict;
};

struct layer {
    s32 layer_id [[color(color_id0)]];
    dictionary attributes;
    s32 reserved_id [[color(color_id1)]]; // must be == -1
};

struct render_object {
    dictionary attributes;
};

struct render_camera {
    s32 camera_id [[color(color_id0)]];
    dictionary attributes;
};

struct palette_note {
    u32 num_names [[color(color_count)]];
    pascal_string color_names[num_names];
};

struct index_map {
    u32 associations[256] [[color(color_arbitrary)]];
};

}

struct chunk_t {
    char chunk_id[4] [[color(color_chunk_type)]];
    u32 num_bytes [[color(color_bytecount0)]];
    u32 num_bytes_children [[color(color_bytecount1)]];
    
    match (chunk_id) {
        ("MAIN"): chunk::main;
        ("SIZE"): chunk::size;
        ("XYZI"): chunk::xyzi;
        ("RGBA"): chunk::rgba;
        ("MATL"): chunk::material;
        ("nTRN"): chunk::node_transform;
        ("nGRP"): chunk::node_group;
        ("nSHP"): chunk::node_shape;
        ("LAYR"): chunk::layer;
        ("rOBJ"): chunk::render_object;
        ("rCAM"): chunk::render_camera;
        ("NOTE"): chunk::palette_note;
        ("IMAP"): chunk::index_map;
        (_): u8 content[num_bytes] [[color(color_arbitrary)]];
    }
    
    u32 current_offset = $;
    chunk_t children_chunks[while($ - current_offset < num_bytes_children)];
} [[name(get_chunk_name(get_chunk_type(this.chunk_id)))]];

struct header {
    char magic[4];
    u32 version;
};

header header @ 0x00;
chunk_t chunks[while(!std::mem::eof())] @ 0x08;
