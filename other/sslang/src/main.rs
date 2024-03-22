/// SSLang - Shitty Shader Language
/// This is a shitty shader language that compiles into WGSL
/// Actually, I lied, this MIGHT become a shitty shader language

use lalrpop_util::lalrpop_mod;

lalrpop_mod!(grammar);

fn main() {
    lalrpop::process_root().unwrap();
}
