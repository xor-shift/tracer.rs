{
  description = "A devShell example";

  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
      in
        with pkgs; {
          devShells.default = mkShell rec {
            packages = with pkgs; [ nushell lldb ];
            buildInputs = [
              openssl
              pkg-config

              xorg.libXcursor
              xorg.libXrandr
              xorg.libXi
              xorg.libX11
              libxkbcommon

              libGL
              vulkan-loader

              (rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
            ];

            LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";

            shellHook = ''
              #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.wayland}/lib:${pkgs.libxkbcommon}/lib
            '';
          };
        }
    );
}
