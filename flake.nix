{
  description = "Agentic Translation Workflow environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        llamaCppCuda = pkgs.llama-cpp.override {
          cudaSupport = true;
        };

        pythonEnv = pkgs.python311.withPackages (ps:
          with ps; [
            aiohttp
            pandas
            ruff
          ]);
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [pkgs.makeWrapper];

          packages = [
            pythonEnv
            llamaCppCuda
            pkgs.cudatoolkit
            pkgs.basedpyright
            pkgs.just
            pkgs.aria2

            pkgs.vulkan-tools
            pkgs.vulkan-loader
            pkgs.vulkan-headers
          ];

          shellHook = ''
            echo "Environment loaded with CUDA support."
            echo "Python version: $(python --version)"
            echo "Llama server: $(llama-server --version | head -n 1)"
          '';
        };
      }
    );
}
