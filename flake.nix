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

        llamaCppCuda =
          (pkgs.llama-cpp.override {
            cudaSupport = true;
          }).overrideAttrs (old: {
            version = "8642"; # gemma4 support

            src = pkgs.fetchFromGitHub {
              owner = "ggml-org";
              repo = "llama.cpp";
              rev = "b8642";
              hash = "sha256-CGoqQd4jdcVlttg1fFkUNW4v3Rxwfkj+TVlcHD59RhI=";
            };
          });

        pythonEnv = pkgs.python311.withPackages (ps:
          with ps; [
            aiohttp
            pandas
            ruff
            sentence-transformers
          ]);
      in {
        packages = {inherit llamaCppCuda pythonEnv;};

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [pkgs.makeWrapper];

          packages = [
            pythonEnv
            llamaCppCuda
            pkgs.cudatoolkit
            pkgs.basedpyright
            pkgs.just
            pkgs.aria2
          ];

          shellHook = ''
            echo "Environment loaded with CUDA support."
            echo "Python version: $(python --version)"
            llama-server --version
          '';
        };
      }
    );
}
