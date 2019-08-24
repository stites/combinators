{ pkgs ? import ./nix/pytorch-world/pin/nixpkgs.nix { } }:

let
  cudatoolkit = pkgs.cudatoolkit_10_0;
  cudnn = pkgs.cudnn_cudatoolkit_10_0;
  nccl = pkgs.nccl_cudatoolkit_10;
  cudaSupport = true;
  mklSupport = true;
  magma = pkgs.callPackage ./nix/pytorch-world/deps/magma_250.nix { inherit cudatoolkit mklSupport; };
  openmpi = pkgs.callPackage ./nix/pytorch-world/deps/openmpi.nix { inherit cudatoolkit cudaSupport; };

  developmentPython36 =
    let
      mypython = pkgs.python36.override {
        packageOverrides = self: super: rec {
          pytorch = super.callPackage ./nix/pytorch-world/pytorch   {
            openMPISupport = true;
            inherit mklSupport magma openmpi;
            inherit cudatoolkit cudnn nccl cudaSupport;
            buildNamedTensor = true;
            buildBinaries = true;
          };
          # overwrite numpy with explicit mkl support so that other packages play nice.
          # If you don't do this, pytorch overrides numpy's blas, but this won't proagate
          # to other packages.
          numpy       = super.numpy.override { blas = pkgs.mkl; };
          probtorch   = super.callPackage ./nix/pytorch-world/probtorch { inherit pytorch; };
          flatdict    = super.callPackage ./nix/deps/flatdict.nix { };
          pygtrie     = super.callPackage ./nix/deps/pygtrie.nix  { };
          matplotlib  = super.matplotlib.override { enableQt = true; };
          combinators = super.callPackage ./. {
            inherit probtorch flatdict pygtrie matplotlib;
          };
        };
        self = mypython;
      };

    in mypython.withPackages(ps: with ps; [
      combinators probtorch ipywidgets ipython jupyter notebook pytorch scipy imageio pillow matplotlib
    ]);
in

pkgs.mkShell {
  # For guidelines on a localized, pip-like install, see https://nixos.wiki/wiki/Python
  buildInputs = [ developmentPython36 ];
}

