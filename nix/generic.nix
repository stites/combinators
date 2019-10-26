{ cudaSupport ? false, pkgs ? import ./pytorch-world/pin/nixpkgs.nix { }, python3 ? pkgs.python3 }:

let
  cudatoolkit = pkgs.cudatoolkit_10_0;
  cudnn = pkgs.cudnn_cudatoolkit_10_0;
  nccl = pkgs.nccl_cudatoolkit_10;
  mklSupport = true;
  magma = pkgs.callPackage ./pytorch-world/deps/magma_250.nix { inherit cudatoolkit mklSupport; };
  openmpi = pkgs.callPackage ./pytorch-world/deps/openmpi.nix { inherit cudatoolkit cudaSupport; };

  mypython = python3.override {
    packageOverrides = self: super: rec {
      pytorch = super.callPackage ./pytorch-world/pytorch   {
        openMPISupport = true;
        inherit mklSupport magma openmpi;
        inherit cudatoolkit cudnn nccl cudaSupport;
        buildNamedTensor = true;
        buildBinaries = false;
      };
      numpy       = super.numpy.override { blas = pkgs.mkl; };
      probtorch   = super.callPackage ./pytorch-world/probtorch { inherit pytorch; };
      flatdict    = super.callPackage ./deps/flatdict.nix { };
      pygtrie     = super.callPackage ./deps/pygtrie.nix  { };
      matplotlib  = super.matplotlib.override { enableQt = true; };
      combinators = super.callPackage ./.. { inherit probtorch flatdict pygtrie matplotlib; };
    };
    self = mypython;
  };
in

{
  python = mypython;
}

