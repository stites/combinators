{ cudaSupport ? false, pkgs ? import ./pytorch-world/pin/nixpkgs.nix { } }:

let
  inherit (pkgs.callPackage ./generic.nix {
      inherit cudaSupport pkgs;
      python3 = pkgs.python36;
    }) python;
in
{
  releasePython36 = python.withPackages (ps: [
    ps.combinators
  ]);
}

