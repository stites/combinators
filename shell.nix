{ pkgs ? import ./nix/pin/nixpkgs.nix { } }:

with pkgs;

let
  pkgs-pytorch_41 = import ./nix/pin/nixpkgs-pytorch_04.nix { };
  developmentPython36 =
    let
      mypython = pkgs.python36.override {
        packageOverrides = self: super: let
          pytorch_11 = (pkgs.callPackage ./nix/pytorch/release.nix { inherit pkgs; pythonPackages = pkgs.python36Packages; }).pytorchWithCuda10;
          pytorch = pkgs-pytorch_41.python36Packages.pytorchWithCuda;
          torchvision = pkgs.python36Packages.torchvision.override { inherit pytorch; };

          probtorch = pkgs.callPackage ./nix/probtorch { inherit (pkgs.python36Packages) buildPythonPackage; inherit pytorch; };
          flatdict = pkgs.callPackage ./nix/flatdict.nix { inherit (pkgs.python36Packages) buildPythonPackage fetchPypi; };
          pygtrie = pkgs.callPackage ./nix/pygtrie.nix { inherit (pkgs.python36Packages) buildPythonPackage fetchPypi; };
          matplotlib = pkgs.python36Packages.matplotlib.override { enableQt = true; };
          combinators = pkgs.callPackage ./. {
            inherit (pkgs.python36Packages) buildPythonPackage;
            inherit probtorch flatdict pygtrie matplotlib;
          };
        in { inherit pytorch probtorch combinators; };
        self = mypython;
      };
    in mypython.withPackages(ps: with ps; [ combinators ipywidgets ipython jupyter notebook bokeh torchvision ]);
in

developmentPython36.env



