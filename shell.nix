{ pkgs ? import ./nix/pin/nixpkgs.nix { } }:

with pkgs;

let
  pkgs-pytorch_041 = import ./nix/pin/nixpkgs-pytorch_04.nix { };

  developmentPython36 =
    let
      mypython = pkgs.python36.override {
        packageOverrides = self: super: rec {
          numpy_041       = pkgs-pytorch_041.python36Packages.numpy;
          pytorch_041     = pkgs-pytorch_041.python36Packages.pytorchWithCuda;
          torchvision_041 = pkgs-pytorch_041.python36Packages.torchvision.override { pytorch = pytorch_041; };

          pytorch_110     = (pkgs.callPackage ./nix/pytorch/release.nix {
            inherit pkgs; pythonPackages = pkgs.python36Packages;
          }).pytorchWithCuda10;

          probtorch   = pkgs.callPackage ./nix/probtorch    { inherit (pkgs.python36Packages) buildPythonPackage; pytorch = pytorch_041; };
          flatdict    = pkgs.callPackage ./nix/flatdict.nix { inherit (pkgs.python36Packages) buildPythonPackage fetchPypi; };
          pygtrie     = pkgs.callPackage ./nix/pygtrie.nix  { inherit (pkgs.python36Packages) buildPythonPackage fetchPypi; };
          matplotlib  = pkgs.python36Packages.matplotlib.override { enableQt = true; numpy = numpy_041; };
          combinators = pkgs.callPackage ./. {
            inherit (pkgs.python36Packages) buildPythonPackage;
            inherit probtorch flatdict pygtrie matplotlib;
          };
        };
        self = mypython;
      };

    in mypython.withPackages(ps: with ps; [
      combinators ipywidgets ipython jupyter notebook torchvision_041 pytorch_041 scipy imageio pillow
    ]); # bokeh is brokehn
in

mkShell {
  # numpy_041 doesn't get added properly from withPackages
  buildInputs = [ developmentPython36 pkgs-pytorch_041.python36Packages.pytorchWithCuda pkgs-pytorch_041.python36Packages.numpy ];
}



