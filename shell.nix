{ cudaSupport ? false, pkgs ? import ./nix/pytorch-world/pin/nixpkgs.nix { } }:

let
  developmentPython36 = (pkgs.callPackage ./nix/generic.nix { inherit cudaSupport pkgs; python3 = pkgs.python36; }).python;
in

pkgs.mkShell {
  # For guidelines on a localized, pip-like install, see https://nixos.wiki/wiki/Python
  buildInputs = [
    (developmentPython36.withPackages(ps: with ps; [
      probtorch ipywidgets ipython jupyter notebook pytorch scipy imageio pillow matplotlib flatdict
      # testing
      pytest pytest-mypy pytestcov protobuf future hypothesis
      # extras
      pip typeguard
    ]))
  ];
  shellHook = ''
    echo 'Entering Python Project Environment'
    set -v

    # extra packages can be installed here
    unset SOURCE_DATE_EPOCH
    export PIP_PREFIX="$(pwd)/pip_packages"
    python_path=(
      "$PIP_PREFIX/lib/python3.6/site-packages"
      "$PYTHONPATH"
    )
    # use double single quotes to escape bash quoting
    IFS=: eval 'python_path="''${python_path[*]}"'
    export PYTHONPATH="$python_path"
    export MPLBACKEND='Qt4Agg'

    set +v
  '';
}

