{ lib, fetchFromGitHub, buildPythonPackage, probtorch, flatdict, pygtrie, matplotlib
, doCheck ? true, pytest, pytestrunner, pytest-mypy, pytestcov, gym, protobuf, future, hypothesis
}:

buildPythonPackage rec {
  pname = "combinators";
  version = "0.0";

  src = ./.;

  inherit doCheck;
  nativeBuildInputs = lib.optionals doCheck [ pytestrunner ];
  checkInputs = lib.optionals doCheck [ pytest pytest-mypy pytestcov gym protobuf future hypothesis ];
  checkPhase = "python setup.py pytest";
  propagatedBuildInputs = [ probtorch flatdict pygtrie matplotlib ];

  meta = with lib; {
    homepage = https://github.com/probtorch/combinators;
    description = "Compositional operators for the design and training of deep probabilistic programs.";
    license = licenses.mit;
    maintainers = with maintainers; [ stites ];
  };
}
