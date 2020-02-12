<<<<<<< HEAD
{ lib, fetchFromGitHub, buildPythonPackage, probtorch, flatdict, pygtrie, matplotlib
, doCheck ? true, pytest, pytestrunner, pytest-mypy, pytestcov, gym, protobuf, future, hypothesis
}:
=======
{ lib, fetchFromGitHub, buildPythonPackage, probtorch, flatdict, pygtrie, matplotlib }:
>>>>>>> 643285372ee37ef99e280ab9829907eacbe70efb

buildPythonPackage rec {
  pname = "combinators";
  version = "0.0";

  src = ./.;

<<<<<<< HEAD
  inherit doCheck;
  nativeBuildInputs = lib.optionals doCheck [ pytestrunner ];
  checkInputs = lib.optionals doCheck [ pytest pytest-mypy pytestcov gym protobuf future hypothesis ];
  checkPhase = "python setup.py pytest";
=======
  doCheck = false;
  # checkInputs = [ pytest ];
>>>>>>> 643285372ee37ef99e280ab9829907eacbe70efb
  propagatedBuildInputs = [ probtorch flatdict pygtrie matplotlib ];

  meta = with lib; {
    homepage = https://github.com/probtorch/combinators;
    description = "Compositional operators for the design and training of deep probabilistic programs.";
    license = licenses.mit;
    maintainers = with maintainers; [ stites ];
  };
}
