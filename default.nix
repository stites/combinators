{ lib, fetchFromGitHub, buildPythonPackage, probtorch, flatdict, pygtrie, matplotlib
, pytest, pytestrunner, gym
}:

buildPythonPackage rec {
  pname = "combinators";
  version = "0.0";

  src = ./.;

  doCheck = true;
  checkInputs = [ pytest pytestrunner gym ];
  propagatedBuildInputs = [ probtorch flatdict pygtrie matplotlib ];

  meta = with lib; {
    homepage = https://github.com/probtorch/combinators;
    description = "Compositional operators for the design and training of deep probabilistic programs.";
    license = licenses.mit;
    maintainers = with maintainers; [ stites ];
  };
}
