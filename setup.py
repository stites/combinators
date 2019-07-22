import io
import os
from setuptools import find_packages, setup, Command
import setuptools.command.build_py


def get_version():
    try:
        import subprocess
        CWD = os.path.dirname(os.path.abspath(__file__))
        rev = subprocess.check_output("git rev-parse --short HEAD".split(), cwd=CWD)
        version = "0.0+" + str(rev.strip().decode('utf-8'))
        return version
    except Exception:
        return "0.0"


def long_description():
    here = os.path.abspath(os.path.dirname(__file__))

    # Import the README and use it as the long-description.
    # Note: this will only work if 'README.md' is present in your MANIFEST.in file!
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return '\n' + f.read()


# Package meta-data.
NAME = 'combinators'
DESCRIPTION = 'Compositional operators for the design and training of deep probabilistic programs'
URL = 'https://github.com/probtorch/combinators'
VERSION = get_version()
REQUIRED = [
    'probtorch',
    'flatdict',
]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description(),
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
)
