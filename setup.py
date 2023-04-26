import sys


try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_namespace_packages

setup(
    packages=find_namespace_packages(where="py_src"),
    package_dir={"": "py_src"},
    cmake_install_dir="py_src/swdatatoolkit",
    include_package_data=True,
)
