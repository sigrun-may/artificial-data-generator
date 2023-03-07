# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Build script for setuptools."""

import os

import setuptools


project_name = "artificial_data_generator"
source_code = "https://github.com/sigrun-may/artificial-data-generator"
keywords = "artificial data"
install_requires = ["pyyaml", "numpy", "pandas", "seaborn", "matplotlib", "statsmodels"]
extras_require = {
    "checking": [
        "black",
        "flake8",
        "isort",
        "mdformat",
        "pydocstyle",
        "mypy",
        "pylint",
        "pylintfileheader",
    ],
    "testing": ["pytest"],
    "doc": ["sphinx", "sphinx_rtd_theme", "myst_parser", "sphinx_copybutton"],
}


def get_version():
    """Read version from ``__init__.py``."""
    version_filepath = os.path.join(os.path.dirname(__file__), project_name, "__init__.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=project_name,
    version=get_version(),
    maintainer="Sigrun May",
    author="Sigrun May",
    author_email="s.may@ostfalia.de",
    description="",  # TODO: add short description
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=source_code,
    project_urls={
        "Bug Tracker": source_code + "/issues",
        # "Documentation": "",  # TODO: add link to Sphinx page
        "Source Code": source_code,
        #        "Contributing": source_code + "/blob/main/CONTRIBUTING.md",
        #        "Code of Conduct": source_code + "/blob/main/CODE_OF_CONDUCT.md",
    },
    packages=setuptools.find_packages(),
    python_requires=">=3.6",  # TODO: check this
    install_requires=install_requires,
    extras_require=extras_require,
    keywords=keywords,
    classifiers=[
        # "Development Status :: 3 - Alpha",
        "Development Status :: 4 - Beta",
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
)
