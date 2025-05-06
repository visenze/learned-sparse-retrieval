from setuptools import setup, find_packages
import os
import sys


long_description = "Learned Sparse Retrieval: https://github.com/visenze/learned-sparse-retrieval/"

# Package meta-data.
NAME = "learned-sparse-retrieval"
DESCRIPTION = "Learned Sparse Retrieval"
LONG_DESCRIPTION = long_description
URL = "https://github.com/visenze/learned-sparse-retrieval/"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.0.1"

with open("requirements.txt") as f:
    REQUIRED = [line.rstrip('\n') for line in f]
LINKS = []
EXTRAS = {}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    dependency_links=LINKS,
    extras_require=EXTRAS,
    include_package_data=True,
    entry_points={
        "console_scripts": [
        ]
    },
    classifiers=["Programming Language :: Python :: 3.9", ],
)
