import os

from setuptools import find_packages, setup

long_description = (
    "Learned Sparse Retrieval: https://github.com/visenze/learned-sparse-retrieval/"
)

# Package meta-data.
NAME = "lsr"
DESCRIPTION = "Learned Sparse Retrieval - A package for training and inference with learned sparse retrieval models"
URL = "https://github.com/visenze/learned-sparse-retrieval/"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "1.0.0"

with open("requirements.txt") as f:
    REQUIRED = [
        line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")
    ]

EXTRAS = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "isort>=5.10.0",
    ]
}

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=REQUIRES_PYTHON,
    url=URL,
    project_urls={
        "Homepage": URL,
        "Repository": URL,
        "Documentation": URL,
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "images"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    package_data={
        "lsr": ["configs/**/*.yaml", "**/*.yaml"],
    },
    entry_points={"console_scripts": []},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="sparse retrieval information-retrieval neural-search transformers",
)
