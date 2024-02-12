""" setup.py - Main setup module """
import os

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(HERE, "README.md")).read()
VERSION = "0.3"

# Publicly Available Packages (PyPi)
INSTALL_REQUIRES = ["ipython",
                    "jupyterlab",
                    "matplotlib",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "torchtext",
                    "peft",
                    "transformers",
                    "datasets",
                    "tensorflow",
                    "sentencepiece",
                    "nltk",
                    "evaluate",
                    "bert-score",
                    "supar",
                    "pandas",
                    "scikit-learn",
                    "tensorboard",
                    "absl-py",
                    "peft"
                    ]

DEV_REQUIRES = [
    "nbqa[toolchain]",
    "black",
    "flake8",
    "isort",
    "mypy",
    "pre-commit",
    "pytest",
    "pytest-cov",
    "toml",
    "types-requests",
    "types-setuptools",
    "docformatter",
]

setup(
    name="paraphrase_attacker",
    version=VERSION,
    description="Codes developed for research with paraphrase generation for LLMs.",
    long_description=README,
    classifiers=["Programming Language :: Python :: 3.11"],
    keywords="NLP, Machine Learning",
    author="Saeed Najafi",
    author_email="snajafi@ualberta.ca",
    license="MIT",
    packages=find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    extras_require={"dev": DEV_REQUIRES},
)