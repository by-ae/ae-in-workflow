#!/usr/bin/env python3
"""
Setup script for ae-in-workflow ComfyUI nodes
"""

from setuptools import setup, find_packages
import os

# Read the README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ae-in-workflow",
    version="0.1.1",
    author="ae",
    author_email="ae@ae-maker.com",
    description="In-workflow Nodes for ComfyUI - Heavy interaction and streaming capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ae-maker/ae-in-workflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "monitor": ["screeninfo>=0.8.0"],
    },
    package_data={
        "": ["web/*"],
    },
    include_package_data=True,
)
