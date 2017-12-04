#!/usr/bin/python
# -*- coding: utf-8 -*-
from setuptools import setup

doclines = """eyring: automated analysis of reaction mechanisms from quantum chemistry

eyring is a Python library that provides automated analysis of reaction
mechanisms from computational chemistry log files.
""".split("\n")

# Chosen from http://www.python.org/pypi?:action=list_classifiers
classifiers = """Development Status :: 3 - Alpha
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Education
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Chemistry
Topic :: Education
Topic :: Software Development :: Libraries :: Python Modules"""

keywords = [
    'chemistry',
    'research',
    'science',
]

# The list of packages to be installed.
packages = [
    'eyring',
]

install_requires = [
    'numpy',
    'matplotlib',
    'networkx',
    'cclib',
    'scipy',
]

setup(
    name='eyring',
    version='0.1.1',
    url='https://github.com/dudektria/eyring',
    download_url='https://github.com/dudektria/eyring/archive/0.1.1.tar.gz',
    author='Felipe Silveira de Souza Schneider',
    author_email='schneider.felipe@posgrad.ufsc.br',
    license='MIT',
    description=doclines[0],
    long_description="\n".join(doclines[2:]),
    classifiers=classifiers.split("\n"),
    packages=packages,
    keywords=keywords,
    install_requires=install_requires,
)
