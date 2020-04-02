#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup the package
To do:
    [type the aims and objectives of this script]
Created on Wed Apr  1 15:03:58 2020

@author: Xiaodong Ming
"""


import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
        name='hipims_io',
        version='0.2.3',
        description='To process input and output files of the HiPIMS model',
        url='https://github.com/mingxiaodong/hipims_io_py_package',
        author='Xiaodong Ming',
        author_email='xiaodong.ming@outlook.com',
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
        ],
        python_requires='>=3.6')

"""
#Sometimes you’ll want to use packages that are properly arranged with 
#setuptools, but aren’t published to PyPI. In those cases, you can specify a 
#list of one or more dependency_links URLs where the package can be downloaded,
#along with some additional hints, and setuptools will find and install the
#package correctly.
#For example, if a library is published on GitHub, you can specify it like:
setup(
    ...
    dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0']
    ...
)
"""