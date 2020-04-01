#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup the package
To do:
    [type the aims and objectives of this script]
Created on Wed Apr  1 15:03:58 2020

@author: Xiaodong Ming
"""


from setuptools import setup

setup(name='hipims_io',
      version='0.1',
      description='To process input and output files of the HiPIMS model',
      url='https://github.com/mingxiaodong/hipims_io_py_package',
      author='Xiaodong Ming',
      author_email='xiaodong.ming@outlook.com',
      license='NCL',
      packages=['hipims_io'],
      install_requires=['gdal',],
      zip_safe=False)

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