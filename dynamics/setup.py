#!/usr/bin/env python3
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from sympy import E

d = generate_distutils_setup()
d['packages'] = ['dynamics']
d['package_dir'] = {'': 'src'}

setup(**d)
e = generate_distutils_setup()
d['packages'] = ['drl']
e['package_dir'] = {'': 'src'}

setup(**e)
