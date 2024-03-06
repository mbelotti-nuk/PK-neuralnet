# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pkuned',
    version='0.1.0',
    description='Neural network train for Point Kernel applications',
    long_description=readme,
    author='Mario Belotti',
    author_email='mbelotti@ind.uned.es',
    url='https://github.com/mbelotti-nuk/PK-neuralnet',
    license=license,
    #packages=find_packages(exclude=('tests', 'docs')),
    packages = ['pknn', 'pknn.functionalities', 'pknn.net','pknn.export', 'pknn.inp_process','nacarte'],
    entry_points={'console_scripts': [
        'trainpknn=pknn.train:main',
        'predictpknn=pknn.predict:main',
        'savepknn=pknn.exportmodel:main',
        'databasepknn=pknn.build_database:main']
        },
    py_version='>=3.8',
    install_requires=['torch>=2.0.0','numpy','matplotlib','seaborn','scipy','leb128','PyYAML'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)

