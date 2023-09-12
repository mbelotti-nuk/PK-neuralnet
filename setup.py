# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pkdnn',
    version='0.1.0',
    description='Neural network train for Point Kernel applications',
    long_description=readme,
    author='Mario Belotti',
    author_email='mbelotti@ind.uned.es',
    url='https://github.com/mbelotti-nuk/PK-neuralnet',
    license=license,
    #packages=find_packages(exclude=('tests', 'docs')),
    packages = ['pkdnn', 'pkdnn.functionalities', 'pkdnn.NET','pkdnn.EXPORT'],
    entry_points={'console_scripts': [
        'trainpknn=pkdnn.train:main',
        'predictpknn=pkdnn.predict:main',
        'savepknn=pkdnn.EXPORT.exportmodel:main']
        }
)

