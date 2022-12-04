#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = []

test_requirements = ['pytest>=3', ]

setup(
    author="Lukas Sykora",
    author_email='lukassykora@seznam.cz',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description="Python Implementation of Apriori Algorithm for Frequent Itemsets Mining.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyapriori',
    name='pyapriori',
    packages=find_packages(include=['pyapriori', 'pyapriori.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/lukassykora/pyapriori',
    version='0.1.0',
    zip_safe=False,
)
