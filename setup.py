import os

from setuptools import setup, find_packages

__version__ = '0.1'




setup(
    name='process9_tts2',
    version=__version__,
    description='tts',
    # url='https://github.com/jupitermoney/ds-jm-fp-inference-service',
    maintainer='Data Science',
    maintainer_email='datascience@jupiter.money',
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    zip_safe=False,
    dependency_links=[],
    # install_requires=requirements,
    python_requires=">=3.7"
)