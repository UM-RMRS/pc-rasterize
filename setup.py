from setuptools import setup

with open("pc_rasterize/_version.py") as fd:
    version = fd.read().strip().split()[2].strip('"')

setup(version=version)
