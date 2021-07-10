import codecs
import os.path
from setuptools import find_packages
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r', encoding='utf-8') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1].strip()
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='mppi_train',
    version=get_version('mppi_train/__init__.py'),
    description='A package for training a neural network dynamic model of a robot',
    long_description_content_type='text/markdown',
    author='Kostya Yamshanov',
    author_email='k.yamshanov@fastsense.tech',
    url='',
    packages=find_packages(),
    license="MIT",
    install_requires=[
        'numpy',
        'torch',
        'tqdm',
        'wandb',
        'pandas',
    ]
)