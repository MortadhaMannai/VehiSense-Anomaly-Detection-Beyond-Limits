from setuptools import find_packages, setup

__version__ = '0.0.1'
URL = 'https://github.com/tbohne/oscillogram_classification'

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='oscillogram_classification',
    version=__version__,
    description='Neural network based anomaly detection for vehicle components using oscilloscope recordings.',
    author='Tim Bohne',
    author_email='tim.bohne@dfki.de',
    url=URL,
    download_url=f'{URL}/archive/{__version__}.tar.gz',
    keywords=[
        'deep-learning',
        'time-series',
        'neural-network',
        'classification',
        'anomaly-detection',
        'xai'
    ],
    python_requires='>=3.7, <3.11',
    install_requires=required,
    packages=find_packages(),
    include_package_data=True,
)
