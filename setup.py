from setuptools import setup

setup(
    name='tomography_toolkit_dev',
    version='0.1',
    description='Development package for QuantOm collaboration',
    author='Kishansingh Rajput, and others...',
    author_email='kishan@jlab.org',
    packages=['tomography_toolkit_dev'],
    install_requires=['numpy==1.21.6',
                      'torch==1.13.0',
                      'torchmetrics==0.11.0',
                      'matplotlib==3.6.1',
                      'horovod==0.19.5',
                      'unittest2==1.1.0']
)
