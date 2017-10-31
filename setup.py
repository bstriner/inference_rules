from setuptools import setup, find_packages

setup(name='inference-rules',
      version='0.0.1',
      install_requires=['Keras','theano'],
      author='Ben Striner',
      url='https://github.com/bstriner/inference_rules',
      packages=find_packages())
