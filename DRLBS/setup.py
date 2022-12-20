from setuptools import find_packages, setup

setup(name='DRLBS',
      version='0.0.1',
      install_requires=['gym', 'numpy', 'mat73'],
      packages=find_packages('src'),
      package_dir={'': 'src'})