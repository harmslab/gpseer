# Try using setuptools first, if it's installed
try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(name='gpseer',
      version='0.1',
      description='Python API that predicts unknown phenotypes in a genotype-phenotype map from known phenotypes.',
      author='Zach Sailer',
      author_email='zachsailer@gmail.com',
      packages=['gpseer'],
      install_requires=[
          'numpy',
          'h5py',
          'gpmap',
          'epistasis'
      ],
      zip_safe=False)
