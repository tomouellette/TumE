from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = '0.1'

setup(name='TumE',
      version=version,
      python_requires='>3.6',
      description='Inferring cancer evolution using synthetic supervised learning',
      url='http://github.com/tomouellette/TumE',
      author='Tom W. Ouellette',
      author_email='t.ouellette@mail.utoronto.ca',
      license='MIT',
      packages=find_packages(),
      install_requires=requirements,
      include_package_data=True,
      )	
