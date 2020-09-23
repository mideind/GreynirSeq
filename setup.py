from setuptools import setup, find_packages

setup(
    name='greynirseq',
    description='Natural language processing for Icelandic, using fairseq',
    version='0.01',
    package_dir={'': 'src'},
    packages=find_packages(where='src')
)
