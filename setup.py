from setuptools import setup, find_packages

setup(
    name='MasterThesisProject',
    version='0.0.1',
    author='Isabel Chaves',
    author_email='isabelpchaves@gmail.com',
    url='https://github.com/isabelchaves/MasterThesisProject',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.5, <4',
    install_requires=open('requirements.txt').readlines(),
)
