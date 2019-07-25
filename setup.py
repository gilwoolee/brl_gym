#import setuptools
from distutils.core import setup

setup(name='brl_gym',
        version='1.0',
        description='Bayesian RL Environments',
        author='Gilwoo Lee',
        author_email='gilwoo301@gmail.com',
        license='',
        package_data={'': ['resource/*.json']},
        include_package_data=True,
        packages=setuptools.find_packages(),
        install_requires=['gym','numpy', 'matplotlib'],
        )
