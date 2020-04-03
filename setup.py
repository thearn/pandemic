from distutils.core import setup
from setuptools import find_packages


setup(name='pandemic',
    version='0.0.1',
    description='Python based dynamical optimization of a pandemic disease model',
    url='https://github.com/OpenMDAO/pandemic',
    classifiers=[
        'Development Status :: Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache 2.0',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    license='Apache License',
    packages=find_packages(),
    install_requires=[
        'openmdao>=3.0.0',
        'dymos>=0.15.0'
    ],
    zip_safe=False,
)