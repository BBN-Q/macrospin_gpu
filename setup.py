from setuptools import setup

setup(
    name='macrospin_gpu',
    version='0.1',
    author='Graham Rowlands',
    package_dir={'':'src'},
    packages=[
        'macrospin_gpu'
    ],
    scripts=[],
    description='GPU-based macrospin simulations using OpenCL',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.11.1",
        "scipy >= 0.17.1",
        "pyopencl >= 2016.2.1",
        "pyopengl >= 3.1.0",
        "jinja2 >= 2.9.4",
        "tqdm >= 4.7.0"
    ]
)
