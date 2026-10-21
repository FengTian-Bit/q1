from setuptools import find_packages
from distutils.core import setup

setup(
    name="wheel_legged_gym",
    version="1.0.0",
    author="Hongxi Wang",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="wanghongxi2001@outlook.com",
    description="Isaac Gym environments for Wheel Legged Robots",
    install_requires=[
        "isaacgym",
        "matplotlib",
        "tensorboard",
        "imageio-ffmpeg>=0.4.3,<0.6",
        "swanlab==0.7.20; python_version < '3.9'",
        "swanlab>=0.10.2,<0.11; python_version >= '3.9'",
        "setuptools==59.5.0",
        "numpy>=1.16.4",
        "numpy<1.20.0",
        "GitPython",
        "onnx",
    ],
)
