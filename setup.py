#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="erdes",
    version="0.1.0",
    description="Benchmarking ERDES Dataset with 3D CNN and transformer-based architectures for ocular ultrasound classification",
    author="PCVLab",
    maintainer="Yasemin Ozkut",
    maintainer_email="ozkutyasemin@gmail.com",
    url="https://github.com/OSUPCVLab/ERDES",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "lightning>=2.0.0",
        "torchmetrics>=0.11.4",
        "hydra-core==1.3.2",
        "hydra-colorlog==1.2.0",
        "hydra-optuna-sweeper==1.2.0",
        "rootutils",
        "rich",
        "pandas",
        "scikit-learn",
        "pytorchvideo",
        "monai",
        "tensorboard",
        "einops",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_command = erdes.train:main",
            "eval_command = erdes.eval:main",
        ]
    },
)
