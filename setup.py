from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="quantum_hybrid_model",
    version="0.1",
    packages=find_packages(),
    install_requires=required,
    entry_points={
        "console_scripts": [
            "qenn=quantum_hybrid_model.main:cli",
        ],
    },
)
