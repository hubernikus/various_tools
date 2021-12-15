from setuptools import setup, find_packages

setup(
    name="vartools",
    version="0.1",
    description="Various Tools",
    author="Lukas Huber",
    author_email="lukas.huber@epfl.ch",
    # packages=find_packages(),
    scripts=[
        "scripts/main.py",
    ],
    package_dir={"": "src"},
)
