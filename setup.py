from setuptools import setup

setup(
    name="libertem",
    version="0.0",
    author_email="a.clausen@fz-juelich.de",
    author="Alexander Clausen",
    include_package_data=True,
    install_requires=[
        "numpy",
        "hdfs3",
        "distributed",
    ],
    package_dir={"": "src"},
    packages=[
        "libertem",
        "libertem.dataset",
        "libertem.executor",
        "libertem.job",
    ],
    classifiers=[
    ],
)
