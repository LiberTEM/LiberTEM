from setuptools import setup

setup(
    name="libertem",
    version="0.0",
    author_email="a.clausen@fz-juelich.de",
    author="Alexander Clausen",
    include_package_data=True,
    install_requires=[
        "numpy",
        "distributed",
        "click",
    ],
    extras_require={
        'hdfs': 'hfds3',
        'hdf5': 'h5py',
    },
    package_dir={"": "src"},
    packages=[
        "libertem",
        "libertem.dataset",
        "libertem.executor",
        "libertem.job",
    ],
    entry_points={
        'console_scripts': [
            'libertem-ingest=libertem.ingest.cli:main',
        ]
    },
    classifiers=[
    ],
)
