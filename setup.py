from setuptools import setup

setup(
    name="libertem",
    version="0.0",
    author_email="a.clausen@fz-juelich.de",
    author="Alexander Clausen",
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "distributed",
        "click",
        "tornado",
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
        "libertem.web",
    ],
    entry_points={
        'console_scripts': [
            'libertem-ingest=libertem.ingest.cli:main',
            'libertem-server=libertem.web.cli:main',
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
