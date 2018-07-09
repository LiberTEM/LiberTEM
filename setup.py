import os
import distutils
import subprocess
from setuptools import setup
import setuptools.command.build_py


class BuildPyCommand(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('build_client')
        self.run_command('copy_client')
        super(BuildPyCommand, self).run()


class BuildClientCommand(distutils.cmd.Command):
    description = 'build the js client'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cwd = os.path.dirname(__file__)
        cwd_client = os.path.join(cwd, 'client')
        self.announce(
            "building js client",
            level=distutils.log.INFO
        )
        for command in [['npm', 'install'],
                        ['npm', 'run-script', 'build']]:
            subprocess.check_call(command, cwd=cwd_client)


class CopyClientCommand(distutils.cmd.Command):
    description = 'copy the js client'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cwd = os.path.dirname(__file__)
        cwd_client = os.path.join(cwd, 'client')
        client = os.path.join(cwd, 'src', 'libertem', 'web', 'client')

        cmd = ["rm", "-rf", client]
        self.announce(
            "preparing output directory: %s" % cmd,
            level=distutils.log.INFO
        )
        subprocess.check_call(cmd)

        cmd = ["mkdir", client]
        self.announce(
            "creating output directory: %s" % cmd,
            level=distutils.log.INFO
        )
        subprocess.check_call(cmd)

        build = os.path.join(cwd_client, "build")
        cmd = "cp -r %s/* %s" % (build, client)
        self.announce(
            "copying client: %s" % cmd,
            level=distutils.log.INFO
        )
        subprocess.check_call(cmd, shell=True)


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
        "matplotlib",
        "pillow",
        "h5py",
    ],
    extras_require={
        'hdfs': 'hfds3',
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
    cmdclass={
        'build_py': BuildPyCommand,
        'build_client': BuildClientCommand,
        'copy_client': CopyClientCommand,
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
