import re
import os
import codecs
import distutils
import subprocess
from setuptools import setup


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
        self.run_command('copy_client')


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


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    here = os.path.abspath(os.path.dirname(__file__))
    return codecs.open(os.path.join(here, *parts), 'r').read()


def find_version(*file_paths):
    """
    "stolen" from pip's setup.py
    """
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="libertem",
    version=find_version("src", "libertem", "__init__.py"),
    url="https://libertem.github.io/LiberTEM/",
    author_email="a.clausen@fz-juelich.de",
    author="Alexander Clausen",
    license='GPL v3',
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "distributed>=1.23.3",
        "click",
        "tornado",
        "matplotlib",
        "pillow",
        "h5py",
        "psutil",
        "numba",
    ],
    extras_require={
        'hdfs': 'hfds3',
        'torch': 'torch',
    },
    package_dir={"": "src"},
    packages=[
        "libertem",
        "libertem.common",
        "libertem.io",
        "libertem.io.dataset",
        "libertem.executor",
        "libertem.job",
        "libertem.web",
        "libertem.analysis",
    ],
    entry_points={
        'console_scripts': [
            'libertem-ingest=libertem.io.ingest.cli:main',
            'libertem-server=libertem.web.cli:main',
        ]
    },
    cmdclass={
        'build_client': BuildClientCommand,
        'copy_client': CopyClientCommand,
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
    ],
)
