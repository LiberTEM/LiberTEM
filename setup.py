import re
import os
import codecs
import subprocess
import distutils
import shutil
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py
from setuptools import setup, find_packages


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
        npm = shutil.which('npm')
        for command in [[npm, 'install'],
                        [npm, 'run-script', 'build']]:
            self.announce(' '.join(command), distutils.log.INFO)
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

        self.announce(
            "preparing output directory: %s" % client,
            level=distutils.log.INFO
        )
        shutil.rmtree(client)

        build = os.path.join(cwd_client, "build")
        self.announce(
            "copying client: %s -> %s" % (build, client),
            level=distutils.log.INFO
        )
        shutil.copytree(build, client)


class BakedRevisionBuilderSdist(sdist):
    def make_release_tree(self, base_dir, files):
        if not self.dry_run:
            write_baked_revision(base_dir)
        sdist.make_release_tree(self, base_dir, files)


class BakedRevisionBuilderBuildPy(build_py):
    def run(self):
        if not self.dry_run:
            write_baked_revision(self.build_lib)
        build_py.run(self)


def mkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    here = os.path.abspath(os.path.dirname(__file__))
    return codecs.open(os.path.join(here, *parts), 'r').read()


def remove_rst_roles(txt):
    return re.sub(':(cite|doc):`[^`]+` ?', '', txt)


def get_git_rev():
    # NOTE: this is a copy from src/libertem/versioning.py
    # this is because it is not guaranteed that we can import our own packages
    # from setup.py AFAIK
    try:
        new_cwd = os.path.abspath(os.path.dirname(__file__))
        rev_raw = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=new_cwd)
        return rev_raw.decode("utf8").strip()
    except Exception:
        return "unknown"


def write_baked_revision(base_dir):
    dest_dir = os.path.join(base_dir, 'libertem')
    baked_dest = os.path.join(dest_dir, '_baked_revision.py')
    mkpath(dest_dir)

    with open(baked_dest, "w") as f:
        f.write(r'revision = "%s"' % get_git_rev())


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
    version=find_version("src", "libertem", "__version__.py"),
    license='GPL v3',
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6,!=3.9.0,!=3.9.1,!=3.9.2',
    install_requires=[
        "numpy",
        "scipy",
        "sparse",
        "distributed>=2.19.0",
        "click",
        "tornado>=5",
        "matplotlib",
        "pillow",
        "h5py",
        "psutil",
        # https://github.com/LiberTEM/LiberTEM/issues/763
        # https://github.com/LiberTEM/LiberTEM/issues/783
        "numba>=0.51",
        "ncempy>=1.4",
        'pywin32!=226;platform_system=="Windows"',
        # FIXME pull request #259
        # https://github.com/LiberTEM/LiberTEM/pull/259#discussion_r251877431
        'scikit-image',
        'cloudpickle',
        'jsonschema',
        'scikit-learn',
        'tqdm',
        'threadpoolctl',
        'primesieve',
        'nbformat',
        'nbconvert',
        'autopep8',
        'empyre>=0.3.0',
    ],
    extras_require={
        'torch': 'torch',
        'hdbscan': 'hdbscan',
        'cupy': 'cupy',
        'bqplot': ['bqplot', 'bqplot-image-gl', 'ipython'],
    },
    package_dir={"": "src"},
    packages=find_packages(where='src'),
    entry_points={
        'console_scripts': [
            'libertem-server=libertem.web.cli:main',
            'libertem-worker=libertem.executor.cli:main',
        ]
    },
    cmdclass={
        'build_client': BuildClientCommand,
        'copy_client': CopyClientCommand,
        'sdist': BakedRevisionBuilderSdist,
        'build_py': BakedRevisionBuilderBuildPy,
    },
    keywords="electron microscopy",
    description="Open pixelated STEM framework",
    long_description=remove_rst_roles(read("README.rst")),
    long_description_content_type="text/x-rst",
    url="https://libertem.github.io/LiberTEM/",
    author_email="libertem-dev@googlegroups.com",
    author="the LiberTEM team",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Environment :: Web Environment',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: JavaScript',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
)
