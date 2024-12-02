import os
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_git_rev():
    # NOTE: Also exists in src/libertem/versioning.py
    try:
        new_cwd = os.path.abspath(os.path.dirname(__file__))
        rev_raw = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=new_cwd)
        return rev_raw.decode("utf8").strip()
    except Exception:
        return "unknown"


def write_baked_revision(dest_dir):
    baked_dest = os.path.join(dest_dir, '_baked_revision.py')
    os.makedirs(dest_dir, exist_ok=True)
    with open(baked_dest, "w") as f:
        f.write(r'revision = "%s"' % get_git_rev())


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = 'custom'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, version, build_data):
        here = os.path.abspath(os.path.dirname(__file__))
        if not os.path.exists(os.path.join(here, '.git')):
            return  # not running from git
        if self.target_name not in ["wheel", "sdist"]:
            return  # something else
        write_baked_revision(
            os.path.join(here, "src", "libertem")
        )
