import os
import subprocess


def get_git_rev():
    try:
        new_cwd = os.path.abspath(os.path.dirname(__file__))
        rev_raw = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=new_cwd)
        return rev_raw.decode("utf8").strip()
    except Exception:
        return "unknown"
