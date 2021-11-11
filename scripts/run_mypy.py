#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil


def main():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(BASE_DIR, "../.mypy-checked")) as f:
        files_project = set(f.read().split())

    for i, f in enumerate(files_project):
        if not os.path.exists(f):
            raise RuntimeError(f"File {f} on line {i} in .mypy-checked does not exist.")

    files_cmdline = set(sys.argv[1:])

    args = []
    if len(files_cmdline) == 0:
        files = files_project
    else:
        files = files_cmdline.intersection(files_project)
        args = [
            f
            for f in files_cmdline
            if f.startswith('--')
        ]

    if not files:
        print("nothing to check")
        return

    cmd = [shutil.which("mypy")] + list(files) + args
    print(f"running mypy: {' '.join(cmd)}")

    subprocess.run(
        cmd, check=True,
        cwd=os.path.join(BASE_DIR, '..'),
    )


if __name__ == "__main__":
    main()
