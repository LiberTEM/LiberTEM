#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil


def main():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(BASE_DIR, "../.mypy-checked")) as f:
        files_project = set(f.read().split())
        print(files_project)

    files_cmdline = set(sys.argv[1:])
    print(files_cmdline)

    files = files_cmdline.intersection(files_project)
    print(files)

    if not files:
        print("nothing to check")
        return

    subprocess.run([shutil.which("mypy")] + list(files), check=True)


if __name__ == "__main__":
    main()
