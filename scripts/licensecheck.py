#!/usr/bin/env python3

import sys
import json

if __name__ == '__main__':
    deps = json.load(sys.stdin)
    mit_keys = [
        k for k in deps.keys()
        if k.startswith('libertem.io') or k.startswith('libertem.common')
    ]
    for mit_key in mit_keys:
        print(
            mit_key, [
                k for k in deps[mit_key].get('imports', [])
                if not k.startswith('libertem.common')
                and not k.startswith('libertem.io')
                and k != 'libertem'  # only contains version info
                and k.startswith('libertem')
            ]
        )
