# -*- mode: python ; coding: utf-8 -*-

import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)

dask_datas = []

dask_files = [
    ('Lib/site-packages/dask/dask.yaml', 'dask'),
    ('Lib/site-packages/distributed/distributed.yaml', 'distributed')
]

for p in sys.path:
    for d, target in dask_files:
        path = os.path.join(p, d)
        if os.path.exists(path):
            dask_datas.append((path, target))

libertem_datas = [('../../client', 'libertem')]

datas = []

try:
    import opentelemetry.sdk
    datas += copy_metadata('opentelemetry.sdk')
except ModuleNotFoundError:
    pass

a = Analysis(
    ['libertem-server.py'],
    pathex=[],
    binaries=[],
    datas=datas + dask_datas + libertem_datas,
    hiddenimports=[],
    hookspath=['.\\extra-hooks\\'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='libertem-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='libertem-server',
)
