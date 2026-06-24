# hook-opentelemetry.py
from PyInstaller.utils.hooks import collect_entry_point

datas, hiddenimports = collect_entry_point('opentelemetry_context')
