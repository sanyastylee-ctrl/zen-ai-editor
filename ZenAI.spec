# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


project_root = Path(SPECPATH)
llama_cpp_binaries = collect_dynamic_libs("llama_cpp")
hiddenimports = []
hiddenimports += collect_submodules("llama_cpp")
hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("sentence_transformers")

datas = [(str(project_root / "assets"), "assets")]
datas += collect_data_files("transformers")
datas += collect_data_files("transformers", include_py_files=True, subdir="models")
datas += collect_data_files("sentence_transformers")

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=llama_cpp_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ZenAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name="ZenAI",
)
