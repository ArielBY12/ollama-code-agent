# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Ollama Code Agent TUI.

Produces a single-file Windows console executable.

Textual and Rich ship CSS/templates inside their packages; collect_all
walks those so they end up in the bundle instead of missing at runtime.
"""

from PyInstaller.utils.hooks import collect_all

datas: list = []
binaries: list = []
hiddenimports: list = []

for pkg in ("textual", "rich", "ollama"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
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
    a.binaries,
    a.datas,
    [],
    name="ollama-code-agent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
