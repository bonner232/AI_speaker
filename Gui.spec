# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Gui.py'],
    pathex=[],
    binaries=[],
    datas=[('voice_classifier_speech_final_no_noise_reduce2.0_20_pre_finalfcc.joblib', '.')],
    hiddenimports=[
        'sklearn.ensemble._forest',
        'sklearn.utils._weight_vector',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.tree._splitter',
        'sklearn.tree._criterion',
        'sklearn.neighbors.quad_tree',
        'sklearn.neighbors._partition_nodes'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
	a.zipfiles,
    a.datas,
    [],
    name='WAV File Selector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
	icon='Voice Classifier Pro.ico',  # Optional: Pfad zu einem ICO-File
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
