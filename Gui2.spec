# Gui2.spec
block_cipher = None

a = Analysis(
    ['Gui2.py'],
    pathex=[],
    binaries=[],
    datas=[
    ('voice_classifier_speech_final_no_noise_reduce2.0_20_pre_finalfcc.joblib', '.'),
    ('Voice_classifier_speech_final_no_noise_reduce2.0joblib', '.') if needed
	],
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
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VoiceClassifierPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Setzen Sie auf True, wenn Sie die Konsole sehen möchten
    icon='Voice Classifier Pro.ico',  # Optional: Pfad zu einem ICO-File
)