# ZenAI Windows Update Strategy

## Recommended Release Format

ZenAI should be distributed as a portable `zip` archive built from the PyInstaller `onedir` output.

Recommended package shape:

```text
ZenAI/
  ZenAI.exe
  _internal/
  assets and bundled runtime files
```

Do not use PyInstaller `onefile` for the primary Windows release at this stage.

## Persistent User Data

User data lives outside the application folder:

```text
%APPDATA%\ZenAI\
  chats\
  sessions\
  memory\
  settings\
```

Updating the portable app folder must not delete or rewrite `%APPDATA%\ZenAI`.

This keeps chat history, sessions, profile selection, settings, and RAG cache independent from application updates.

## GGUF Models

GGUF models are external runtime data and are not included in the update archive.

Packaged Windows builds look for models here:

```text
ZenAI\
  ZenAI.exe
  models\
    model-name.gguf
```

The `models/` folder must stay next to `ZenAI.exe`.

Update archives should not contain user GGUF models. This keeps release archives smaller and prevents updates from deleting or replacing local model files.

## Manual Update Flow

Recommended update procedure:

1. Close ZenAI.
2. Download the new portable `zip`.
3. Extract the new files over the existing ZenAI application folder.
4. Keep the existing `models/` folder.
5. Start `ZenAI.exe`.
6. Confirm chats, profiles, settings, and model selection still load.

Important:

- Do not delete `%APPDATA%\ZenAI`.
- Do not delete `ZenAI\models`.
- Do not move GGUF files into `_internal/`.

## Missing Models Behavior

If `models/` is missing or empty, the application should not create a project-local `models/` folder.

Expected behavior:

- profile/model picker shows an empty state;
- model loading fails with a clear message that GGUF files must be placed in `models/` next to `ZenAI.exe`;
- opened projects never become the model root.

## Why Not PyInstaller Onefile Yet

PyInstaller `onefile` is not the recommended release format right now because:

- startup is slower due to extraction on each launch;
- asset and runtime paths are more fragile;
- PyQt and `llama-cpp-python` packaging is easier to diagnose in `onedir`;
- native DLL and runtime issues are easier to inspect when files are visible;
- logs, dependency inspection, and user support are simpler with a portable folder.

`onedir` is the safer release format until the runtime behavior is stable across more user machines.

## Future Option

A built-in updater can be added later.

Possible future flow:

1. App checks release metadata.
2. User confirms update.
3. Updater downloads portable archive.
4. App exits.
5. Updater replaces application files while preserving:
   - `%APPDATA%\ZenAI`;
   - `models/`;
   - user-created local files.
6. App restarts.

This should be implemented only after the portable zip release path is stable.
