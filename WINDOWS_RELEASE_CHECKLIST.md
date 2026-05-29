# Windows Release Checklist

Use this checklist on a clean Windows machine or a clean workspace before marking the build as a release candidate.

## 1. Prepare Clean Environment

- [ ] Open PowerShell in the project root.
- [ ] Create a clean virtual environment:

```powershell
py -m venv .venv-release
```

- [ ] Activate it:

```powershell
.\.venv-release\Scripts\Activate.ps1
```

- [ ] Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install Dependencies

- [ ] Install project dependencies:

```powershell
pip install -r requirements.txt
```

- [ ] Install PyInstaller:

```powershell
pip install pyinstaller
```

- [ ] Confirm PyInstaller is available:

```powershell
pyinstaller --version
```

## 3. Run Tests

- [ ] Run unit tests:

```powershell
py -m unittest discover -s tests -v
```

- [ ] Run compile checks:

```powershell
py -m compileall -q main.py ai core rag sandbox ui widgets comfy tests
```

- [ ] Validate the spec file:

```powershell
py -m py_compile ZenAI.spec
```

## 4. Build Windows App

- [ ] Build with PyInstaller:

```powershell
pyinstaller --clean --noconfirm ZenAI.spec
```

- [ ] Confirm the executable exists:

```powershell
Test-Path .\dist\ZenAI\ZenAI.exe
```

Expected result:

```text
True
```

## 5. Prepare Runtime Models

- [ ] Create a `models` folder next to the executable:

```powershell
New-Item -ItemType Directory -Force .\dist\ZenAI\models
```

- [ ] Copy a small test GGUF model into:

```text
dist\ZenAI\models\
```

- [ ] Confirm GGUF models are not bundled elsewhere inside the build.

## 6. Launch Packaged App

- [ ] Start the packaged app:

```powershell
.\dist\ZenAI\ZenAI.exe
```

- [ ] Open or create a test project.

## 7. Manual Verification

- [ ] The application starts successfully.
- [ ] Opening a project does not create `.zen_ai` inside that project.
- [ ] `%APPDATA%\ZenAI` is created.
- [ ] `%APPDATA%\ZenAI\chats` is created.
- [ ] `%APPDATA%\ZenAI\sessions` is created.
- [ ] `%APPDATA%\ZenAI\memory` is created.
- [ ] `%APPDATA%\ZenAI\settings` is created.
- [ ] Profile selection works.
- [ ] Selected profile is preserved after restart.
- [ ] Coder answer appears immediately during/after generation.
- [ ] Sending a message saves chat history.
- [ ] Restarting the app restores the last chat.
- [ ] Restored chat preserves message order and roles.
- [ ] RAG cache is written under `%APPDATA%\ZenAI\memory`.
- [ ] `models` are not included inside bundled assets or Python archive.
- [ ] The app loads GGUF models from `dist\ZenAI\models`.

## 8. Release Candidate Decision

Mark the release candidate as ready only if all checks above pass.

Release candidate status:

- [x] Ready
- [ ] Blocked

Blocking notes:

```text
The transformers\models startup blocker was fixed in the 2026-05-27
packaging fix below.

The final packaged GPU agent UI smoke test found a new blocker:
after edit_file successfully changed hello.py to print("hello world"),
the agent repeated the same edit_file call four more times, received
[error: old_str not found], and displayed
[Агент остановлен: достигнут лимит итераций].

The file result is correct, terminal execution succeeds, and persistence
restores the chat, but the agent edit flow is not release-ready.

Additionally, `agent_confirmation_policy` was `auto_confirm`, but the
`run_terminal` tool executed without a visible `Agent confirmation` dialog
during the packaged UI test. Confirm the intended terminal confirmation
policy before release.
```

## Actual Portable Archive Run - 2026-05-27 (llama_cpp Packaging Fix)

Build command:

```powershell
.\.venv-release\Scripts\pyinstaller.exe --clean --noconfirm ZenAI.spec
```

Result:

- [x] Unit tests passed: `15 tests OK`.
- [x] Compile check passed: `py -m compileall -q main.py ai core rag sandbox ui widgets comfy tests`.
- [x] PyInstaller `onedir` build completed successfully.
- [x] Executable produced: `dist\ZenAI\ZenAI.exe`.
- [x] `llama_cpp` native runtime is present under `dist\ZenAI\_internal\llama_cpp\lib`.
- [x] Packaged executable remained running after startup validation and was closed after the check.
- [x] Portable archive produced: `ZenAI-v0.1.0-win64.zip`.
- [x] Archive contains `ZenAI.exe`, `_internal\`, and bundled assets.
- [x] Archive contains five `llama_cpp` runtime DLLs under `_internal\llama_cpp\lib`.
- [x] Archive contains no top-level `models\` folder and no `.gguf` files.
- [x] Archive contains no `.venv*`, `build\`, `.git\`, or `__pycache__\` entries.

Artifact details:

```text
Archive: ZenAI-v0.1.0-win64.zip
Size:    289711835 bytes
```

Known issues:

- PyInstaller emitted non-blocking optional dependency warnings for `torch.utils.tensorboard` and `scipy.special._cdflib`.
- A full end-to-end packaged run with a real GGUF model remains a manual verification step on the user's local model environment.
- Superseded by the 2026-05-27 GPU-ready run below: the earlier archive used a CPU-only `llama-cpp-python` runtime and must not be published.

## Actual GPU-Ready Portable Archive Run - 2026-05-27

Environment:

```text
Dev Python:     D:\Zen Ai Editor\.venv\Scripts\python.exe
Release Python: D:\Zen Ai Editor\.venv-release\Scripts\python.exe
llama-cpp-python: 0.3.23 in both environments
Release wheel:  llama-cpp-python 0.3.23 from the CUDA 12.4 wheel index
```

Build command:

```powershell
.\.venv-release\Scripts\pyinstaller.exe --clean --noconfirm ZenAI.spec
```

Result:

- [x] Release venv contains `llama_cpp\lib\ggml-cuda.dll`.
- [x] Release venv reports `llama_supports_gpu_offload() == True`.
- [x] Unit tests passed: `30 tests OK`.
- [x] Compile check passed: `py -m compileall -q main.py ai core rag sandbox ui widgets comfy tests`.
- [x] PyInstaller `onedir` build completed successfully.
- [x] Executable produced: `dist\ZenAI\ZenAI.exe`.
- [x] Packaged executable stayed running during startup validation and was closed after the check.
- [x] `dist\ZenAI\_internal\llama_cpp\lib\ggml-cuda.dll` is present.
- [x] CUDA runtime DLLs are present in `dist\ZenAI\_internal`: `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`.
- [x] Packaged CUDA DLLs were loaded successfully with `ctypes` before model smoke testing.
- [x] Active coder profile has `n_gpu_layers=-1`.
- [x] GPU-gated release-runtime smoke test passed: simple coder prompt returned an answer.
- [x] GPU-gated release-runtime agent smoke test passed: `write_file` created `hello.py` containing `print('hi')`.
- [x] Smoke test printed `ggml_cuda_init` for `NVIDIA GeForce RTX 5070 Ti`.
- [x] Portable archive recreated: `ZenAI-v0.1.0-win64.zip`.
- [x] Archive contains `ZenAI.exe`, `_internal\`, bundled assets, `ggml-cuda.dll`, `cudart64_12.dll`, `cublas64_12.dll`, and `cublasLt64_12.dll`.
- [x] Archive contains no `models\` folder and no `.gguf` files.
- [x] Archive contains no `.venv*`, `build\`, `.git\`, or `__pycache__\` entries.

Artifact details:

```text
Archive: ZenAI-v0.1.0-win64.zip
Size:    909223317 bytes
```

Known issues:

- PyInstaller still emits non-blocking optional dependency warnings for `torch.utils.tensorboard` and `scipy.special._cdflib`.
- The automated model smoke test used the GPU-confirmed release venv/source harness because `ZenAI.exe` does not expose a CLI prompt mode. Packaged startup and packaged CUDA DLL loading were verified separately.

## Final Manual UI Smoke Test - 2026-05-27 (Superseded Startup Failure)

Setup:

```text
Archive: ZenAI-v0.1.0-win64.zip
Extracted to: C:\ZenAI_RC
Model copied to: C:\ZenAI_RC\models\qwen2.5-coder-14b-instruct-q4_k_m.gguf
```

Result:

- [x] Archive extracted into a clean `C:\ZenAI_RC` directory.
- [x] `C:\ZenAI_RC\models` was created next to `ZenAI.exe`.
- [x] `qwen2.5-coder-14b-instruct-q4_k_m.gguf` was copied into that `models` directory.
- [x] No `.gguf` files were found under `C:\ZenAI_RC\_internal`.
- [ ] `ZenAI.exe` opened the main UI.
- [ ] Coder/agent UI commands were tested.
- [ ] Restart restore was tested.
- [ ] v0.1.0 was marked release-ready.

Failure:

```text
Window title:
Unhandled exception in script

Dialog message:
Failed to execute script 'main' due to unhandled exception:
[WinError 3] Системе не удается найти указанный путь:
'C:\ZenAI_RC\_internal\transformers\models'

Traceback tail:
File "transformers\__init__.py", line 792, in <module>
  import_structure = define_import_structure(Path(__file__).parent / "models", prefix="models")
File "transformers\utils\import_utils.py", line 3022, in define_import_structure
  import_structure = create_import_structure_from_path(module_path)
File "transformers\utils\import_utils.py", line 2734, in create_import_structure_from_path
  with os.scandir(module_path) as entries:
FileNotFoundError: [WinError 3] Системе не удается найти указанный путь:
'C:\ZenAI_RC\_internal\transformers\models'
```

Notes:

- The failure happens before model loading and before the main UI is usable.
- `%APPDATA%\ZenAI\logs\zenai.log` did not receive a packaged startup entry for this run; the latest entries were from earlier dev/release-harness smoke tests.
- GPU/CPU load behavior was not tested in the UI because startup failed first.
- No code was changed and the archive was not rebuilt during this smoke test.
- Superseded by the packaging fix below.

## Packaged transformers models Fix - 2026-05-27

Root cause:

```text
PyInstaller bundled transformers enough for imports from the Python archive,
but transformers performs an import-time os.scandir(Path(__file__).parent / "models").
The physical _internal\transformers\models directory was missing from the archive.
```

Spec changes:

```text
collect_submodules("transformers")
collect_data_files("transformers")
collect_data_files("transformers", include_py_files=True, subdir="models")
collect_submodules("sentence_transformers")
collect_data_files("sentence_transformers")
```

Build command:

```powershell
.\.venv-release\Scripts\pyinstaller.exe --clean --noconfirm ZenAI.spec
```

Result:

- [x] PyInstaller `onedir` build completed successfully.
- [x] `dist\ZenAI\_internal\transformers\models` exists.
- [x] `dist\ZenAI\_internal\transformers\models` contains 2119 files.
- [x] `dist\ZenAI\_internal\sentence_transformers` exists.
- [x] `dist\ZenAI\_internal\llama_cpp\lib\ggml-cuda.dll` is still present.
- [x] `dist\ZenAI` contains no top-level `models\` directory.
- [x] `dist\ZenAI` contains no `.gguf` files.
- [x] Unit tests passed: `30 tests OK`.
- [x] Compile check passed: `py -m compileall -q main.py ai core rag sandbox ui widgets comfy tests`.
- [x] `dist\ZenAI\ZenAI.exe` opened the main window: `Zen AI Editor — PDF Combine`.
- [x] Fresh archive recreated: `ZenAI-v0.1.0-win64.zip`.
- [x] Archive contains `_internal\transformers\models` with 2119 files.
- [x] Archive contains `_internal\sentence_transformers`.
- [x] Archive contains CUDA runtime files and no `.gguf` files.

Clean archive startup smoke:

```text
Archive extracted to: C:\ZenAI_RC
Model copied to: C:\ZenAI_RC\models\qwen2.5-coder-14b-instruct-q4_k_m.gguf
Main window: Zen AI Editor — PDF Combine
Window class: ZenEditor
Responding: true
```

Artifact details:

```text
Archive: ZenAI-v0.1.0-win64.zip
Size:    922291480 bytes
```

Remaining manual release step:

```text
Run the final packaged coder/agent UI command flow with the real model:
1. create hello.py with print('hi')
2. edit hello.py to print hello world
3. run python hello.py
4. explain the changes
```

## Final GPU Agent UI Smoke Test - 2026-05-27 (Blocked)

Setup:

```text
Archive:      ZenAI-v0.1.0-win64.zip
Extracted to: C:\ZenAI_RC
Model:        C:\ZenAI_RC\models\qwen2.5-coder-14b-instruct-q4_k_m.gguf
Project:      C:\ZenAI_RC_TestProject
```

Runtime diagnostics from `%APPDATA%\ZenAI\logs\zenai.log`:

```text
exe=C:\ZenAI_RC\ZenAI.exe
frozen=True
gpu_backend_files=['ggml-cuda.dll']
gpu_backend_present=True
model_path=C:\ZenAI_RC\models\qwen2.5-coder-14b-instruct-q4_k_m.gguf
models_dir=C:\ZenAI_RC\models
n_gpu_layers=-1
```

Scenario result:

- [x] Packaged app opened the clean test project.
- [x] `создай файл hello.py и запиши print('hi')` created `hello.py` containing `print("hi")`.
- [ ] `измени hello.py, чтобы он печатал hello world` completed cleanly.
- [x] The file was changed to `print("hello world")`.
- [x] `выполни python hello.py` executed through `run_terminal`.
- [x] Stored terminal tool-card output contains `hello world` and `[exit 0]`.
- [x] `объясни, что сделал` produced a visible final assistant answer.
- [x] After normal close and restart, the chat and final answer were restored.
- [x] After restart, `hello.py` remained in the project.
- [x] `C:\ZenAI_RC_TestProject\models` was not created.
- [x] `C:\ZenAI_RC_TestProject\.zen_ai` was not created.
- [x] `%APPDATA%\ZenAI\chats`, `sessions`, `memory`, `settings`, and `logs` exist.

Observed GPU/resource behavior:

```text
GPU utilization during generation: up to 99%
GPU memory during generation:      approximately 12.9 GB
ZenAI CPU utilization sampled:     up to 4.8%
ZenAI RAM sampled:                 approximately 9.7 GB maximum
Application responding:            true throughout the sampled flow
```

Blocking symptom:

```text
On the edit command, the first edit_file call succeeded and updated hello.py.
The agent then repeated edit_file against already changed content four times.

Log:
[agent_tool_error] edit_file: [error: old_str not found]
[agent_tool_error] edit_file: [error: old_str not found]
[agent_tool_error] edit_file: [error: old_str not found]
[agent_tool_error] edit_file: [error: old_str not found]

Visible assistant message:
[Агент остановлен: достигнут лимит итераций]
```

Additional observation:

```text
Configured policy in %APPDATA%\ZenAI\settings\settings.json:
agent_confirmation_policy = "auto_confirm"

Observed behavior:
run_terminal executed successfully, but no Agent confirmation dialog was
presented during the UI flow.
```

## Dev/Source Agent Policy Smoke Test - 2026-05-27

Environment:

```text
Python:       D:\Zen Ai Editor\.venv\Scripts\python.exe
llama_cpp:    D:\Zen Ai Editor\.venv\Lib\site-packages\llama_cpp\lib
CUDA backend: ggml-cuda.dll
Model:        D:\Zen Ai Editor\models\qwen2.5-coder-14b-instruct-q4_k_m.gguf
n_gpu_layers: -1
```

Result:

- [x] Clean temporary test project opened through the source UI flow.
- [x] Agent-coder showed a short plan before the first file mutation.
- [x] `создай файл hello.py и запиши print('hi')` created `hello.py`.
- [x] `write_file` was followed by verification `read_file`.
- [x] `измени hello.py, чтобы он печатал hello world` read `hello.py` before `edit_file`.
- [x] `edit_file` changed the file to `print('hello world')`.
- [x] `edit_file` was followed by verification `read_file`.
- [x] `выполни python hello.py` executed through a real `run_terminal` tool call.
- [x] Terminal tool output contained `hello world` and `[exit 0]`.
- [x] `объясни, что сделал` produced a normal final assistant response.
- [x] No parser/tool errors were recorded in the final smoke run.
- [x] The agent did not hit `max_agent_steps`.
- [x] No continuation state remained after the successful final answer.

Notes:

```text
This validates the source/dev agent policy fixes only. The portable archive was
not rebuilt for this check, so the packaged release remains blocked until the
archive is rebuilt and the packaged UI smoke test is repeated.
```

## Final GPU-Ready Archive Rebuild After Agent Policy Fixes - 2026-05-27

Build environment:

```text
Release Python: D:\Zen Ai Editor\.venv-release\Scripts\python.exe
PyInstaller:    D:\Zen Ai Editor\.venv-release\Scripts\pyinstaller.exe
Archive:        ZenAI-v0.1.0-win64.zip
```

Commands:

```powershell
py -m unittest discover -s tests -v
py -m compileall -q main.py ai core rag sandbox ui widgets comfy tests
.\.venv-release\Scripts\pyinstaller.exe --clean --noconfirm ZenAI.spec
```

Build result:

- [x] Unit tests passed: `45 tests OK`.
- [x] Compile check passed.
- [x] PyInstaller `onedir` build completed successfully.
- [x] `dist\ZenAI\ZenAI.exe` exists.
- [x] `dist\ZenAI\_internal\llama_cpp\lib\ggml-cuda.dll` exists.
- [x] `dist\ZenAI\_internal\transformers\models` exists and contains 2119 files.
- [x] CUDA runtime DLLs are present: `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`.
- [x] `dist\ZenAI\models` was not created.
- [x] No `.gguf` files were found in `dist\ZenAI`.

Archive result:

- [x] Fresh `ZenAI-v0.1.0-win64.zip` created from `dist\ZenAI`.
- [x] Archive contains `ZenAI.exe`.
- [x] Archive contains `_internal\`.
- [x] Archive contains `_internal\llama_cpp\lib\ggml-cuda.dll`.
- [x] Archive contains CUDA runtime DLLs.
- [x] Archive contains `_internal\transformers\models`.
- [x] Archive contains no top-level `models\`.
- [x] Archive contains no `.gguf` files.

Artifact details:

```text
Archive: ZenAI-v0.1.0-win64.zip
Size:    922308071 bytes
Entries: 5498
```

Packaged UI smoke test:

```text
Extracted to: C:\ZenAI_RC
Model:        C:\ZenAI_RC\models\qwen2.5-coder-14b-instruct-q4_k_m.gguf
Project:      C:\ZenAI_RC_TestProject
Window:       Zen AI Editor — ZenAI_RC_TestProject
```

Runtime diagnostics:

```text
exe=C:\ZenAI_RC\ZenAI.exe
frozen=True
gpu_backend_files=['ggml-cuda.dll']
gpu_backend_present=True
llama_cpp_lib_dir=C:\ZenAI_RC\_internal\llama_cpp\lib
model_path=C:\ZenAI_RC\models\qwen2.5-coder-14b-instruct-q4_k_m.gguf
models_dir=C:\ZenAI_RC\models
n_gpu_layers=-1
```

Scenario result:

- [x] `создай файл hello.py и запиши print('hi')` created `hello.py`.
- [x] `write_file` was followed by verification `read_file`.
- [x] `измени hello.py, чтобы он печатал hello world` read `hello.py` before `edit_file`.
- [x] `edit_file` changed the file to `print('hello world')`.
- [x] `edit_file` was followed by verification `read_file`.
- [x] `выполни python hello.py` executed through `run_terminal`.
- [x] Terminal output contains `hello world` and `[exit 0]`.
- [x] No parser/tool errors were recorded.
- [x] The agent did not hit `max_agent_steps`.
- [x] Final packaged GPU-ready archive is release-ready for v0.1.0.

Conclusion:

```text
GPU packaged runtime, terminal execution, fixed models directory, and session
restore are verified. v0.1.0 is not release-ready because agent editing enters
an unnecessary repeated-tool loop after a successful edit; terminal
confirmation behavior should also be verified before release.
```
