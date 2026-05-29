# Zen AI Editor — Pre-release Notes

## Added

- Persistence layer for application data.
- Windows AppData storage under `%APPDATA%\ZenAI\`:
  - `chats/` for chat JSON sessions;
  - `sessions/` for active chat/profile state and recent projects;
  - `memory/` for per-project RAG cache;
  - `settings/` for profiles and app settings.
- Chat/session/profile restore on application startup.
- Chat history autosave after user/model messages and agent tool updates.
- RAG cache migration from legacy project `.zen_ai/` into AppData-backed storage.
- Safe migration from legacy `~/.zen_ai` settings/profile/recent-project files without overwriting newer AppData files.
- PyInstaller `onedir` packaging preparation via `ZenAI.spec`.
- Packaged-runtime path handling for external `models/` next to `ZenAI.exe`.
- Invisible coder answer fix: single-chunk coder/agent responses now render immediately in the active chat, not only after profile switching.
- Runtime diagnostics written to `%APPDATA%\ZenAI\logs\zenai.log`.
- Agent safety limits: max 5 agent steps, max 10 tool calls, 180 second generation timeout, and context-size guard.
- Packaged startup fix for `llama_cpp/lib` runtime DLLs.
- Agent-coder policy hardening in source/dev mode: plan-before-tools, read-before-edit,
  verification reads after writes/edits, duplicate tool-call guard for terminal commands,
  and explicit-run fallback through `run_terminal` when the model omits the tool call.

## Verified

- Unit tests pass: `py -m unittest discover -s tests -v`.
- Static checks pass:
  - `py -m compileall -q main.py ai core rag sandbox ui widgets comfy tests`;
  - `py -m py_compile ZenAI.spec`;
  - `git diff --check`.
- Corrupt, empty, missing, and legacy chat/session JSON are handled without crashing.
- Chat restore preserves message order, roles, assistant content, history, and tool-card error output.
- Active profile restore works through the existing `ChatView` API.
- Repeated quick chat saves leave valid latest JSON.
- Agent/tool error cards persist with `ok=false` and output text.
- Attachments are cleared after sending and are not restored into later sessions.
- `generation_done` does not duplicate assistant messages.
- Worker state is cleared after completion in the tested UI flow.
- Runtime storage no longer creates `.zen_ai` inside the opened project during tested packaged/source scenarios.
- PyInstaller spec includes `assets/` and does not include `models/`.
- Frozen path behavior resolves models to `models/` beside `ZenAI.exe`.
- Agent loop stops on max-step and generation-timeout guards in unit tests.
- `n_gpu_layers` is passed into `Llama()` and is part of the model cache key.
- Diagnostics create `%APPDATA%\ZenAI\logs\zenai.log`.
- Dev/source GPU UI smoke test passed for agent-coder create/edit/run/explain flow with
  `qwen2.5-coder-14b-instruct-q4_k_m.gguf` and `n_gpu_layers=-1`.
- Rebuilt GPU-ready portable archive passed packaged UI smoke test for
  create/edit/run agent-coder flow with CUDA backend enabled.

## Known Limitations

- GGUF models are not bundled into the application build. They must be placed next to `ZenAI.exe` in a `models/` folder for packaged runs.
- Full validation with real large local models requires the user's local GPU/model environment.
- The current release environment's `llama-cpp-python` build contains CPU runtime DLLs only (`ggml-cpu.dll`, `llama.dll`, etc.). CUDA/Vulkan backend DLLs were not present, so this archive should be treated as CPU-only until rebuilt with a GPU-enabled `llama-cpp-python` wheel.
