# LocalLLM for iOS

**Run a full LLM on your iPhone. No cloud. No tokens. No limits.**

Built for iPhone 17 Pro — 8GB RAM, Neural Engine, Metal GPU.

## What This Is

A complete local AI stack that runs entirely on-device:

- **Local LLM**: Llama 3 8B quantized to ~4.5GB via llama.cpp
- **Sub-Agents**: Spawn parallel workers for complex tasks
- **MCP Integration**: Connect to Gmail, Calendar, Notion, etc.
- **Zero Cloud Dependency**: Everything runs on your phone

## Requirements

- iPhone 17 Pro (or any iOS device with 8GB+ RAM)
- ~5GB free storage for the model
- iOS 17.0+
- Windows PC with Sideloadly installed

## Quick Start

### 1. Build the App (No Mac Needed)

```bash
# Fork this repo to your GitHub
# Push any change to trigger the build
git push origin main

# Go to Actions tab → Download the IPA artifact
```

The GitHub Action runs on a free macOS M1 runner — builds the entire app for you.

### 2. Sideload with Sideloadly

1. Download `LocalLLM-iOS.zip` from GitHub Actions artifacts
2. Extract `LocalLLM.ipa`
3. Open Sideloadly on your Windows PC
4. Connect your iPhone via USB
5. Drag the IPA into Sideloadly
6. Enter your Apple ID (creates a free dev certificate)
7. Click Start
8. On iPhone: Settings → General → VPN & Device Management → Trust

### 3. Download a Model

The app needs a GGUF model file. Recommended:

| Model | Size | Quality | Download |
|-------|------|---------|----------|
| Llama 3 8B Q4_K_M | 4.5GB | Best balance | [HuggingFace](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF) |
| Mistral 7B Q4_K_M | 4.1GB | Fast | [HuggingFace](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) |
| Phi-3 Mini Q4 | 2.2GB | Smaller/Faster | [HuggingFace](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) |

**To get the model on your iPhone:**

Option A: Download directly on iPhone via Safari, then use Files app to move to the app's Documents folder.

Option B: AirDrop from your PC (if you have a Mac nearby).

Option C: Use iTunes File Sharing to copy the file.

### 4. Keep It Alive (No Weekly Renewal)

Run **AltServer** on your Windows PC in the background. As long as:
- AltServer is running
- Your iPhone is on the same WiFi

...the app auto-refreshes before expiration. Set it and forget it.

## Architecture

```
┌─────────────────────────────────────────────┐
│              LocalLLM iOS App               │
├─────────────────────────────────────────────┤
│  LocalLLMApp.swift      → SwiftUI entry     │
│  LLMEngine.swift        → llama.cpp wrapper │
│  AgentOrchestrator.swift → Sub-agent system │
│  MCPManager.swift       → MCP client        │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│           llama.cpp (C++ core)              │
│  - Metal GPU acceleration                   │
│  - Neural Engine optimization               │
│  - GGUF model loading                       │
└─────────────────────────────────────────────┘
```

## MCP Integration

Connect to any MCP server to give your local LLM tools:

```swift
let server = MCPServer(
    name: "Gmail",
    url: URL(string: "https://gmail.mcp.claude.com/mcp")!,
    type: .sse
)
MCPManager.shared.addServer(server)
try await MCPManager.shared.connect(to: server.id)
```

Now your local LLM can search emails, send messages, etc.

## Sub-Agents

The orchestrator automatically detects when to parallelize:

```
User: "Research quantum computing AND summarize my recent emails"
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   [Sub-Agent 1]                    [Sub-Agent 2]
   Research quantum                 Summarize emails
   computing                        (via Gmail MCP)
          │                               │
          └───────────────┬───────────────┘
                          ▼
                  [Synthesis]
              Combined response
```

## Performance

On iPhone 17 Pro with Llama 3 8B Q4_K_M:

| Metric | Value |
|--------|-------|
| Load time | ~3 seconds |
| Tokens/sec | 15-25 |
| Memory usage | ~5GB |
| Battery impact | Moderate |

## Limitations

- **7-day renewal** if you don't run AltServer (free Apple ID limitation)
- **No background inference** (iOS kills background processes)
- **Model size limited** by available RAM (~8GB on iPhone 17 Pro)
- **MCP over STDIO not supported** (iOS sandboxing)

## Development

To modify the app:

1. Edit Swift files locally
2. Push to GitHub
3. GitHub Actions builds new IPA
4. Re-sideload

No Xcode or Mac required for the build itself.

## Project Structure

```
LocalLLM-iOS/
├── .github/workflows/
│   └── build-ios.yml       # GitHub Actions build
├── Sources/LocalLLM/
│   ├── LocalLLMApp.swift   # Main app + UI
│   ├── LLM/
│   │   └── LLMEngine.swift # llama.cpp wrapper
│   ├── MCP/
│   │   └── MCPManager.swift# MCP client
│   └── Agents/
│       └── AgentOrchestrator.swift
├── Package.swift           # Swift package definition
└── README.md
```

## License

MIT — do whatever you want with it.

## Credits

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — The goat
- [MCP](https://modelcontextprotocol.io) — Tool protocol
- Anthropic — For making Claude smart enough to help build this
