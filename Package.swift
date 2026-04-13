// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "LocalLLM",
    platforms: [
        .iOS(.v17)
    ],
    products: [
        .executable(name: "LocalLLM", targets: ["LocalLLM"])
    ],
    dependencies: [],
    targets: [
        // Main app target
        .executableTarget(
            name: "LocalLLM",
            dependencies: ["LlamaKit", "MCPClient", "AgentOrchestrator"],
            path: "Sources/LocalLLM",
            resources: [
                .copy("Resources")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        
        // LLM inference wrapper around llama.cpp
        .target(
            name: "LlamaKit",
            dependencies: ["CLlama"],
            path: "Sources/LocalLLM/LLM"
        ),
        
        // C wrapper for llama.cpp
        .systemLibrary(
            name: "CLlama",
            path: "Dependencies/llama.cpp",
            pkgConfig: nil,
            providers: []
        ),
        
        // MCP client implementation
        .target(
            name: "MCPClient",
            dependencies: [],
            path: "Sources/LocalLLM/MCP"
        ),
        
        // Sub-agent orchestration
        .target(
            name: "AgentOrchestrator",
            dependencies: ["LlamaKit", "MCPClient"],
            path: "Sources/LocalLLM/Agents"
        )
    ]
)
