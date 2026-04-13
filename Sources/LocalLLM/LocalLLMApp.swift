import SwiftUI

@main
struct LocalLLMApp: App {
    @StateObject private var llmEngine = LLMEngine.shared
    @StateObject private var agentOrchestrator = AgentOrchestrator.shared
    @StateObject private var mcpManager = MCPManager.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(llmEngine)
                .environmentObject(agentOrchestrator)
                .environmentObject(mcpManager)
                .preferredColorScheme(.dark)
        }
    }
}

// MARK: - Main Content View

struct ContentView: View {
    @EnvironmentObject var llmEngine: LLMEngine
    @EnvironmentObject var orchestrator: AgentOrchestrator
    
    @State private var inputText: String = ""
    @State private var conversation: [Message] = []
    @State private var isProcessing: Bool = false
    @State private var showSettings: Bool = false
    @State private var showMCPConfig: Bool = false
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Status bar
                StatusBar(
                    modelLoaded: llmEngine.isModelLoaded,
                    activeAgents: orchestrator.activeAgentCount,
                    mcpConnections: MCPManager.shared.activeConnections
                )
                
                // Conversation view
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(conversation) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                            
                            if isProcessing {
                                ThinkingIndicator()
                            }
                        }
                        .padding()
                    }
                    .onChange(of: conversation.count) { _, _ in
                        if let last = conversation.last {
                            withAnimation {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                }
                
                // Input area
                InputBar(
                    text: $inputText,
                    isProcessing: isProcessing,
                    onSend: sendMessage
                )
            }
            .navigationTitle("Local LLM")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showMCPConfig = true
                    } label: {
                        Image(systemName: "link.circle")
                    }
                }
                
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gear")
                    }
                }
            }
            .sheet(isPresented: $showSettings) {
                SettingsView()
            }
            .sheet(isPresented: $showMCPConfig) {
                MCPConfigView()
            }
            .task {
                await loadModel()
            }
        }
    }
    
    private func loadModel() async {
        do {
            try await llmEngine.loadModel()
        } catch {
            conversation.append(Message(
                role: .system,
                content: "⚠️ Failed to load model: \(error.localizedDescription)"
            ))
        }
    }
    
    private func sendMessage() {
        guard !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        let userMessage = Message(role: .user, content: inputText)
        conversation.append(userMessage)
        
        let prompt = inputText
        inputText = ""
        isProcessing = true
        
        Task {
            do {
                // Route through orchestrator - it decides if sub-agents are needed
                let response = try await orchestrator.process(
                    prompt: prompt,
                    conversation: conversation
                )
                
                await MainActor.run {
                    conversation.append(Message(role: .assistant, content: response))
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    conversation.append(Message(
                        role: .system,
                        content: "❌ Error: \(error.localizedDescription)"
                    ))
                    isProcessing = false
                }
            }
        }
    }
}

// MARK: - Supporting Views

struct StatusBar: View {
    let modelLoaded: Bool
    let activeAgents: Int
    let mcpConnections: Int
    
    var body: some View {
        HStack {
            Label(
                modelLoaded ? "Model Ready" : "Loading...",
                systemImage: modelLoaded ? "checkmark.circle.fill" : "circle.dashed"
            )
            .foregroundStyle(modelLoaded ? .green : .orange)
            
            Spacer()
            
            if activeAgents > 0 {
                Label("\(activeAgents) agents", systemImage: "person.2.fill")
                    .foregroundStyle(.blue)
            }
            
            if mcpConnections > 0 {
                Label("\(mcpConnections) MCP", systemImage: "link")
                    .foregroundStyle(.purple)
            }
        }
        .font(.caption)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}

struct MessageBubble: View {
    let message: Message
    
    var body: some View {
        HStack {
            if message.role == .user { Spacer() }
            
            VStack(alignment: message.role == .user ? .trailing : .leading) {
                if message.role == .system {
                    Text(message.content)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text(message.content)
                        .padding()
                        .background(
                            message.role == .user
                                ? Color.blue
                                : Color(.systemGray5)
                        )
                        .foregroundStyle(message.role == .user ? .white : .primary)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                
                if let agentInfo = message.agentInfo {
                    Text("via \(agentInfo)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: UIScreen.main.bounds.width * 0.75, alignment: message.role == .user ? .trailing : .leading)
            
            if message.role != .user { Spacer() }
        }
    }
}

struct InputBar: View {
    @Binding var text: String
    let isProcessing: Bool
    let onSend: () -> Void
    
    var body: some View {
        HStack(spacing: 12) {
            TextField("Message", text: $text, axis: .vertical)
                .textFieldStyle(.plain)
                .padding(12)
                .background(Color(.systemGray6))
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .lineLimit(1...5)
            
            Button(action: onSend) {
                Image(systemName: isProcessing ? "stop.fill" : "arrow.up.circle.fill")
                    .font(.title)
                    .foregroundStyle(text.isEmpty ? .gray : .blue)
            }
            .disabled(text.isEmpty && !isProcessing)
        }
        .padding()
        .background(.ultraThinMaterial)
    }
}

struct ThinkingIndicator: View {
    @State private var dots = 0
    
    var body: some View {
        HStack {
            Text("Thinking" + String(repeating: ".", count: dots))
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
                dots = (dots + 1) % 4
            }
        }
    }
}

// MARK: - Data Models

struct Message: Identifiable {
    let id = UUID()
    let role: Role
    let content: String
    var agentInfo: String? = nil
    let timestamp = Date()
    
    enum Role {
        case user
        case assistant
        case system
    }
}

// MARK: - Placeholder Views

struct SettingsView: View {
    var body: some View {
        NavigationStack {
            List {
                Section("Model") {
                    Text("Llama 3 8B Q4_K_M")
                    Text("Memory: ~4.5GB")
                }
                
                Section("Performance") {
                    Text("Metal GPU: Enabled")
                    Text("Threads: 6")
                }
            }
            .navigationTitle("Settings")
        }
    }
}

struct MCPConfigView: View {
    var body: some View {
        NavigationStack {
            List {
                Section("Connected Servers") {
                    Text("No servers connected")
                        .foregroundStyle(.secondary)
                }
                
                Section {
                    Button("Add MCP Server") {}
                }
            }
            .navigationTitle("MCP Connections")
        }
    }
}
