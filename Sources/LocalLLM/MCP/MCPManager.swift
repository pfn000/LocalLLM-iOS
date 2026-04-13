import Foundation
import Combine

/// MCPManager handles connections to Model Context Protocol servers
/// Enables your local LLM to use tools from Gmail, Calendar, Notion, etc.
@MainActor
public final class MCPManager: ObservableObject {
    public static let shared = MCPManager()
    
    // MARK: - Published State
    
    @Published public private(set) var servers: [MCPServer] = []
    @Published public private(set) var activeConnections: Int = 0
    @Published public private(set) var availableTools: [MCPTool] = []
    
    // MARK: - Types
    
    public struct MCPServer: Identifiable, Codable {
        public let id: UUID
        public var name: String
        public var url: URL
        public var type: ServerType
        public var isConnected: Bool
        public var tools: [MCPTool]
        
        public enum ServerType: String, Codable {
            case sse     // Server-Sent Events (most common)
            case stdio   // Standard I/O (local processes)
            case http    // HTTP REST
        }
        
        public init(name: String, url: URL, type: ServerType = .sse) {
            self.id = UUID()
            self.name = name
            self.url = url
            self.type = type
            self.isConnected = false
            self.tools = []
        }
    }
    
    public struct MCPTool: Identifiable, Codable {
        public let id: String
        public let name: String
        public let description: String
        public let inputSchema: [String: Any]
        public let serverID: UUID
        
        enum CodingKeys: String, CodingKey {
            case id, name, description, serverID
        }
        
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decode(String.self, forKey: .id)
            name = try container.decode(String.self, forKey: .name)
            description = try container.decode(String.self, forKey: .description)
            serverID = try container.decode(UUID.self, forKey: .serverID)
            inputSchema = [:]
        }
        
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(id, forKey: .id)
            try container.encode(name, forKey: .name)
            try container.encode(description, forKey: .description)
            try container.encode(serverID, forKey: .serverID)
        }
        
        init(id: String, name: String, description: String, inputSchema: [String: Any], serverID: UUID) {
            self.id = id
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
            self.serverID = serverID
        }
    }
    
    public struct ToolCall: Codable {
        public let toolName: String
        public let arguments: [String: Any]
        
        enum CodingKeys: String, CodingKey {
            case toolName
        }
        
        public init(toolName: String, arguments: [String: Any]) {
            self.toolName = toolName
            self.arguments = arguments
        }
        
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            toolName = try container.decode(String.self, forKey: .toolName)
            arguments = [:]
        }
        
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(toolName, forKey: .toolName)
        }
    }
    
    public struct ToolResult {
        public let success: Bool
        public let content: Any
        public let error: String?
    }
    
    // MARK: - Private State
    
    private var eventSources: [UUID: URLSessionDataTask] = [:]
    private var pendingCalls: [String: CheckedContinuation<ToolResult, Error>] = [:]
    
    // MARK: - Initialization
    
    private init() {
        loadSavedServers()
    }
    
    // MARK: - Server Management
    
    public func addServer(_ server: MCPServer) {
        servers.append(server)
        saveServers()
    }
    
    public func removeServer(_ id: UUID) {
        disconnect(from: id)
        servers.removeAll { $0.id == id }
        saveServers()
    }
    
    // MARK: - Connection
    
    public func connect(to serverID: UUID) async throws {
        guard let index = servers.firstIndex(where: { $0.id == serverID }) else {
            throw MCPError.serverNotFound
        }
        
        let server = servers[index]
        
        switch server.type {
        case .sse:
            try await connectSSE(server: server, index: index)
        case .http:
            try await connectHTTP(server: server, index: index)
        case .stdio:
            throw MCPError.stdioNotSupported
        }
    }
    
    public func disconnect(from serverID: UUID) {
        eventSources[serverID]?.cancel()
        eventSources.removeValue(forKey: serverID)
        
        if let index = servers.firstIndex(where: { $0.id == serverID }) {
            servers[index].isConnected = false
            servers[index].tools = []
        }
        
        updateState()
    }
    
    // MARK: - Tool Execution
    
    public func callTool(_ call: ToolCall) async throws -> ToolResult {
        // Find which server has this tool
        guard let tool = availableTools.first(where: { $0.name == call.toolName }),
              let server = servers.first(where: { $0.id == tool.serverID && $0.isConnected }) else {
            throw MCPError.toolNotFound(call.toolName)
        }
        
        // Build MCP tool call request
        let request = MCPRequest(
            jsonrpc: "2.0",
            id: UUID().uuidString,
            method: "tools/call",
            params: [
                "name": call.toolName,
                "arguments": call.arguments
            ]
        )
        
        // Send request based on server type
        switch server.type {
        case .sse:
            return try await sendSSERequest(request, to: server)
        case .http:
            return try await sendHTTPRequest(request, to: server)
        case .stdio:
            throw MCPError.stdioNotSupported
        }
    }
    
    /// Format tools for LLM consumption
    public func toolsPrompt() -> String {
        guard !availableTools.isEmpty else { return "" }
        
        var prompt = "You have access to the following tools:\n\n"
        
        for tool in availableTools {
            prompt += """
            <tool>
            name: \(tool.name)
            description: \(tool.description)
            </tool>
            
            """
        }
        
        prompt += """
        
        To use a tool, output:
        <tool_use>
        name: tool_name
        arguments:
          key: value
        </tool_use>
        """
        
        return prompt
    }
    
    // MARK: - SSE Implementation
    
    private func connectSSE(server: MCPServer, index: Int) async throws {
        var request = URLRequest(url: server.url)
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        
        let (stream, response) = try await URLSession.shared.bytes(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw MCPError.connectionFailed
        }
        
        // Mark as connected
        servers[index].isConnected = true
        updateState()
        
        // Fetch available tools
        let tools = try await fetchTools(from: server)
        servers[index].tools = tools
        updateState()
        
        // Listen for events (in background)
        Task {
            for try await line in stream.lines {
                handleSSEEvent(line, serverID: server.id)
            }
            
            // Connection closed
            await MainActor.run {
                self.servers[index].isConnected = false
                self.updateState()
            }
        }
    }
    
    private func handleSSEEvent(_ line: String, serverID: UUID) {
        guard line.hasPrefix("data: ") else { return }
        
        let data = String(line.dropFirst(6))
        
        guard let jsonData = data.data(using: .utf8),
              let response = try? JSONDecoder().decode(MCPResponse.self, from: jsonData) else {
            return
        }
        
        // Handle response
        if let id = response.id, let continuation = pendingCalls[id] {
            pendingCalls.removeValue(forKey: id)
            
            if let error = response.error {
                continuation.resume(throwing: MCPError.toolError(error.message))
            } else {
                continuation.resume(returning: ToolResult(
                    success: true,
                    content: response.result ?? [:],
                    error: nil
                ))
            }
        }
    }
    
    private func sendSSERequest(_ request: MCPRequest, to server: MCPServer) async throws -> ToolResult {
        // For SSE, we typically POST to a separate endpoint
        let postURL = server.url.deletingLastPathComponent().appendingPathComponent("message")
        
        var urlRequest = URLRequest(url: postURL)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        
        return try await withCheckedThrowingContinuation { continuation in
            pendingCalls[request.id] = continuation
            
            Task {
                do {
                    let (_, response) = try await URLSession.shared.data(for: urlRequest)
                    
                    guard let httpResponse = response as? HTTPURLResponse,
                          httpResponse.statusCode == 200 else {
                        pendingCalls.removeValue(forKey: request.id)
                        continuation.resume(throwing: MCPError.requestFailed)
                        return
                    }
                    
                    // Response will come via SSE stream
                } catch {
                    pendingCalls.removeValue(forKey: request.id)
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    // MARK: - HTTP Implementation
    
    private func connectHTTP(server: MCPServer, index: Int) async throws {
        // Verify connection with initialize request
        let request = MCPRequest(
            jsonrpc: "2.0",
            id: UUID().uuidString,
            method: "initialize",
            params: [
                "protocolVersion": "2024-11-05",
                "capabilities": [:],
                "clientInfo": [
                    "name": "LocalLLM-iOS",
                    "version": "1.0.0"
                ]
            ]
        )
        
        _ = try await sendHTTPRequest(request, to: server)
        
        servers[index].isConnected = true
        
        // Fetch tools
        let tools = try await fetchTools(from: server)
        servers[index].tools = tools
        
        updateState()
    }
    
    private func sendHTTPRequest(_ request: MCPRequest, to server: MCPServer) async throws -> ToolResult {
        var urlRequest = URLRequest(url: server.url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        
        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw MCPError.requestFailed
        }
        
        let mcpResponse = try JSONDecoder().decode(MCPResponse.self, from: data)
        
        if let error = mcpResponse.error {
            throw MCPError.toolError(error.message)
        }
        
        return ToolResult(
            success: true,
            content: mcpResponse.result ?? [:],
            error: nil
        )
    }
    
    // MARK: - Tool Discovery
    
    private func fetchTools(from server: MCPServer) async throws -> [MCPTool] {
        let request = MCPRequest(
            jsonrpc: "2.0",
            id: UUID().uuidString,
            method: "tools/list",
            params: [:]
        )
        
        let result: ToolResult
        switch server.type {
        case .sse:
            result = try await sendSSERequest(request, to: server)
        case .http:
            result = try await sendHTTPRequest(request, to: server)
        case .stdio:
            throw MCPError.stdioNotSupported
        }
        
        guard let toolsArray = (result.content as? [String: Any])?["tools"] as? [[String: Any]] else {
            return []
        }
        
        return toolsArray.compactMap { dict -> MCPTool? in
            guard let name = dict["name"] as? String,
                  let description = dict["description"] as? String else {
                return nil
            }
            
            return MCPTool(
                id: name,
                name: name,
                description: description,
                inputSchema: dict["inputSchema"] as? [String: Any] ?? [:],
                serverID: server.id
            )
        }
    }
    
    // MARK: - State Management
    
    private func updateState() {
        activeConnections = servers.filter(\.isConnected).count
        availableTools = servers.flatMap(\.tools)
    }
    
    // MARK: - Persistence
    
    private func saveServers() {
        if let data = try? JSONEncoder().encode(servers) {
            UserDefaults.standard.set(data, forKey: "mcp_servers")
        }
    }
    
    private func loadSavedServers() {
        guard let data = UserDefaults.standard.data(forKey: "mcp_servers"),
              let saved = try? JSONDecoder().decode([MCPServer].self, from: data) else {
            return
        }
        servers = saved
    }
}

// MARK: - MCP Protocol Types

struct MCPRequest: Codable {
    let jsonrpc: String
    let id: String
    let method: String
    let params: [String: Any]
    
    enum CodingKeys: String, CodingKey {
        case jsonrpc, id, method, params
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(jsonrpc, forKey: .jsonrpc)
        try container.encode(id, forKey: .id)
        try container.encode(method, forKey: .method)
        // Params encoded separately due to Any type
        if let paramsData = try? JSONSerialization.data(withJSONObject: params),
           let paramsString = String(data: paramsData, encoding: .utf8) {
            try container.encode(paramsString, forKey: .params)
        }
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        jsonrpc = try container.decode(String.self, forKey: .jsonrpc)
        id = try container.decode(String.self, forKey: .id)
        method = try container.decode(String.self, forKey: .method)
        params = [:]
    }
    
    init(jsonrpc: String, id: String, method: String, params: [String: Any]) {
        self.jsonrpc = jsonrpc
        self.id = id
        self.method = method
        self.params = params
    }
}

struct MCPResponse: Codable {
    let jsonrpc: String
    let id: String?
    let result: [String: Any]?
    let error: MCPResponseError?
    
    enum CodingKeys: String, CodingKey {
        case jsonrpc, id, error
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        jsonrpc = try container.decode(String.self, forKey: .jsonrpc)
        id = try container.decodeIfPresent(String.self, forKey: .id)
        error = try container.decodeIfPresent(MCPResponseError.self, forKey: .error)
        result = nil
    }
}

struct MCPResponseError: Codable {
    let code: Int
    let message: String
}

// MARK: - Errors

public enum MCPError: LocalizedError {
    case serverNotFound
    case connectionFailed
    case requestFailed
    case toolNotFound(String)
    case toolError(String)
    case stdioNotSupported
    
    public var errorDescription: String? {
        switch self {
        case .serverNotFound:
            return "MCP server not found"
        case .connectionFailed:
            return "Failed to connect to MCP server"
        case .requestFailed:
            return "MCP request failed"
        case .toolNotFound(let name):
            return "Tool not found: \(name)"
        case .toolError(let message):
            return "Tool error: \(message)"
        case .stdioNotSupported:
            return "STDIO transport not supported on iOS"
        }
    }
}
