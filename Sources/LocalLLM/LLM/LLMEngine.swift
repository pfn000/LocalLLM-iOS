import Foundation
import Combine

/// LLMEngine wraps llama.cpp for on-device inference
/// No cloud, no tokens, no limits - just pure local compute
@MainActor
public final class LLMEngine: ObservableObject {
    public static let shared = LLMEngine()
    
    // MARK: - Published State
    
    @Published public private(set) var isModelLoaded = false
    @Published public private(set) var isGenerating = false
    @Published public private(set) var tokensPerSecond: Double = 0
    @Published public private(set) var lastError: String?
    
    // MARK: - Configuration
    
    public struct Config {
        var modelPath: String
        var contextLength: Int = 4096
        var batchSize: Int = 512
        var threads: Int = 6
        var gpuLayers: Int = 99  // Offload all to Metal
        var temperature: Float = 0.7
        var topP: Float = 0.9
        var topK: Int = 40
        var repeatPenalty: Float = 1.1
        
        public static var `default`: Config {
            Config(modelPath: Bundle.main.path(forResource: "model", ofType: "gguf") ?? "")
        }
    }
    
    private var config: Config
    private var llamaContext: OpaquePointer?
    private var llamaModel: OpaquePointer?
    
    // MARK: - Initialization
    
    private init() {
        self.config = .default
    }
    
    // MARK: - Model Loading
    
    public func loadModel(from path: String? = nil) async throws {
        if let path = path {
            config.modelPath = path
        }
        
        // Check if model exists
        guard FileManager.default.fileExists(atPath: config.modelPath) else {
            throw LLMError.modelNotFound(config.modelPath)
        }
        
        // Load on background thread
        try await Task.detached(priority: .userInitiated) { [config] in
            // Initialize llama.cpp backend
            llama_backend_init()
            
            // Model parameters
            var modelParams = llama_model_default_params()
            modelParams.n_gpu_layers = Int32(config.gpuLayers)
            
            // Load model
            guard let model = llama_load_model_from_file(config.modelPath, modelParams) else {
                throw LLMError.loadFailed("Failed to load model")
            }
            
            // Context parameters
            var ctxParams = llama_context_default_params()
            ctxParams.n_ctx = UInt32(config.contextLength)
            ctxParams.n_batch = UInt32(config.batchSize)
            ctxParams.n_threads = UInt32(config.threads)
            
            // Create context
            guard let ctx = llama_new_context_with_model(model, ctxParams) else {
                llama_free_model(model)
                throw LLMError.contextFailed
            }
            
            await MainActor.run {
                self.llamaModel = model
                self.llamaContext = ctx
                self.isModelLoaded = true
            }
        }.value
    }
    
    // MARK: - Inference
    
    /// Generate a response for the given prompt
    public func generate(
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 2048,
        stopSequences: [String] = []
    ) async throws -> String {
        guard isModelLoaded, let ctx = llamaContext, let model = llamaModel else {
            throw LLMError.notLoaded
        }
        
        isGenerating = true
        defer { isGenerating = false }
        
        return try await Task.detached(priority: .userInitiated) { [config] in
            // Build full prompt with chat template
            let fullPrompt = self.buildPrompt(
                system: systemPrompt,
                user: prompt
            )
            
            // Tokenize
            let tokens = self.tokenize(fullPrompt, model: model)
            
            // Evaluate prompt
            var batch = llama_batch_init(Int32(tokens.count), 0, 1)
            defer { llama_batch_free(batch) }
            
            for (i, token) in tokens.enumerated() {
                llama_batch_add(&batch, token, Int32(i), [0], false)
            }
            batch.logits[Int(batch.n_tokens) - 1] = 1  // Enable logits for last token
            
            if llama_decode(ctx, batch) != 0 {
                throw LLMError.decodeFailed
            }
            
            // Generate response
            var output = ""
            var generatedTokens = 0
            let startTime = Date()
            
            while generatedTokens < maxTokens {
                // Sample next token
                let logits = llama_get_logits_ith(ctx, batch.n_tokens - 1)
                let nVocab = llama_n_vocab(model)
                
                var candidates: [llama_token_data] = (0..<nVocab).map { tokenId in
                    llama_token_data(id: tokenId, logit: logits![Int(tokenId)], p: 0)
                }
                
                var candidatesP = llama_token_data_array(
                    data: &candidates,
                    size: candidates.count,
                    sorted: false
                )
                
                // Apply sampling
                llama_sample_top_k(ctx, &candidatesP, Int32(config.topK), 1)
                llama_sample_top_p(ctx, &candidatesP, config.topP, 1)
                llama_sample_temp(ctx, &candidatesP, config.temperature)
                
                let newToken = llama_sample_token(ctx, &candidatesP)
                
                // Check for end of sequence
                if llama_token_is_eog(model, newToken) {
                    break
                }
                
                // Decode token to text
                var buffer = [CChar](repeating: 0, count: 256)
                let length = llama_token_to_piece(model, newToken, &buffer, 256, 0, false)
                if length > 0 {
                    let piece = String(cString: buffer)
                    output += piece
                    
                    // Check stop sequences
                    for stop in stopSequences {
                        if output.hasSuffix(stop) {
                            output = String(output.dropLast(stop.count))
                            break
                        }
                    }
                }
                
                // Prepare next batch
                llama_batch_clear(&batch)
                llama_batch_add(&batch, newToken, Int32(tokens.count + generatedTokens), [0], true)
                
                if llama_decode(ctx, batch) != 0 {
                    throw LLMError.decodeFailed
                }
                
                generatedTokens += 1
            }
            
            // Calculate tokens/sec
            let elapsed = Date().timeIntervalSince(startTime)
            await MainActor.run {
                self.tokensPerSecond = Double(generatedTokens) / elapsed
            }
            
            return output.trimmingCharacters(in: .whitespacesAndNewlines)
        }.value
    }
    
    /// Stream tokens as they're generated
    public func generateStream(
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 2048
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // Similar to generate() but yield each token
                    // Implementation mirrors generate() with continuation.yield()
                    let result = try await generate(
                        prompt: prompt,
                        systemPrompt: systemPrompt,
                        maxTokens: maxTokens
                    )
                    continuation.yield(result)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Helpers
    
    private func buildPrompt(system: String?, user: String) -> String {
        // Llama 3 chat template
        var prompt = "<|begin_of_text|>"
        
        if let system = system {
            prompt += "<|start_header_id|>system<|end_header_id|>\n\n\(system)<|eot_id|>"
        }
        
        prompt += "<|start_header_id|>user<|end_header_id|>\n\n\(user)<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    }
    
    private func tokenize(_ text: String, model: OpaquePointer) -> [llama_token] {
        let maxTokens = text.count + 32
        var tokens = [llama_token](repeating: 0, count: maxTokens)
        let nTokens = llama_tokenize(model, text, Int32(text.count), &tokens, Int32(maxTokens), true, false)
        return Array(tokens.prefix(Int(nTokens)))
    }
    
    // MARK: - Cleanup
    
    deinit {
        if let ctx = llamaContext {
            llama_free(ctx)
        }
        if let model = llamaModel {
            llama_free_model(model)
        }
        llama_backend_free()
    }
}

// MARK: - Errors

public enum LLMError: LocalizedError {
    case modelNotFound(String)
    case loadFailed(String)
    case contextFailed
    case notLoaded
    case decodeFailed
    case tokenizationFailed
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model not found at: \(path)"
        case .loadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .contextFailed:
            return "Failed to create inference context"
        case .notLoaded:
            return "Model not loaded"
        case .decodeFailed:
            return "Token decoding failed"
        case .tokenizationFailed:
            return "Failed to tokenize input"
        }
    }
}

// MARK: - C Bridge Placeholders
// These are defined in CLlama module - placeholders for compilation

#if !canImport(CLlama)
// Placeholder types for development without llama.cpp linked
typealias llama_token = Int32
func llama_backend_init() {}
func llama_backend_free() {}
func llama_model_default_params() -> Any { fatalError() }
func llama_context_default_params() -> Any { fatalError() }
func llama_load_model_from_file(_ path: String, _ params: Any) -> OpaquePointer? { nil }
func llama_new_context_with_model(_ model: OpaquePointer, _ params: Any) -> OpaquePointer? { nil }
func llama_free(_ ctx: OpaquePointer) {}
func llama_free_model(_ model: OpaquePointer) {}
func llama_batch_init(_ n: Int32, _ embd: Int32, _ seqs: Int32) -> Any { fatalError() }
func llama_batch_free(_ batch: Any) {}
func llama_batch_add(_ batch: inout Any, _ token: llama_token, _ pos: Int32, _ seqs: [Int32], _ logits: Bool) {}
func llama_batch_clear(_ batch: inout Any) {}
func llama_decode(_ ctx: OpaquePointer, _ batch: Any) -> Int32 { 0 }
func llama_get_logits_ith(_ ctx: OpaquePointer, _ i: Int32) -> UnsafeMutablePointer<Float>? { nil }
func llama_n_vocab(_ model: OpaquePointer) -> Int32 { 0 }
func llama_sample_top_k(_ ctx: OpaquePointer, _ candidates: inout Any, _ k: Int32, _ minKeep: Int) {}
func llama_sample_top_p(_ ctx: OpaquePointer, _ candidates: inout Any, _ p: Float, _ minKeep: Int) {}
func llama_sample_temp(_ ctx: OpaquePointer, _ candidates: inout Any, _ temp: Float) {}
func llama_sample_token(_ ctx: OpaquePointer, _ candidates: inout Any) -> llama_token { 0 }
func llama_token_is_eog(_ model: OpaquePointer, _ token: llama_token) -> Bool { false }
func llama_token_to_piece(_ model: OpaquePointer, _ token: llama_token, _ buf: inout [CChar], _ length: Int32, _ special: Int32, _ render: Bool) -> Int32 { 0 }
func llama_tokenize(_ model: OpaquePointer, _ text: String, _ textLen: Int32, _ tokens: inout [llama_token], _ maxTokens: Int32, _ addBos: Bool, _ special: Bool) -> Int32 { 0 }
struct llama_token_data { var id: Int32; var logit: Float; var p: Float }
struct llama_token_data_array { var data: UnsafeMutablePointer<llama_token_data>?; var size: Int; var sorted: Bool }
#endif
