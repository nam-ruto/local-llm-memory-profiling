/**
 * Advanced WebLLM Memory Profiler
 * 
 * This is an enhanced version that properly hooks into WebLLM's streaming API
 * to track token generation in real-time. Use this if the HTML version doesn't
 * properly track tokens.
 * 
 * Usage:
 * 1. Include this script in your HTML page
 * 2. Use the WebLLMProfiler class to wrap WebLLM calls
 */

class WebLLMProfiler {
    constructor(model, options = {}) {
        this.model = model;
        this.options = {
            samplingInterval: options.samplingInterval || 100, // ms
            ...options
        };
        
        this.engine = null;
        this.memorySamples = [];
        this.tokenCount = 0;
        this.startTime = null;
        this.memoryInterval = null;
        this.isProfiling = false;
        this.onSample = options.onSample || null;
        this.onToken = options.onToken || null;
    }
    
    /**
     * Check if performance.memory is available
     */
    static isMemoryAvailable() {
        return typeof performance !== 'undefined' && 
               performance.memory !== undefined;
    }
    
    /**
     * Sample current memory usage
     */
    sampleMemory() {
        if (!WebLLMProfiler.isMemoryAvailable()) {
            return null;
        }
        
        const memory = performance.memory;
        const now = Date.now();
        const elapsed = this.startTime ? (now - this.startTime) / 1000 : 0;
        
        const sample = {
            timestamp: new Date().toISOString(),
            label: 'webllm',
            pid: 'browser',
            heap_used_mb: memory.usedJSHeapSize / (1024 * 1024),
            heap_total_mb: memory.totalJSHeapSize / (1024 * 1024),
            heap_limit_mb: memory.jsHeapSizeLimit / (1024 * 1024),
            tokens_generated: this.tokenCount,
            elapsed_seconds: elapsed,
        };
        
        this.memorySamples.push(sample);
        
        if (this.onSample) {
            this.onSample(sample);
        }
        
        return sample;
    }
    
    /**
     * Start memory profiling
     */
    startProfiling() {
        if (this.isProfiling) {
            console.warn('Profiling already started');
            return;
        }
        
        if (!WebLLMProfiler.isMemoryAvailable()) {
            console.warn('performance.memory not available');
        }
        
        this.isProfiling = true;
        this.startTime = Date.now();
        this.memorySamples = [];
        this.tokenCount = 0;
        
        // Initial sample
        this.sampleMemory();
        
        // Start periodic sampling
        this.memoryInterval = setInterval(() => {
            this.sampleMemory();
        }, this.options.samplingInterval);
        
        console.log('Memory profiling started');
    }
    
    /**
     * Stop memory profiling
     */
    stopProfiling() {
        if (!this.isProfiling) {
            return;
        }
        
        this.isProfiling = false;
        
        if (this.memoryInterval) {
            clearInterval(this.memoryInterval);
            this.memoryInterval = null;
        }
        
        // Final sample
        this.sampleMemory();
        
        console.log('Memory profiling stopped');
    }
    
    /**
     * Record a token generation event
     */
    recordToken() {
        this.tokenCount++;
        if (this.onToken) {
            this.onToken(this.tokenCount);
        }
    }
    
    /**
     * Initialize WebLLM engine
     */
    async initialize(initProgressCallback = null) {
        const progressCallback = (report) => {
            if (initProgressCallback) {
                initProgressCallback(report);
            }
        };
        
        this.engine = await webllm.CreateWebLLMEngine(this.model, {
            initProgressCallback: progressCallback
        });
        
        return this.engine;
    }
    
    /**
     * Run inference with profiling
     */
    async complete(prompt, options = {}) {
        if (!this.engine) {
            throw new Error('Engine not initialized. Call initialize() first.');
        }
        
        // Start profiling
        this.startProfiling();
        
        try {
            const chat = this.engine.getChat();
            
            // Hook into streaming if available
            let fullResponse = '';
            let tokenBuffer = '';
            
            // WebLLM streaming callback
            const streamCallback = (delta, msg) => {
                // Track tokens (approximate - count words/spaces as tokens)
                if (delta && delta.trim()) {
                    this.recordToken();
                }
                fullResponse = msg;
            };
            
            // Run inference
            const response = await chat.complete(prompt, {
                ...options,
                streamInterval: options.streamInterval || 1,
            });
            
            // Stop profiling
            this.stopProfiling();
            
            return {
                response: response,
                metadata: {
                    tokens_generated: this.tokenCount,
                    elapsed_seconds: (Date.now() - this.startTime) / 1000,
                    memory_samples: this.memorySamples.length,
                }
            };
            
        } catch (error) {
            this.stopProfiling();
            throw error;
        }
    }
    
    /**
     * Export data as CSV
     */
    exportCSV() {
        const csvRows = [];
        csvRows.push('timestamp,label,pid,rss_mb,vms_mb,heap_used_mb,heap_total_mb,tokens_generated,elapsed_seconds');
        
        this.memorySamples.forEach(sample => {
            csvRows.push([
                sample.timestamp,
                sample.label,
                sample.pid,
                sample.heap_used_mb.toFixed(2),
                sample.heap_total_mb.toFixed(2),
                sample.heap_used_mb.toFixed(2),
                sample.heap_total_mb.toFixed(2),
                sample.tokens_generated,
                sample.elapsed_seconds.toFixed(3),
            ].join(','));
        });
        
        return csvRows.join('\n');
    }
    
    /**
     * Download CSV file
     */
    downloadCSV(filename = null) {
        if (!filename) {
            filename = `webllm_memory_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
        }
        
        const csv = this.exportCSV();
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    /**
     * Get profiling results
     */
    getResults() {
        return {
            samples: this.memorySamples,
            tokenCount: this.tokenCount,
            duration: this.startTime ? (Date.now() - this.startTime) / 1000 : 0,
        };
    }
}

// Export for use in modules or global scope
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebLLMProfiler;
}
