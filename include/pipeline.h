#ifndef VROOM_PIPELINE_H
#define VROOM_PIPELINE_H

#include "vroom.h"
#include "cuda_kernels.h"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

// ============================================================================
// ASYNC PIPELINE FOR BATCHED PROCESSING
// ============================================================================

// A batch of pixel data ready for GPU processing
struct PixelBatch {
    int batch_id;
    int start_pixel;        // Global pixel index
    int npixels;            // Number of pixels in this batch
    
    float* h_Q;             // Pinned host memory [npixels * nfreq]
    float* h_U;             // Pinned host memory [npixels * nfreq]
    float* h_I;             // Pinned host memory (nullptr if no model I)
    
    // Output buffers (pinned)
    float* h_fdf_real;      // [npixels * nphi] (optional)
    float* h_fdf_imag;      // [npixels * nphi] (optional)
    float* h_peak_rm;       // [npixels]
    float* h_peak_pi;       // [npixels]
    float* h_rm_err;        // [npixels]
    
    // Clean outputs (optional)
    float* h_clean_peak_rm; // [npixels]
    float* h_clean_peak_pi; // [npixels]
    
    bool ready_for_gpu;     // Data loaded and ready
    bool gpu_complete;      // GPU processing done
    bool written;           // Output written to disk
};

// Pipeline statistics
struct PipelineStats {
    std::atomic<int> batches_loaded{0};
    std::atomic<int> batches_processed{0};
    std::atomic<int> batches_written{0};
    std::atomic<size_t> pixels_processed{0};
    
    double load_time_ms{0};
    double gpu_time_ms{0};
    double write_time_ms{0};
    double total_time_ms{0};
};

// ============================================================================
// PIPELINE CLASS
// ============================================================================

class RMSynthPipeline {
public:
    RMSynthPipeline(const PipelineConfig& config);
    ~RMSynthPipeline();
    
    // Initialize with input files
    bool init(
        const char* q_file,
        const char* u_file,
        const char* freq_file,
        const char* i_file = nullptr,      // Optional model I
        const char* weights_file = nullptr  // Optional weights
    );
    
    // Run the full pipeline
    bool run();
    
    // Get stats
    const PipelineStats& get_stats() const { return stats_; }
    
    // For separate RM-CLEAN: load existing FDF and run clean
    bool run_rmclean_only(
        const char* fdf_real_file,
        const char* fdf_imag_file,
        const char* rmsf_file
    );
    
private:
    // Configuration
    PipelineConfig config_;
    CubeInfo cube_info_;
    
    // Precomputed values (host) — only valid (non-NaN) channels
    float* h_freq_;
    float* h_lambda_sq_;
    float* h_weights_;
    float* h_phi_;
    float mean_lambda_sq_;

    // Channel flagging: maps valid channel indices to original cube channels
    int* h_chan_map_;        // [nfreq_valid] -> original channel index
    int nfreq_orig_;        // Original number of channels in cube
    int nfreq_valid_;       // Number of valid (non-NaN) channels
    
    // RMSF
    RMSF rmsf_;
    float rmsf_fwhm_;
    
    // GPU resources
    GPUBuffers* gpu_buffers_[2];    // Double buffering
    cudaStream_t streams_[2];
    
    // Pipeline state
    int total_batches_;
    int current_load_batch_;
    int current_gpu_batch_;
    int current_write_batch_;
    
    // Batch queues
    std::queue<PixelBatch*> load_queue_;
    std::queue<PixelBatch*> gpu_queue_;
    std::queue<PixelBatch*> write_queue_;
    std::queue<PixelBatch*> free_queue_;
    
    // Synchronization
    std::mutex load_mutex_;
    std::mutex gpu_mutex_;
    std::mutex write_mutex_;
    std::mutex free_mutex_;
    std::condition_variable load_cv_;
    std::condition_variable gpu_cv_;
    std::condition_variable write_cv_;
    std::condition_variable free_cv_;
    std::atomic<bool> done_loading_{false};
    std::atomic<bool> done_processing_{false};
    std::atomic<bool> abort_{false};
    
    // File handles (kept open for batched reading)
    const char* q_filename_;
    const char* u_filename_;
    const char* i_filename_;
    
    // Output buffers (full images, accumulated)
    float* out_peak_rm_;
    float* out_peak_pi_;
    float* out_rm_err_;
    float* out_clean_peak_rm_;
    float* out_clean_peak_pi_;
    float* out_fdf_real_;    // Only if save_fdf_cube
    float* out_fdf_imag_;
    
    // Stats
    PipelineStats stats_;
    
    // Worker methods
    void loader_thread();
    void gpu_thread();
    void writer_thread();
    
    // Batch management
    PixelBatch* allocate_batch(int npixels);
    void free_batch(PixelBatch* batch);
    
    // Actual processing
    bool load_batch(PixelBatch* batch);
    bool process_batch_gpu(PixelBatch* batch, int buffer_idx);
    bool write_batch(PixelBatch* batch);
    
    // Initialization helpers
    bool compute_lambda_sq();
    bool compute_rmsf();
    bool allocate_output_buffers();
    bool write_outputs();
};

#endif // VROOM_PIPELINE_H
