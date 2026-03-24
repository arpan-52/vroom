#include "vroom.h"
#include "pipeline.h"
#include "cuda_kernels.h"
#include "io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <sys/sysinfo.h>

// ============================================================================
// MEMORY INFO
// ============================================================================

MemoryInfo get_memory_info() {
    MemoryInfo info = {0};
    
    // RAM info
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        info.ram_total = si.totalram * si.mem_unit;
        info.ram_available = si.freeram * si.mem_unit;
    }
    
    // VRAM info
    size_t free_vram, total_vram;
    if (cudaMemGetInfo(&free_vram, &total_vram) == cudaSuccess) {
        info.vram_total = total_vram;
        info.vram_available = free_vram;
    }
    
    return info;
}

void print_memory_info(const MemoryInfo& info) {
    printf("Memory Info:\n");
    printf("  RAM:  %.2f GB available / %.2f GB total\n",
           info.ram_available / 1e9, info.ram_total / 1e9);
    printf("  VRAM: %.2f GB available / %.2f GB total\n",
           info.vram_available / 1e9, info.vram_total / 1e9);
}

// ============================================================================
// BATCH SIZE COMPUTATION
// ============================================================================

int compute_optimal_batch_size(
    const CubeInfo& cube,
    const FaradayConfig& faraday,
    const MemoryInfo& mem,
    bool do_rmclean
) {
    // Memory per pixel:
    // RAM (pinned, double-buffered so x2):
    //   - Q, U input: 2 * nfreq * 4 bytes
    //   - I input (optional): nfreq * 4 bytes  
    //   - FDF output: 2 * nphi * 4 bytes (real, imag)
    //   - Peak outputs: 5 * 4 bytes (rm, pi, err, clean_rm, clean_pi)
    //
    // VRAM:
    // Exact per-pixel VRAM cost (must match allocate_gpu_buffers)
    int nf = cube.nfreq;
    int np = faraday.nphi;
    int input_bufs = do_rmclean ? 2 : 2;  // Q, U (I only if normalize)
    // Not counting I here — added below if needed

    size_t vram_per_pixel_per_set =
          2 * nf * sizeof(float)            // d_Q, d_U
        + 2 * nf * sizeof(float)            // d_Q_norm, d_U_norm
        + 2 * np * sizeof(float)            // d_fdf_real, d_fdf_imag
        + 3 * sizeof(float);               // d_peak_rm, d_peak_pi, d_rm_err

    if (do_rmclean) {
        vram_per_pixel_per_set += 4 * np * sizeof(float); // residual + model (real/imag)
    }

    size_t vram_per_pixel = vram_per_pixel_per_set * 2;  // x2 for double buffering

    // Fixed overhead per buffer set (constants, RMSF, NUFFT workspace)
    int nufft_batch = (NUFFT_MAX_BATCH < 256) ? NUFFT_MAX_BATCH : NUFFT_MAX_BATCH;
    size_t fixed_per_set =
          nf * sizeof(float) * 4            // d_freq, d_lambda_sq, d_weights, d_nufft_x
        + np * sizeof(float) * 3            // d_phi, d_rmsf_real, d_rmsf_imag
        + (size_t)nufft_batch * nf * sizeof(cuFloatComplex)  // d_nufft_c
        + (size_t)nufft_batch * np * sizeof(cuFloatComplex); // d_nufft_f

    size_t vram_overhead = fixed_per_set * 2;  // x2 buffer sets

    // RAM per pixel (pinned host, x2 for double buffering)
    size_t ram_per_pixel = (2 * nf * sizeof(float)         // h_Q, h_U
                         + 3 * sizeof(float)) * 2;         // h_peak_rm/pi/err, x2 buffers

    // Use 80% of available, minus overhead
    size_t usable_vram = (size_t)(mem.vram_available * 0.8) - vram_overhead;
    size_t usable_ram = (size_t)(mem.ram_available * 0.8);

    int batch_from_vram = (int)(usable_vram / vram_per_pixel);
    int batch_from_ram = (int)(usable_ram / ram_per_pixel);
    
    int batch_size = (batch_from_ram < batch_from_vram) ? batch_from_ram : batch_from_vram;
    
    // Clamp to reasonable values
    if (batch_size < 1024) batch_size = 1024;
    if (batch_size > (int)cube.npix) batch_size = (int)cube.npix;
    
    // Round down to multiple of 256 for GPU efficiency
    batch_size = (batch_size / 256) * 256;
    if (batch_size < 256) batch_size = 256;
    
    printf("Batch size: %d pixels (RAM limit: %d, VRAM limit: %d)\n",
           batch_size, batch_from_ram, batch_from_vram);
    
    return batch_size;
}

// ============================================================================
// PIPELINE IMPLEMENTATION
// ============================================================================

RMSynthPipeline::RMSynthPipeline(const PipelineConfig& config) 
    : config_(config)
    , h_freq_(nullptr)
    , h_lambda_sq_(nullptr)
    , h_weights_(nullptr)
    , h_phi_(nullptr)
    , mean_lambda_sq_(0.0f)
    , h_chan_map_(nullptr)
    , nfreq_orig_(0)
    , nfreq_valid_(0)
    , rmsf_fwhm_(0.0f)
    , total_batches_(0)
    , current_load_batch_(0)
    , current_gpu_batch_(0)
    , current_write_batch_(0)
    , q_filename_(nullptr)
    , u_filename_(nullptr)
    , i_filename_(nullptr)
    , out_peak_rm_(nullptr)
    , out_peak_pi_(nullptr)
    , out_rm_err_(nullptr)
    , out_clean_peak_rm_(nullptr)
    , out_clean_peak_pi_(nullptr)
    , out_fdf_real_(nullptr)
    , out_fdf_imag_(nullptr)
{
    memset(&cube_info_, 0, sizeof(cube_info_));
    memset(&rmsf_, 0, sizeof(rmsf_));
    
    gpu_buffers_[0] = nullptr;
    gpu_buffers_[1] = nullptr;
    streams_[0] = 0;
    streams_[1] = 0;
}

RMSynthPipeline::~RMSynthPipeline() {
    // Free host memory
    if (h_freq_) free(h_freq_);
    if (h_lambda_sq_) free(h_lambda_sq_);
    if (h_weights_) free(h_weights_);
    if (h_phi_) free(h_phi_);
    if (h_chan_map_) free(h_chan_map_);
    
    if (rmsf_.rmsf_real) free(rmsf_.rmsf_real);
    if (rmsf_.rmsf_imag) free(rmsf_.rmsf_imag);
    if (rmsf_.rmsf_amp) free(rmsf_.rmsf_amp);
    
    if (out_peak_rm_) free(out_peak_rm_);
    if (out_peak_pi_) free(out_peak_pi_);
    if (out_rm_err_) free(out_rm_err_);
    if (out_clean_peak_rm_) free(out_clean_peak_rm_);
    if (out_clean_peak_pi_) free(out_clean_peak_pi_);
    if (out_fdf_real_) free(out_fdf_real_);
    if (out_fdf_imag_) free(out_fdf_imag_);
    
    // Free GPU resources
    if (gpu_buffers_[0]) free_gpu_buffers(gpu_buffers_[0]);
    if (gpu_buffers_[1]) free_gpu_buffers(gpu_buffers_[1]);
    
    if (streams_[0]) cudaStreamDestroy(streams_[0]);
    if (streams_[1]) cudaStreamDestroy(streams_[1]);
}

bool RMSynthPipeline::init(
    const char* q_file,
    const char* u_file, 
    const char* freq_file,
    const char* i_file,
    const char* weights_file
) {
    printf("=== VROOM: GPU RM-Synthesis + RM-CLEAN ===\n\n");
    
    // Store filenames for batch reading
    q_filename_ = q_file;
    u_filename_ = u_file;
    i_filename_ = i_file;
    
    // Get memory info
    MemoryInfo mem = get_memory_info();
    print_memory_info(mem);
    printf("\n");
    
    // Read frequencies
    int nfreq;
    if (read_frequencies(freq_file, &h_freq_, &nfreq) != 0) {
        fprintf(stderr, "Failed to read frequencies\n");
        return false;
    }
    printf("Frequencies: %d channels (%.1f - %.1f MHz)\n",
           nfreq, h_freq_[0] / 1e6, h_freq_[nfreq - 1] / 1e6);
    
    // Get cube dimensions from Q file
    float* dummy_data = nullptr;
    if (read_fits_cube(q_file, &dummy_data, &cube_info_) != 0) {
        fprintf(stderr, "Failed to read Q cube header\n");
        return false;
    }
    free(dummy_data);  // We just needed the dimensions
    
    if (cube_info_.nfreq != nfreq) {
        fprintf(stderr, "Mismatch: cube has %d freq channels, file has %d\n",
                cube_info_.nfreq, nfreq);
        return false;
    }
    
    nfreq_orig_ = cube_info_.nfreq;
    printf("Cube dimensions: %d x %d x %d (%.2f GB per cube)\n",
           cube_info_.nx, cube_info_.ny, cube_info_.nfreq,
           cube_info_.nelems * sizeof(float) / 1e9);

    // Check if model I is provided
    config_.normalize_by_i = (i_file != nullptr);
    printf("Normalization: %s\n", config_.normalize_by_i ? "Q/I, U/I" : "None (raw Q, U)");

    // Read or compute weights (for all original channels)
    float* all_weights = nullptr;
    if (weights_file && read_weights(weights_file, &all_weights, nfreq) == 0) {
        printf("Weights: loaded from file\n");
    } else {
        all_weights = (float*)malloc(nfreq * sizeof(float));
        for (int i = 0; i < nfreq; i++) all_weights[i] = 1.0f;
        printf("Weights: uniform\n");
    }

    // --- Detect bad (NaN) channels by reading a sample pixel from Q cube ---
    {
        // Read center pixel spectrum
        int center_pix = (cube_info_.ny / 2) * cube_info_.nx + (cube_info_.nx / 2);
        float* sample_q = (float*)malloc(nfreq * sizeof(float));
        float* sample_u = (float*)malloc(nfreq * sizeof(float));
        read_fits_cube_batch(q_file, sample_q, cube_info_, center_pix, 1);
        read_fits_cube_batch(u_file, sample_u, cube_info_, center_pix, 1);

        // Build valid channel map
        h_chan_map_ = (int*)malloc(nfreq * sizeof(int));
        nfreq_valid_ = 0;
        for (int i = 0; i < nfreq; i++) {
            if (std::isfinite(sample_q[i]) && std::isfinite(sample_u[i])
                && all_weights[i] > 0.0f) {
                h_chan_map_[nfreq_valid_++] = i;
            }
        }
        free(sample_q);
        free(sample_u);

        printf("Valid channels: %d / %d (flagged %d bad channels)\n",
               nfreq_valid_, nfreq, nfreq - nfreq_valid_);

        if (nfreq_valid_ < 2) {
            fprintf(stderr, "Too few valid channels!\n");
            free(all_weights);
            return false;
        }
    }

    // --- Compact frequencies and weights to valid-only arrays ---
    float* all_freq = h_freq_;  // Save original full array
    h_freq_ = (float*)malloc(nfreq_valid_ * sizeof(float));
    h_weights_ = (float*)malloc(nfreq_valid_ * sizeof(float));
    for (int i = 0; i < nfreq_valid_; i++) {
        h_freq_[i] = all_freq[h_chan_map_[i]];
        h_weights_[i] = all_weights[h_chan_map_[i]];
    }
    free(all_freq);
    free(all_weights);

    // Update cube_info to reflect valid channel count for GPU processing
    cube_info_.nfreq = nfreq_valid_;

    printf("Frequency range (valid): %.1f - %.1f MHz\n",
           h_freq_[0] / 1e6, h_freq_[nfreq_valid_ - 1] / 1e6);

    // Compute lambda squared (using compacted valid channels)
    if (!compute_lambda_sq()) {
        fprintf(stderr, "Failed to compute lambda squared\n");
        return false;
    }
    
    // Setup Faraday grid
    config_.faraday.nphi = (int)((config_.faraday.phi_max - config_.faraday.phi_min) 
                                  / config_.faraday.dphi) + 1;
    h_phi_ = (float*)malloc(config_.faraday.nphi * sizeof(float));
    for (int i = 0; i < config_.faraday.nphi; i++) {
        h_phi_[i] = config_.faraday.phi_min + i * config_.faraday.dphi;
    }
    printf("Faraday depth: %.1f to %.1f rad/m² (dphi=%.1f, nphi=%d)\n",
           config_.faraday.phi_min, config_.faraday.phi_max,
           config_.faraday.dphi, config_.faraday.nphi);
    
    // Compute optimal batch size
    config_.batch.batch_pixels = compute_optimal_batch_size(
        cube_info_, config_.faraday, mem, config_.do_rmclean
    );
    
    total_batches_ = ((int)cube_info_.npix + config_.batch.batch_pixels - 1) 
                     / config_.batch.batch_pixels;
    printf("Total batches: %d\n", total_batches_);
    
    // Create CUDA streams
    cudaStreamCreate(&streams_[0]);
    cudaStreamCreate(&streams_[1]);

    // Allocate GPU buffers (double buffered)
    gpu_buffers_[0] = allocate_gpu_buffers(
        cube_info_.nfreq, config_.faraday.nphi, config_.batch.batch_pixels,
        config_.normalize_by_i, config_.do_rmclean
    );
    gpu_buffers_[1] = allocate_gpu_buffers(
        cube_info_.nfreq, config_.faraday.nphi, config_.batch.batch_pixels,
        config_.normalize_by_i, config_.do_rmclean
    );

    if (!gpu_buffers_[0] || !gpu_buffers_[1]) {
        fprintf(stderr, "Failed to allocate GPU buffers\n");
        return false;
    }
    
    // Upload constants to both buffer sets (using valid channel count)
    int nfreq_gpu = cube_info_.nfreq;  // = nfreq_valid_ after compaction
    for (int i = 0; i < 2; i++) {
        cudaMemcpy(gpu_buffers_[i]->d_freq, h_freq_,
                   nfreq_gpu * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_buffers_[i]->d_lambda_sq, h_lambda_sq_,
                   nfreq_gpu * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_buffers_[i]->d_weights, h_weights_,
                   nfreq_gpu * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_buffers_[i]->d_phi, h_phi_,
                   config_.faraday.nphi * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Initialize NUFFT plans for both buffer sets
    printf("Initializing NUFFT plans...\n");
    for (int i = 0; i < 2; i++) {
        int ier = cu_init_nufft_plan(
            gpu_buffers_[i],
            config_.faraday.phi_min,
            config_.faraday.dphi,
            mean_lambda_sq_,
            streams_[i]
        );
        if (ier != 0) {
            fprintf(stderr, "Failed to initialize NUFFT plan %d\n", i);
            return false;
        }
    }
    printf("NUFFT plans initialized successfully.\n");
    
    // Compute RMSF
    if (!compute_rmsf()) {
        fprintf(stderr, "Failed to compute RMSF\n");
        return false;
    }
    
    // Allocate output buffers
    if (!allocate_output_buffers()) {
        fprintf(stderr, "Failed to allocate output buffers\n");
        return false;
    }
    
    printf("\nInitialization complete.\n\n");
    return true;
}

bool RMSynthPipeline::compute_lambda_sq() {
    const float c = 299792458.0f;
    h_lambda_sq_ = (float*)malloc(cube_info_.nfreq * sizeof(float));
    
    double sum_wl2 = 0.0, sum_w = 0.0;
    for (int i = 0; i < cube_info_.nfreq; i++) {
        float lambda = c / h_freq_[i];
        h_lambda_sq_[i] = lambda * lambda;
        sum_wl2 += h_weights_[i] * h_lambda_sq_[i];
        sum_w += h_weights_[i];
    }
    mean_lambda_sq_ = (float)(sum_wl2 / sum_w);
    
    printf("Lambda squared: %.4f - %.4f m² (mean: %.4f m²)\n",
           h_lambda_sq_[cube_info_.nfreq - 1], h_lambda_sq_[0], mean_lambda_sq_);
    
    return true;
}

bool RMSynthPipeline::compute_rmsf() {
    int nphi = config_.faraday.nphi;
    
    // Allocate host RMSF
    rmsf_.rmsf_real = (float*)malloc(nphi * sizeof(float));
    rmsf_.rmsf_imag = (float*)malloc(nphi * sizeof(float));
    rmsf_.rmsf_amp = (float*)malloc(nphi * sizeof(float));
    
    // Compute on GPU using first buffer
    cu_compute_rmsf(
        gpu_buffers_[0]->d_lambda_sq,
        gpu_buffers_[0]->d_weights,
        gpu_buffers_[0]->d_phi,
        gpu_buffers_[0]->d_rmsf_real,
        gpu_buffers_[0]->d_rmsf_imag,
        mean_lambda_sq_,
        cube_info_.nfreq,
        nphi,
        streams_[0]
    );
    cudaStreamSynchronize(streams_[0]);
    
    // Copy back
    cudaMemcpy(rmsf_.rmsf_real, gpu_buffers_[0]->d_rmsf_real,
               nphi * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rmsf_.rmsf_imag, gpu_buffers_[0]->d_rmsf_imag,
               nphi * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute amplitude and FWHM
    float max_amp = 0.0f;
    int peak_idx = nphi / 2;
    for (int i = 0; i < nphi; i++) {
        rmsf_.rmsf_amp[i] = sqrtf(rmsf_.rmsf_real[i] * rmsf_.rmsf_real[i] +
                                   rmsf_.rmsf_imag[i] * rmsf_.rmsf_imag[i]);
        if (rmsf_.rmsf_amp[i] > max_amp) {
            max_amp = rmsf_.rmsf_amp[i];
            peak_idx = i;
        }
    }
    
    // Find FWHM
    float half_max = max_amp * 0.5f;
    int left_idx = peak_idx, right_idx = peak_idx;
    for (int i = peak_idx; i >= 0; i--) {
        if (rmsf_.rmsf_amp[i] < half_max) { left_idx = i; break; }
    }
    for (int i = peak_idx; i < nphi; i++) {
        if (rmsf_.rmsf_amp[i] < half_max) { right_idx = i; break; }
    }
    rmsf_.fwhm = h_phi_[right_idx] - h_phi_[left_idx];
    rmsf_fwhm_ = rmsf_.fwhm;
    
    // Max recoverable scale (approximate)
    float delta_lambda_sq = h_lambda_sq_[0] - h_lambda_sq_[cube_info_.nfreq - 1];
    rmsf_.max_scale = M_PI / fabsf(delta_lambda_sq);
    
    printf("RMSF: FWHM = %.2f rad/m², max scale = %.1f rad/m²\n",
           rmsf_.fwhm, rmsf_.max_scale);
    
    // Copy RMSF to second buffer too
    cudaMemcpy(gpu_buffers_[1]->d_rmsf_real, gpu_buffers_[0]->d_rmsf_real,
               nphi * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_buffers_[1]->d_rmsf_imag, gpu_buffers_[0]->d_rmsf_imag,
               nphi * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Save RMSF if requested
    if (config_.save_rmsf) {
        char filename[512];
        snprintf(filename, sizeof(filename), "%s_rmsf.fits", config_.output_prefix);
        save_rmsf(filename, rmsf_, config_.faraday);
        printf("Saved RMSF: %s\n", filename);
    }
    
    return true;
}

bool RMSynthPipeline::allocate_output_buffers() {
    size_t npix = cube_info_.npix;
    
    out_peak_rm_ = (float*)calloc(npix, sizeof(float));
    out_peak_pi_ = (float*)calloc(npix, sizeof(float));
    out_rm_err_ = (float*)calloc(npix, sizeof(float));
    
    if (config_.do_rmclean) {
        out_clean_peak_rm_ = (float*)calloc(npix, sizeof(float));
        out_clean_peak_pi_ = (float*)calloc(npix, sizeof(float));
    }
    
    if (config_.save_fdf_cube) {
        size_t fdf_size = npix * config_.faraday.nphi;
        out_fdf_real_ = (float*)calloc(fdf_size, sizeof(float));
        out_fdf_imag_ = (float*)calloc(fdf_size, sizeof(float));
        if (!out_fdf_real_ || !out_fdf_imag_) {
            fprintf(stderr, "Failed to allocate FDF cube (%.2f GB)\n",
                    fdf_size * 2 * sizeof(float) / 1e9);
            return false;
        }
    }
    
    // Initialize with NaN
    float nan_val = NAN;
    for (size_t i = 0; i < npix; i++) {
        out_peak_rm_[i] = nan_val;
        out_peak_pi_[i] = nan_val;
        out_rm_err_[i] = nan_val;
        if (config_.do_rmclean) {
            out_clean_peak_rm_[i] = nan_val;
            out_clean_peak_pi_[i] = nan_val;
        }
    }
    
    return true;
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

PixelBatch* RMSynthPipeline::allocate_batch(int npixels) {
    PixelBatch* batch = new PixelBatch();
    batch->npixels = npixels;
    
    size_t freq_size = npixels * cube_info_.nfreq * sizeof(float);
    size_t phi_size = npixels * config_.faraday.nphi * sizeof(float);
    
    // Pinned memory for async transfers
    cudaMallocHost(&batch->h_Q, freq_size);
    cudaMallocHost(&batch->h_U, freq_size);
    if (config_.normalize_by_i) {
        cudaMallocHost(&batch->h_I, freq_size);
    } else {
        batch->h_I = nullptr;
    }
    
    cudaMallocHost(&batch->h_peak_rm, npixels * sizeof(float));
    cudaMallocHost(&batch->h_peak_pi, npixels * sizeof(float));
    cudaMallocHost(&batch->h_rm_err, npixels * sizeof(float));
    
    if (config_.do_rmclean) {
        cudaMallocHost(&batch->h_clean_peak_rm, npixels * sizeof(float));
        cudaMallocHost(&batch->h_clean_peak_pi, npixels * sizeof(float));
    } else {
        batch->h_clean_peak_rm = nullptr;
        batch->h_clean_peak_pi = nullptr;
    }
    
    if (config_.save_fdf_cube) {
        cudaMallocHost(&batch->h_fdf_real, phi_size);
        cudaMallocHost(&batch->h_fdf_imag, phi_size);
    } else {
        batch->h_fdf_real = nullptr;
        batch->h_fdf_imag = nullptr;
    }
    
    return batch;
}

void RMSynthPipeline::free_batch(PixelBatch* batch) {
    if (!batch) return;
    
    cudaFreeHost(batch->h_Q);
    cudaFreeHost(batch->h_U);
    if (batch->h_I) cudaFreeHost(batch->h_I);
    cudaFreeHost(batch->h_peak_rm);
    cudaFreeHost(batch->h_peak_pi);
    cudaFreeHost(batch->h_rm_err);
    if (batch->h_clean_peak_rm) cudaFreeHost(batch->h_clean_peak_rm);
    if (batch->h_clean_peak_pi) cudaFreeHost(batch->h_clean_peak_pi);
    if (batch->h_fdf_real) cudaFreeHost(batch->h_fdf_real);
    if (batch->h_fdf_imag) cudaFreeHost(batch->h_fdf_imag);
    
    delete batch;
}

bool RMSynthPipeline::load_batch(PixelBatch* batch) {
    int start = batch->start_pixel;
    int count = batch->npixels;

    // Fast channel-major read: 78 slab reads per cube instead of 722K pixel reads
    if (read_fits_cube_batch_fast(q_filename_, batch->h_Q,
            cube_info_.nx, cube_info_.ny, start, count,
            h_chan_map_, nfreq_valid_, nfreq_orig_) != 0) {
        return false;
    }

    if (read_fits_cube_batch_fast(u_filename_, batch->h_U,
            cube_info_.nx, cube_info_.ny, start, count,
            h_chan_map_, nfreq_valid_, nfreq_orig_) != 0) {
        return false;
    }

    if (config_.normalize_by_i && i_filename_) {
        if (read_fits_cube_batch_fast(i_filename_, batch->h_I,
                cube_info_.nx, cube_info_.ny, start, count,
                h_chan_map_, nfreq_valid_, nfreq_orig_) != 0) {
            return false;
        }
    }

    batch->ready_for_gpu = true;
    return true;
}

bool RMSynthPipeline::process_batch_gpu(PixelBatch* batch, int buffer_idx) {
    GPUBuffers* buf = gpu_buffers_[buffer_idx];
    cudaStream_t stream = streams_[buffer_idx];
    int npixels = batch->npixels;
    int nfreq = cube_info_.nfreq;
    int nphi = config_.faraday.nphi;
    
    size_t freq_size = npixels * nfreq * sizeof(float);
    size_t phi_size = npixels * nphi * sizeof(float);
    
    // Upload Q, U
    cudaMemcpyAsync(buf->d_Q, batch->h_Q, freq_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buf->d_U, batch->h_U, freq_size, cudaMemcpyHostToDevice, stream);
    
    // Normalize or copy
    if (config_.normalize_by_i && batch->h_I) {
        cudaMemcpyAsync(buf->d_I, batch->h_I, freq_size, cudaMemcpyHostToDevice, stream);
        cu_normalize_qu(buf->d_Q, buf->d_U, buf->d_I, buf->d_Q_norm, buf->d_U_norm,
                        nfreq, npixels, 1e-10f, stream);
    } else {
        cu_copy_qu(buf->d_Q, buf->d_U, buf->d_Q_norm, buf->d_U_norm,
                   nfreq, npixels, stream);
    }
    
    // RM-Synthesis via NUFFT
    cu_rm_synthesis_nufft(buf, buf->d_Q_norm, buf->d_U_norm, buf->d_weights,
                          buf->d_fdf_real, buf->d_fdf_imag, npixels, stream);

    // Peak finding
    cu_find_fdf_peak(buf->d_fdf_real, buf->d_fdf_imag, buf->d_phi,
                     buf->d_peak_rm, buf->d_peak_pi, buf->d_rm_err,
                     rmsf_fwhm_, nphi, npixels, stream);

    // Copy dirty peaks back BEFORE RM-CLEAN overwrites anything
    cudaMemcpyAsync(batch->h_peak_rm, buf->d_peak_rm,
                    npixels * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(batch->h_peak_pi, buf->d_peak_pi,
                    npixels * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(batch->h_rm_err, buf->d_rm_err,
                    npixels * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // RM-CLEAN
    if (config_.do_rmclean) {
        // Use d_fdf_real/imag as clean output (dirty FDF is read first, then overwritten)
        // Use d_clean_model_real/imag as model workspace
        // Use d_residual_real/imag as residual workspace
        // Use d_peak_rm/pi for clean peak output (dirty already copied above)
        cu_rm_clean(buf->d_fdf_real, buf->d_fdf_imag,
                    buf->d_rmsf_real, buf->d_rmsf_imag, buf->d_phi,
                    buf->d_residual_real, buf->d_residual_imag,   // clean FDF output
                    buf->d_clean_model_real, buf->d_clean_model_imag, // residual workspace
                    buf->d_fdf_real, buf->d_fdf_imag,             // model workspace (dirty no longer needed)
                    buf->d_peak_rm, buf->d_peak_pi,
                    config_.clean, rmsf_fwhm_, nphi, npixels, stream);

        // Copy clean peaks back
        cudaMemcpyAsync(batch->h_clean_peak_rm, buf->d_peak_rm,
                        npixels * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(batch->h_clean_peak_pi, buf->d_peak_pi,
                        npixels * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    
    if (config_.save_fdf_cube) {
        cudaMemcpyAsync(batch->h_fdf_real, buf->d_fdf_real, phi_size,
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(batch->h_fdf_imag, buf->d_fdf_imag, phi_size,
                        cudaMemcpyDeviceToHost, stream);
    }
    
    cudaStreamSynchronize(stream);
    batch->gpu_complete = true;
    
    return true;
}

bool RMSynthPipeline::write_batch(PixelBatch* batch) {
    int start = batch->start_pixel;
    int count = batch->npixels;
    
    // Copy to output arrays
    for (int i = 0; i < count; i++) {
        int global_idx = start + i;
        out_peak_rm_[global_idx] = batch->h_peak_rm[i];
        out_peak_pi_[global_idx] = batch->h_peak_pi[i];
        out_rm_err_[global_idx] = batch->h_rm_err[i];
        
        if (config_.do_rmclean) {
            out_clean_peak_rm_[global_idx] = batch->h_clean_peak_rm[i];
            out_clean_peak_pi_[global_idx] = batch->h_clean_peak_pi[i];
        }
        
        if (config_.save_fdf_cube) {
            int nphi = config_.faraday.nphi;
            for (int p = 0; p < nphi; p++) {
                out_fdf_real_[global_idx * nphi + p] = batch->h_fdf_real[i * nphi + p];
                out_fdf_imag_[global_idx * nphi + p] = batch->h_fdf_imag[i * nphi + p];
            }
        }
    }
    
    batch->written = true;
    return true;
}

// ============================================================================
// MAIN RUN LOOP - ASYNC PIPELINE
// ============================================================================

bool RMSynthPipeline::run() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    printf("Processing %zu pixels in %d batches...\n", cube_info_.npix, total_batches_);
    
    // Allocate two batches for double buffering
    PixelBatch* batches[2];
    batches[0] = allocate_batch(config_.batch.batch_pixels);
    batches[1] = allocate_batch(config_.batch.batch_pixels);

    // Setup helper to init a batch descriptor
    auto setup_batch = [&](int buf_idx, int batch_id) {
        int start = batch_id * config_.batch.batch_pixels;
        int remaining = (int)cube_info_.npix - start;
        int count = (remaining < config_.batch.batch_pixels) ? remaining : config_.batch.batch_pixels;
        batches[buf_idx]->batch_id = batch_id;
        batches[buf_idx]->start_pixel = start;
        batches[buf_idx]->npixels = count;
        batches[buf_idx]->ready_for_gpu = false;
        batches[buf_idx]->gpu_complete = false;
        batches[buf_idx]->written = false;
    };

    // Load first batch synchronously (nothing to overlap with yet)
    setup_batch(0, 0);
    printf("Batch 1/%d: loading...\n", total_batches_);
    load_batch(batches[0]);

    int next_to_load = 1;   // Next batch ID to load
    int next_to_gpu = 0;    // Next batch buffer ready for GPU
    int completed = 0;

    // Pipeline: overlap GPU processing of current batch with I/O loading of next
    while (completed < total_batches_) {
        int cur = next_to_gpu;
        int other = 1 - cur;

        // Start loading NEXT batch in background thread (overlaps with GPU below)
        std::thread* load_thread = nullptr;
        if (next_to_load < total_batches_) {
            setup_batch(other, next_to_load);
            load_thread = new std::thread([this, batch = batches[other]]() {
                load_batch(batch);
            });
            next_to_load++;
        }

        // GPU process current batch (runs while load_thread reads next from disk)
        process_batch_gpu(batches[cur], cur);

        // Write results to output arrays (trivial memcpy)
        write_batch(batches[cur]);
        completed++;
        stats_.batches_written++;
        stats_.pixels_processed += batches[cur]->npixels;

        printf("\rBatch %d/%d complete (%.1f%%)",
               completed, total_batches_,
               100.0f * stats_.pixels_processed / cube_info_.npix);
        fflush(stdout);

        // Wait for next batch load to finish before we can use that buffer
        if (load_thread) {
            load_thread->join();
            delete load_thread;
        }

        // Next iteration processes the freshly loaded buffer
        next_to_gpu = other;
    }
    
    printf("\n");
    
    // Free batch buffers
    free_batch(batches[0]);
    free_batch(batches[1]);
    
    // Write output files
    if (!write_outputs()) {
        fprintf(stderr, "Failed to write output files\n");
        return false;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    stats_.total_time_ms = duration.count();
    
    printf("\nDone! Total time: %.1f s (%.1f pixels/s)\n",
           stats_.total_time_ms / 1000.0,
           cube_info_.npix / (stats_.total_time_ms / 1000.0));
    
    return true;
}

bool RMSynthPipeline::write_outputs() {
    char filename[512];
    
    printf("\nWriting output files:\n");
    
    // Peak RM
    snprintf(filename, sizeof(filename), "%s_peak_rm.fits", config_.output_prefix);
    write_fits_image(filename, out_peak_rm_, cube_info_.ny, cube_info_.nx, "PEAK_RM");
    printf("  %s\n", filename);
    
    // Peak PI  
    snprintf(filename, sizeof(filename), "%s_peak_pi.fits", config_.output_prefix);
    write_fits_image(filename, out_peak_pi_, cube_info_.ny, cube_info_.nx, "PEAK_PI");
    printf("  %s\n", filename);
    
    // RM error
    snprintf(filename, sizeof(filename), "%s_rm_err.fits", config_.output_prefix);
    write_fits_image(filename, out_rm_err_, cube_info_.ny, cube_info_.nx, "RM_ERR");
    printf("  %s\n", filename);
    
    if (config_.do_rmclean) {
        snprintf(filename, sizeof(filename), "%s_clean_peak_rm.fits", config_.output_prefix);
        write_fits_image(filename, out_clean_peak_rm_, cube_info_.ny, cube_info_.nx, "CLEAN_RM");
        printf("  %s\n", filename);
        
        snprintf(filename, sizeof(filename), "%s_clean_peak_pi.fits", config_.output_prefix);
        write_fits_image(filename, out_clean_peak_pi_, cube_info_.ny, cube_info_.nx, "CLEAN_PI");
        printf("  %s\n", filename);
    }
    
    if (config_.save_fdf_cube) {
        snprintf(filename, sizeof(filename), "%s_fdf_real.fits", config_.output_prefix);
        write_fdf_cube(filename, out_fdf_real_, config_.faraday.nphi,
                       cube_info_.ny, cube_info_.nx, config_.faraday);
        printf("  %s\n", filename);
        
        snprintf(filename, sizeof(filename), "%s_fdf_imag.fits", config_.output_prefix);
        write_fdf_cube(filename, out_fdf_imag_, config_.faraday.nphi,
                       cube_info_.ny, cube_info_.nx, config_.faraday);
        printf("  %s\n", filename);
    }
    
    return true;
}

// ============================================================================
// STANDALONE RM-CLEAN
// ============================================================================

bool RMSynthPipeline::run_rmclean_only(
    const char* fdf_real_file,
    const char* fdf_imag_file,
    const char* rmsf_file
) {
    printf("=== VROOM: Standalone RM-CLEAN ===\n\n");
    
    // Load RMSF
    if (load_rmsf(rmsf_file, &rmsf_, &config_.faraday) != 0) {
        fprintf(stderr, "Failed to load RMSF from %s\n", rmsf_file);
        return false;
    }
    rmsf_fwhm_ = rmsf_.fwhm;
    printf("Loaded RMSF: FWHM = %.2f rad/m²\n", rmsf_fwhm_);
    
    // Load FDF cubes
    float* fdf_real = nullptr;
    float* fdf_imag = nullptr;
    CubeInfo fdf_info;
    
    if (read_fits_cube(fdf_real_file, &fdf_real, &fdf_info) != 0) {
        fprintf(stderr, "Failed to load FDF real from %s\n", fdf_real_file);
        return false;
    }
    if (read_fits_cube(fdf_imag_file, &fdf_imag, &fdf_info) != 0) {
        fprintf(stderr, "Failed to load FDF imag from %s\n", fdf_imag_file);
        free(fdf_real);
        return false;
    }
    
    cube_info_.nx = fdf_info.nx;
    cube_info_.ny = fdf_info.ny;
    cube_info_.npix = fdf_info.npix;
    config_.faraday.nphi = fdf_info.nfreq;  // Third axis is Faraday depth
    
    printf("FDF cube: %d x %d x %d\n", cube_info_.nx, cube_info_.ny, config_.faraday.nphi);
    
    // TODO: Implement batched RM-CLEAN on loaded FDF
    // For now this is a placeholder
    
    free(fdf_real);
    free(fdf_imag);
    
    printf("Standalone RM-CLEAN not yet fully implemented\n");
    return false;
}

// ============================================================================
// COMMAND LINE PARSING
// ============================================================================

void print_usage(const char* prog) {
    printf("VROOM - GPU RM-Synthesis + RM-CLEAN\n\n");
    printf("Usage:\n");
    printf("  %s [options] -q Q.fits -u U.fits -f freq.txt -o output_prefix\n\n", prog);
    printf("Required:\n");
    printf("  -q, --q-cube FILE        Q Stokes cube (FITS)\n");
    printf("  -u, --u-cube FILE        U Stokes cube (FITS)\n");
    printf("  -f, --freq FILE          Frequency file (Hz, one per line)\n");
    printf("  -o, --output PREFIX      Output file prefix\n\n");
    printf("Optional:\n");
    printf("  -i, --i-cube FILE        Model I cube for normalization (Q/I, U/I)\n");
    printf("  -w, --weights FILE       Per-channel weights file\n");
    printf("  --phi-min VALUE          Min Faraday depth [rad/m²] (default: -500)\n");
    printf("  --phi-max VALUE          Max Faraday depth [rad/m²] (default: +500)\n");
    printf("  --dphi VALUE             Faraday depth step [rad/m²] (default: 1)\n");
    printf("  --no-clean               Skip RM-CLEAN, only do RM-synthesis\n");
    printf("  --clean-gain VALUE       RM-CLEAN loop gain (default: 0.1)\n");
    printf("  --clean-cutoff VALUE     RM-CLEAN threshold (default: 3-sigma)\n");
    printf("  --clean-maxiter VALUE    Max CLEAN iterations (default: 1000)\n");
    printf("  --save-fdf               Save full FDF cube (warning: large!)\n");
    printf("  --save-rmsf              Save RMSF to file\n");
    printf("  -h, --help               Show this help\n\n");
    printf("Examples:\n");
    printf("  %s -q Q.fits -u U.fits -f freq.txt -o results\n", prog);
    printf("  %s -q Q.fits -u U.fits -i I.fits -f freq.txt -o results --save-rmsf\n", prog);
}

int parse_args(int argc, char** argv, 
               const char** q_file, const char** u_file, const char** i_file,
               const char** freq_file, const char** weights_file, 
               PipelineConfig* config) {
    
    *q_file = nullptr;
    *u_file = nullptr;
    *i_file = nullptr;
    *freq_file = nullptr;
    *weights_file = nullptr;
    
    // Defaults
    config->faraday.phi_min = -500.0f;
    config->faraday.phi_max = 500.0f;
    config->faraday.dphi = 1.0f;
    
    config->clean.gain = 0.1f;
    config->clean.threshold = 0.0f;  // Will be set to 3-sigma
    config->clean.threshold_rel = 0.001f;
    config->clean.max_iter = 1000;
    config->clean.use_threshold_rel = true;
    
    config->do_rmclean = true;
    config->save_fdf_cube = false;
    config->save_rmsf = false;
    config->normalize_by_i = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--q-cube") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            *q_file = argv[i];
        } else if (strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "--u-cube") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            *u_file = argv[i];
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--i-cube") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            *i_file = argv[i];
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--freq") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            *freq_file = argv[i];
        } else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--weights") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            *weights_file = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            strncpy(config->output_prefix, argv[i], sizeof(config->output_prefix) - 1);
        } else if (strcmp(argv[i], "--phi-min") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            config->faraday.phi_min = atof(argv[i]);
        } else if (strcmp(argv[i], "--phi-max") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            config->faraday.phi_max = atof(argv[i]);
        } else if (strcmp(argv[i], "--dphi") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            config->faraday.dphi = atof(argv[i]);
        } else if (strcmp(argv[i], "--no-clean") == 0) {
            config->do_rmclean = false;
        } else if (strcmp(argv[i], "--clean-gain") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            config->clean.gain = atof(argv[i]);
        } else if (strcmp(argv[i], "--clean-cutoff") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            config->clean.threshold = atof(argv[i]);
            config->clean.use_threshold_rel = false;
        } else if (strcmp(argv[i], "--clean-maxiter") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing argument for %s\n", argv[i-1]); return -1; }
            config->clean.max_iter = atoi(argv[i]);
        } else if (strcmp(argv[i], "--save-fdf") == 0) {
            config->save_fdf_cube = true;
        } else if (strcmp(argv[i], "--save-rmsf") == 0) {
            config->save_rmsf = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            return 1;  // Signal to print help
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    
    // Validate required args
    if (!*q_file || !*u_file || !*freq_file || config->output_prefix[0] == '\0') {
        fprintf(stderr, "Error: Missing required arguments\n");
        return -1;
    }
    
    return 0;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    const char* q_file;
    const char* u_file;
    const char* i_file;
    const char* freq_file;
    const char* weights_file;
    PipelineConfig config;
    memset(&config, 0, sizeof(config));
    
    int result = parse_args(argc, argv, &q_file, &u_file, &i_file, 
                            &freq_file, &weights_file, &config);
    if (result != 0) {
        print_usage(argv[0]);
        return (result < 0) ? 1 : 0;
    }
    
    // Create and run pipeline
    RMSynthPipeline pipeline(config);
    
    if (!pipeline.init(q_file, u_file, freq_file, i_file, weights_file)) {
        fprintf(stderr, "Pipeline initialization failed\n");
        return 1;
    }
    
    if (!pipeline.run()) {
        fprintf(stderr, "Pipeline execution failed\n");
        return 1;
    }
    
    return 0;
}
