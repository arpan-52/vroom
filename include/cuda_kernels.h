#ifndef VROOM_CUDA_KERNELS_H
#define VROOM_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <cufinufft.h>
#include "vroom.h"

// ============================================================================
// NUFFT-BASED RM-SYNTHESIS
// ============================================================================
// Uses cuFINUFFT (https://github.com/flatironinstitute/cufinufft)
//
// RM-synthesis as Type-1 NUFFT:
//   FDF(φ) = Σ_j w_j P_j exp(-2i φ (λ²_j - λ²_0))
//
// Mapping to NUFFT Type-1 (nonuniform to uniform):
//   - Nonuniform points: x_j = -2 * (λ²_j - λ²_0) * dphi / (2π)
//   - Strengths: c_j = w_j * (Q_j + i*U_j)  
//   - Uniform output grid: FDF[k] for k = 0..nphi-1
//
// Complexity: O(N_freq log N_freq + N_phi log N_phi) vs O(N_freq * N_phi) for DFT

// ============================================================================
// GPU MEMORY MANAGEMENT
// ============================================================================

struct GPUBuffers {
    // Input data (per batch)
    float* d_Q;             // [batch_pixels * nfreq]
    float* d_U;             // [batch_pixels * nfreq]
    float* d_I;             // [batch_pixels * nfreq] (optional, for normalization)
    
    // Normalized Q/U
    float* d_Q_norm;        // [batch_pixels * nfreq]
    float* d_U_norm;        // [batch_pixels * nfreq]
    
    // Precomputed constants (persistent across batches)
    float* d_freq;          // [nfreq]
    float* d_lambda_sq;     // [nfreq]
    float* d_weights;       // [nfreq]
    float* d_phi;           // [nphi] Faraday depth grid
    
    // NUFFT coordinates (precomputed from lambda_sq) - single precision
    float* d_nufft_x;       // [nfreq] scaled λ² for NUFFT
    
    // NUFFT workspace - single precision
    cufinufftf_plan nufft_plan;      // Single precision plan
    cuFloatComplex* d_nufft_c;       // [batch_pixels * nfreq] input strengths
    cuFloatComplex* d_nufft_f;       // [batch_pixels * nphi] output FDF
    
    // RMSF (computed once)
    float* d_rmsf_real;     // [nphi]
    float* d_rmsf_imag;     // [nphi]
    
    // FDF output (per batch) - float version for downstream
    float* d_fdf_real;      // [batch_pixels * nphi]
    float* d_fdf_imag;      // [batch_pixels * nphi]
    
    // Peak finding output (per batch)
    float* d_peak_rm;       // [batch_pixels]
    float* d_peak_pi;       // [batch_pixels]
    float* d_rm_err;        // [batch_pixels]
    
    // RM-CLEAN workspace (per batch)
    float* d_residual_real; // [batch_pixels * nphi]
    float* d_residual_imag; // [batch_pixels * nphi]
    float* d_clean_model_real;  // [batch_pixels * nphi]
    float* d_clean_model_imag;  // [batch_pixels * nphi]
    
    // Dimensions
    int nfreq;
    int nphi;
    int batch_pixels;
    int nufft_ntransf;  // NUFFT plan batch size (may be < batch_pixels)
    bool has_model_i;
};

// Maximum NUFFT sub-batch size (limits NUFFT workspace memory)
constexpr int NUFFT_MAX_BATCH = 128;

// Allocate GPU buffers for a given batch size
GPUBuffers* allocate_gpu_buffers(
    int nfreq,
    int nphi,
    int batch_pixels,
    bool need_model_i,
    bool need_rmclean
);

// Free GPU buffers
void free_gpu_buffers(GPUBuffers* buffers);

// ============================================================================
// PREPROCESSING KERNELS
// ============================================================================

// Compute λ² from frequencies
void cu_compute_lambda_sq(
    const float* d_freq,
    float* d_lambda_sq,
    int nfreq,
    cudaStream_t stream = 0
);

// Generate Faraday depth grid
void cu_generate_phi_grid(
    float* d_phi,
    float phi_min,
    float dphi,
    int nphi,
    cudaStream_t stream = 0
);

// Compute mean λ² (needed for RMSF centering)
void cu_compute_mean_lambda_sq(
    const float* d_lambda_sq,
    const float* d_weights,
    float* d_mean_lambda_sq,
    int nfreq,
    cudaStream_t stream = 0
);

// ============================================================================
// NORMALIZATION KERNEL
// ============================================================================

// Normalize Q/U by model I
// Handles division by zero: sets Q_norm/U_norm to 0 where |I| < threshold
void cu_normalize_qu(
    const float* d_Q,
    const float* d_U,
    const float* d_I,
    float* d_Q_norm,
    float* d_U_norm,
    int nfreq,
    int npixels,
    float zero_threshold,   // Values of |I| below this are treated as zero
    cudaStream_t stream = 0
);

// Copy Q/U to normalized buffers (when no model I provided)
void cu_copy_qu(
    const float* d_Q,
    const float* d_U,
    float* d_Q_norm,
    float* d_U_norm,
    int nfreq,
    int npixels,
    cudaStream_t stream = 0
);

// ============================================================================
// RMSF COMPUTATION
// ============================================================================

// Compute theoretical RMSF (same for all pixels with uniform weighting)
// This is the response to a delta function at φ=0
void cu_compute_rmsf(
    const float* d_lambda_sq,
    const float* d_weights,
    const float* d_phi,
    float* d_rmsf_real,
    float* d_rmsf_imag,
    float mean_lambda_sq,
    int nfreq,
    int nphi,
    cudaStream_t stream = 0
);

// Find RMSF FWHM
float cu_compute_rmsf_fwhm(
    const float* d_rmsf_real,
    const float* d_rmsf_imag,
    const float* d_phi,
    int nphi
);

// ============================================================================
// RM-SYNTHESIS VIA NUFFT
// ============================================================================

// Initialize NUFFT plan for RM-synthesis
// Call once per batch size configuration
// Returns 0 on success
int cu_init_nufft_plan(
    GPUBuffers* buffers,
    float phi_min,
    float dphi,
    float mean_lambda_sq,
    cudaStream_t stream = 0
);

// Destroy NUFFT plan
void cu_destroy_nufft_plan(GPUBuffers* buffers);

// Compute Faraday Dispersion Function (FDF) for a batch of pixels using NUFFT
// FDF(φ) = K ∑_i w_i P_i exp(-2i φ (λ²_i - λ²_0))
// where P = Q + iU (complex polarization)
//
// Uses Type-1 NUFFT: nonuniform λ² -> uniform φ grid
void cu_rm_synthesis_nufft(
    GPUBuffers* buffers,
    const float* d_Q,           // [npixels * nfreq] normalized Q
    const float* d_U,           // [npixels * nfreq] normalized U
    const float* d_weights,     // [nfreq]
    float* d_fdf_real,          // [npixels * nphi] output
    float* d_fdf_imag,          // [npixels * nphi] output
    int npixels,
    cudaStream_t stream = 0
);

// ============================================================================
// PEAK FINDING
// ============================================================================

// Find peak in FDF and estimate RM with uncertainty
// Uses parabolic interpolation for sub-channel precision
void cu_find_fdf_peak(
    const float* d_fdf_real,    // [npixels * nphi]
    const float* d_fdf_imag,    // [npixels * nphi]
    const float* d_phi,         // [nphi]
    float* d_peak_rm,           // [npixels] output: RM at peak
    float* d_peak_pi,           // [npixels] output: polarized intensity at peak
    float* d_rm_err,            // [npixels] output: RM uncertainty
    float rmsf_fwhm,            // RMSF FWHM for error estimation
    int nphi,
    int npixels,
    cudaStream_t stream = 0
);

// ============================================================================
// RM-CLEAN
// ============================================================================

// Full RM-CLEAN algorithm:
// 1. Find peak in dirty FDF
// 2. Subtract gain * RMSF centered at peak
// 3. Add delta component to clean model
// 4. Repeat until threshold or max_iter
// 5. Convolve clean model with restoring beam
// 6. Add residuals
void cu_rm_clean(
    const float* d_fdf_real,        // [npixels * nphi] dirty FDF
    const float* d_fdf_imag,        // [npixels * nphi]
    const float* d_rmsf_real,       // [nphi] RMSF
    const float* d_rmsf_imag,       // [nphi]
    const float* d_phi,             // [nphi]
    float* d_clean_fdf_real,        // [npixels * nphi] output: cleaned FDF
    float* d_clean_fdf_imag,        // [npixels * nphi] output
    float* d_residual_real,         // [npixels * nphi] workspace/output: residuals
    float* d_residual_imag,         // [npixels * nphi]
    float* d_model_real,            // [npixels * nphi] workspace: clean components
    float* d_model_imag,            // [npixels * nphi]
    float* d_clean_peak_rm,         // [npixels] output: peak RM after clean
    float* d_clean_peak_pi,         // [npixels] output: peak PI after clean
    const CleanConfig& config,
    float rmsf_fwhm,                // For restoring beam
    int nphi,
    int npixels,
    cudaStream_t stream = 0
);

// Simplified version that just returns cleaned peak values
// (doesn't save full cleaned FDF cube)
void cu_rm_clean_peaks_only(
    const float* d_fdf_real,
    const float* d_fdf_imag,
    const float* d_rmsf_real,
    const float* d_rmsf_imag,
    const float* d_phi,
    float* d_workspace,             // [npixels * nphi * 2] for residuals
    float* d_clean_peak_rm,         // [npixels] output
    float* d_clean_peak_pi,         // [npixels] output
    const CleanConfig& config,
    int nphi,
    int npixels,
    cudaStream_t stream = 0
);

#endif // VROOM_CUDA_KERNELS_H
