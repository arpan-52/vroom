#ifndef VROOM_H
#define VROOM_H

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// ============================================================================
// CONFIGURATION & DATA STRUCTURES
// ============================================================================

struct CubeInfo {
    int nfreq;      // Number of frequency channels
    int ny;         // Spatial Y dimension
    int nx;         // Spatial X dimension
    size_t npix;    // Total pixels (ny * nx)
    size_t nelems;  // Total elements (nfreq * ny * nx)
};

struct FaradayConfig {
    float phi_min;      // Minimum Faraday depth [rad/m²]
    float phi_max;      // Maximum Faraday depth [rad/m²]
    float dphi;         // Faraday depth resolution [rad/m²]
    int nphi;           // Number of Faraday depth channels
};

struct CleanConfig {
    float gain;             // Loop gain (typically 0.1)
    float threshold;        // Absolute threshold to stop cleaning
    float threshold_rel;    // Relative threshold (fraction of peak)
    int max_iter;           // Maximum iterations per pixel
    bool use_threshold_rel; // Use relative threshold instead of absolute
};

struct BatchConfig {
    size_t ram_limit;       // Available RAM in bytes
    size_t vram_limit;      // Available VRAM in bytes
    int batch_pixels;       // Number of pixels per batch (auto-computed)
    int num_streams;        // Number of CUDA streams for async
};

struct PipelineConfig {
    FaradayConfig faraday;
    CleanConfig clean;
    BatchConfig batch;
    
    bool normalize_by_i;    // Whether to normalize Q/U by model I
    bool do_rmclean;        // Whether to run RM-CLEAN
    bool save_fdf_cube;     // Whether to save full FDF cube
    bool save_rmsf;         // Whether to save RMSF (theoretical)
    
    char output_prefix[256];
};

// ============================================================================
// OUTPUT PRODUCTS
// ============================================================================

// Per-pixel RM-synthesis outputs
struct RMSynthOutput {
    float* fdf_real;        // Faraday dispersion function (real) [npix * nphi]
    float* fdf_imag;        // Faraday dispersion function (imag) [npix * nphi]
    float* fdf_amp;         // |FDF| amplitude [npix * nphi] (optional)
    float* peak_rm;         // Peak RM value [npix]
    float* peak_pi;         // Peak polarized intensity [npix]
    float* rm_err;          // RM uncertainty [npix]
};

// Per-pixel RM-CLEAN outputs  
struct RMCleanOutput {
    float* clean_fdf_real;  // Cleaned FDF (real) [npix * nphi]
    float* clean_fdf_imag;  // Cleaned FDF (imag) [npix * nphi]
    float* clean_components_rm;   // Clean component locations [npix * max_components]
    float* clean_components_pi;   // Clean component amplitudes [npix * max_components]
    int* clean_components_count;  // Number of components per pixel [npix]
    float* residual_fdf_real;     // Residual FDF (real) [npix * nphi]
    float* residual_fdf_imag;     // Residual FDF (imag) [npix * nphi]
};

// Theoretical RMSF (same for all pixels if uniform weighting)
struct RMSF {
    float* rmsf_real;       // RMSF real part [nphi]
    float* rmsf_imag;       // RMSF imaginary part [nphi]
    float* rmsf_amp;        // |RMSF| [nphi]
    float fwhm;             // FWHM of main lobe [rad/m²]
    float max_scale;        // Maximum recoverable scale
};

// ============================================================================
// MEMORY INFO
// ============================================================================

struct MemoryInfo {
    size_t ram_total;
    size_t ram_available;
    size_t vram_total;
    size_t vram_available;
};

MemoryInfo get_memory_info();
void print_memory_info(const MemoryInfo& info);

// ============================================================================
// BATCH COMPUTATION
// ============================================================================

int compute_optimal_batch_size(
    const CubeInfo& cube,
    const FaradayConfig& faraday,
    const MemoryInfo& mem,
    bool do_rmclean
);

#endif // VROOM_H
