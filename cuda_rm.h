#ifndef CUDA_RM_H
#define CUDA_RM_H

#include <cuda_runtime.h>

// ============================================================================
// GPU FUNCTION DECLARATIONS
// ============================================================================

// Spline fit to I(freq)
void cu_spline_fit(
    const float *d_freq,
    const float *d_I,
    float *d_I_fit,
    float *d_I_residual,
    int NFREQ,
    int N_valid
);

// Normalize Q, U by fitted I
void cu_normalize_qu(
    const float *d_I_fit,
    float *d_Q_norm,
    float *d_U_norm,
    int NFREQ,
    int N_valid
);

// Compute λ² for all frequencies
void cu_compute_lambda_sq(
    const float *d_freq,
    float *d_lambda_sq,
    int NFREQ
);

// Compute RMSF per pixel
void cu_compute_rmsf(
    const float *d_Q,
    const float *d_U,
    const float *d_lambda_sq,
    const float *d_weights,
    const float *d_fara_grid,
    float2 *d_rmsf,
    int NFREQ,
    int NFARA,
    int N_valid
);

// Find peak RM and uncertainty
void cu_find_peak_rm(
    const float2 *d_rmsf,
    const float *d_fara_grid,
    float *d_peak_rm,
    float *d_rm_value,
    float *d_rm_err,
    int NFARA,
    int N_valid
);

// RM-CLEAN deconvolution
void cu_rm_clean(
    const float2 *d_rmsf,
    const float *d_rm_value,
    const float *d_peak_rm,
    const float *d_fara_grid,
    float *d_rm_clean,
    int NFARA,
    int N_valid,
    float gain,
    float threshold,
    int max_iter
);

#endif // CUDA_RM_H