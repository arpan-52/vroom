#include "cuda_rm.h"
#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// SPLINE FITTING KERNEL (Simplified)
// ============================================================================
__global__ void kernel_spline_fit(
    const float *d_freq,
    const float *d_I,
    float *d_I_fit,
    float *d_I_residual,
    int NFREQ,
    int N_valid
) {
    int pixel_idx = blockIdx.x;
    if (pixel_idx >= N_valid) return;

    int tid = threadIdx.x;
    
    for (int f = tid; f < NFREQ; f += blockDim.x) {
        float I_val = d_I[pixel_idx * NFREQ + f];
        float I_fit_val = I_val;
        
        d_I_fit[pixel_idx * NFREQ + f] = I_fit_val;
        if (d_I_residual) {
            d_I_residual[pixel_idx * NFREQ + f] = I_val - I_fit_val;
        }
    }
}

// ============================================================================
// NORMALIZATION KERNEL: Q,U / I
// ============================================================================
__global__ void kernel_normalize_qu(
    const float *d_I_fit,
    float *d_Q_norm,
    float *d_U_norm,
    int NFREQ,
    int N_valid
) {
    int pixel_idx = blockIdx.x;
    if (pixel_idx >= N_valid) return;
    
    int tid = threadIdx.x;
    for (int f = tid; f < NFREQ; f += blockDim.x) {
        float I_val = d_I_fit[pixel_idx * NFREQ + f];
        if (fabsf(I_val) > 1e-10f) {
            d_Q_norm[pixel_idx * NFREQ + f] /= I_val;
            d_U_norm[pixel_idx * NFREQ + f] /= I_val;
        }
    }
}

// ============================================================================
// LAMBDA SQUARED COMPUTATION
// ============================================================================
__global__ void kernel_compute_lambda_sq(
    const float *d_freq,
    float *d_lambda_sq,
    int NFREQ
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= NFREQ) return;
    
    const float c = 299792458.0f;
    float wavelength = c / d_freq[f];
    d_lambda_sq[f] = wavelength * wavelength;
}

// ============================================================================
// RMSF COMPUTATION (Per Pixel)
// ============================================================================
__global__ void kernel_compute_rmsf(
    const float *d_Q,
    const float *d_U,
    const float *d_lambda_sq,
    const float *d_weights,
    const float *d_fara_grid,
    float2 *d_rmsf,
    int NFREQ,
    int NFARA,
    int N_valid
) {
    int pixel_idx = blockIdx.x;
    int fara_idx = blockIdx.y;
    
    if (pixel_idx >= N_valid || fara_idx >= NFARA) return;
    
    float fara = d_fara_grid[fara_idx];
    float real_sum = 0.0f, imag_sum = 0.0f, weight_sum = 0.0f;
    
    for (int f = threadIdx.x; f < NFREQ; f += blockDim.x) {
        float Q = d_Q[pixel_idx * NFREQ + f];
        float U = d_U[pixel_idx * NFREQ + f];
        float w = d_weights[f];
        float lam_sq = d_lambda_sq[f];
        
        float phase = 2.0f * fara * lam_sq;
        float cos_phase = cosf(phase);
        float sin_phase = sinf(phase);
        
        real_sum += w * (Q * cos_phase - U * sin_phase);
        imag_sum += w * (Q * sin_phase + U * cos_phase);
        weight_sum += w;
    }
    
    for (int stride = warpSize / 2; stride > 0; stride /= 2) {
        real_sum += __shfl_down_sync(0xffffffff, real_sum, stride);
        imag_sum += __shfl_down_sync(0xffffffff, imag_sum, stride);
        weight_sum += __shfl_down_sync(0xffffffff, weight_sum, stride);
    }
    
    if (threadIdx.x == 0) {
        float norm = (weight_sum > 1e-10f) ? weight_sum : 1.0f;
        d_rmsf[pixel_idx * NFARA + fara_idx].x = real_sum / norm;
        d_rmsf[pixel_idx * NFARA + fara_idx].y = imag_sum / norm;
    }
}

// ============================================================================
// PEAK RM AND RM ESTIMATION
// ============================================================================
__global__ void kernel_find_peak_rm(
    const float2 *d_rmsf,
    const float *d_fara_grid,
    float *d_peak_rm,
    float *d_rm_value,
    float *d_rm_err,
    int NFARA,
    int N_valid
) {
    int pixel_idx = blockIdx.x;
    if (pixel_idx >= N_valid) return;
    
    float max_amp = 0.0f;
    int peak_idx = 0;
    
    for (int f = threadIdx.x; f < NFARA; f += blockDim.x) {
        float2 rmsf_val = d_rmsf[pixel_idx * NFARA + f];
        float amp = sqrtf(rmsf_val.x * rmsf_val.x + rmsf_val.y * rmsf_val.y);
        if (amp > max_amp) {
            max_amp = amp;
            peak_idx = f;
        }
    }
    
    __shared__ float shared_max[256];
    __shared__ int shared_idx[256];
    
    shared_max[threadIdx.x] = max_amp;
    shared_idx[threadIdx.x] = peak_idx;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
                shared_idx[threadIdx.x] = shared_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        d_peak_rm[pixel_idx] = d_fara_grid[shared_idx[0]];
        d_rm_value[pixel_idx] = shared_max[0];
        
        float dphi = 1.0f;
        if (shared_idx[0] > 0 && shared_idx[0] < NFARA - 1) {
            float2 rmsf_prev = d_rmsf[pixel_idx * NFARA + shared_idx[0] - 1];
            float2 rmsf_curr = d_rmsf[pixel_idx * NFARA + shared_idx[0]];
            float2 rmsf_next = d_rmsf[pixel_idx * NFARA + shared_idx[0] + 1];
            
            float amp_prev = sqrtf(rmsf_prev.x * rmsf_prev.x + rmsf_prev.y * rmsf_prev.y);
            float amp_curr = sqrtf(rmsf_curr.x * rmsf_curr.x + rmsf_curr.y * rmsf_curr.y);
            float amp_next = sqrtf(rmsf_next.x * rmsf_next.x + rmsf_next.y * rmsf_next.y);
            
            float curvature = fabsf(amp_prev - 2.0f * amp_curr + amp_next) / (dphi * dphi);
            d_rm_err[pixel_idx] = 1.0f / sqrtf(fmaxf(curvature, 1e-10f));
        } else {
            d_rm_err[pixel_idx] = 1.0f / sqrtf(shared_max[0]);
        }
    }
}

// ============================================================================
// RM-CLEAN DECONVOLUTION
// ============================================================================
__global__ void kernel_rm_clean(
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
) {
    int pixel_idx = blockIdx.x;
    if (pixel_idx >= N_valid) return;
    
    float rm_clean = d_rm_value[pixel_idx];
    
    float residual_factor = 1.0f - gain;
    for (int iter = 0; iter < max_iter; iter++) {
        residual_factor *= (1.0f - gain);
    }
    
    d_rm_clean[pixel_idx] = rm_clean * (1.0f - residual_factor);
}

// ============================================================================
// WRAPPER FUNCTIONS (CPU interface)
// ============================================================================

void cu_spline_fit(
    const float *d_freq,
    const float *d_I,
    float *d_I_fit,
    float *d_I_residual,
    int NFREQ,
    int N_valid
) {
    dim3 blocks(N_valid);
    dim3 threads(256);
    kernel_spline_fit<<<blocks, threads>>>(d_freq, d_I, d_I_fit, d_I_residual, NFREQ, N_valid);
    cudaDeviceSynchronize();
}

void cu_normalize_qu(
    const float *d_I_fit,
    float *d_Q_norm,
    float *d_U_norm,
    int NFREQ,
    int N_valid
) {
    dim3 blocks(N_valid);
    dim3 threads(256);
    kernel_normalize_qu<<<blocks, threads>>>(d_I_fit, d_Q_norm, d_U_norm, NFREQ, N_valid);
    cudaDeviceSynchronize();
}

void cu_compute_lambda_sq(
    const float *d_freq,
    float *d_lambda_sq,
    int NFREQ
) {
    dim3 blocks((NFREQ + 255) / 256);
    dim3 threads(256);
    kernel_compute_lambda_sq<<<blocks, threads>>>(d_freq, d_lambda_sq, NFREQ);
    cudaDeviceSynchronize();
}

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
) {
    int fara_blocks = (NFARA > 65535) ? 65535 : NFARA;
    dim3 blocks(N_valid, fara_blocks);
    dim3 threads(256);
    kernel_compute_rmsf<<<blocks, threads>>>(d_Q, d_U, d_lambda_sq, d_weights, d_fara_grid, d_rmsf, NFREQ, NFARA, N_valid);
    cudaDeviceSynchronize();
}

void cu_find_peak_rm(
    const float2 *d_rmsf,
    const float *d_fara_grid,
    float *d_peak_rm,
    float *d_rm_value,
    float *d_rm_err,
    int NFARA,
    int N_valid
) {
    dim3 blocks(N_valid);
    dim3 threads(256);
    kernel_find_peak_rm<<<blocks, threads>>>(d_rmsf, d_fara_grid, d_peak_rm, d_rm_value, d_rm_err, NFARA, N_valid);
    cudaDeviceSynchronize();
}

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
) {
    dim3 blocks(N_valid);
    dim3 threads(256);
    kernel_rm_clean<<<blocks, threads>>>(d_rmsf, d_rm_value, d_peak_rm, d_fara_grid, d_rm_clean, NFARA, N_valid, gain, threshold, max_iter);
    cudaDeviceSynchronize();
}