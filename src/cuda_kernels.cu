#include "cuda_kernels.h"
#include <cstdio>
#include <cmath>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// CONSTANTS
// ============================================================================

constexpr float C_LIGHT = 299792458.0f;  // Speed of light [m/s]
constexpr float PI = 3.14159265358979323846f;
constexpr int BLOCK_SIZE = 256;

// ============================================================================
// GPU BUFFER MANAGEMENT
// ============================================================================

GPUBuffers* allocate_gpu_buffers(
    int nfreq,
    int nphi,
    int batch_pixels,
    bool need_model_i,
    bool need_rmclean
) {
    GPUBuffers* buf = new GPUBuffers();
    buf->nfreq = nfreq;
    buf->nphi = nphi;
    buf->batch_pixels = batch_pixels;
    buf->nufft_ntransf = 0;
    buf->has_model_i = need_model_i;
    buf->nufft_plan = nullptr;

    size_t freq_size = nfreq * sizeof(float);
    size_t phi_size = nphi * sizeof(float);
    size_t batch_freq_size = (size_t)batch_pixels * nfreq * sizeof(float);
    size_t batch_phi_size = (size_t)batch_pixels * nphi * sizeof(float);
    size_t batch_pix_size = batch_pixels * sizeof(float);

    // Track total allocated
    size_t total_alloc = 0;
    cudaError_t err;

    #define SAFE_MALLOC(ptr, sz) do { \
        err = cudaMalloc(&(ptr), (sz)); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "cudaMalloc failed (%zu bytes, total so far %.1f MB): %s\n", \
                    (size_t)(sz), total_alloc / 1e6, cudaGetErrorString(err)); \
            cudaGetLastError(); /* clear sticky error */ \
            delete buf; \
            return nullptr; \
        } \
        total_alloc += (sz); \
    } while(0)

    // Input buffers
    SAFE_MALLOC(buf->d_Q, batch_freq_size);
    SAFE_MALLOC(buf->d_U, batch_freq_size);
    if (need_model_i) {
        SAFE_MALLOC(buf->d_I, batch_freq_size);
    } else {
        buf->d_I = nullptr;
    }

    // Normalized Q/U
    SAFE_MALLOC(buf->d_Q_norm, batch_freq_size);
    SAFE_MALLOC(buf->d_U_norm, batch_freq_size);

    // Persistent constants
    SAFE_MALLOC(buf->d_freq, freq_size);
    SAFE_MALLOC(buf->d_lambda_sq, freq_size);
    SAFE_MALLOC(buf->d_weights, freq_size);
    SAFE_MALLOC(buf->d_phi, phi_size);

    // NUFFT coordinates and workspace (single precision)
    int nufft_batch = (batch_pixels < NUFFT_MAX_BATCH) ? batch_pixels : NUFFT_MAX_BATCH;
    SAFE_MALLOC(buf->d_nufft_x, nfreq * sizeof(float));
    SAFE_MALLOC(buf->d_nufft_c, (size_t)nufft_batch * nfreq * sizeof(cuFloatComplex));
    SAFE_MALLOC(buf->d_nufft_f, (size_t)nufft_batch * nphi * sizeof(cuFloatComplex));

    // RMSF
    SAFE_MALLOC(buf->d_rmsf_real, phi_size);
    SAFE_MALLOC(buf->d_rmsf_imag, phi_size);

    // FDF output
    SAFE_MALLOC(buf->d_fdf_real, batch_phi_size);
    SAFE_MALLOC(buf->d_fdf_imag, batch_phi_size);

    // Peak output
    SAFE_MALLOC(buf->d_peak_rm, batch_pix_size);
    SAFE_MALLOC(buf->d_peak_pi, batch_pix_size);
    SAFE_MALLOC(buf->d_rm_err, batch_pix_size);

    // RM-CLEAN workspace
    if (need_rmclean) {
        SAFE_MALLOC(buf->d_residual_real, batch_phi_size);
        SAFE_MALLOC(buf->d_residual_imag, batch_phi_size);
        SAFE_MALLOC(buf->d_clean_model_real, batch_phi_size);
        SAFE_MALLOC(buf->d_clean_model_imag, batch_phi_size);
    } else {
        buf->d_residual_real = nullptr;
        buf->d_residual_imag = nullptr;
        buf->d_clean_model_real = nullptr;
        buf->d_clean_model_imag = nullptr;
    }

    #undef SAFE_MALLOC

    printf("GPU buffer set allocated: %.1f MB (%d pixels)\n", total_alloc / 1e6, batch_pixels);
    return buf;
}

void free_gpu_buffers(GPUBuffers* buf) {
    if (!buf) return;
    
    // Destroy NUFFT plan if exists
    cu_destroy_nufft_plan(buf);
    
    cudaFree(buf->d_Q);
    cudaFree(buf->d_U);
    if (buf->d_I) cudaFree(buf->d_I);
    cudaFree(buf->d_Q_norm);
    cudaFree(buf->d_U_norm);
    cudaFree(buf->d_freq);
    cudaFree(buf->d_lambda_sq);
    cudaFree(buf->d_weights);
    cudaFree(buf->d_phi);
    cudaFree(buf->d_nufft_x);
    cudaFree(buf->d_nufft_c);
    cudaFree(buf->d_nufft_f);
    cudaFree(buf->d_rmsf_real);
    cudaFree(buf->d_rmsf_imag);
    cudaFree(buf->d_fdf_real);
    cudaFree(buf->d_fdf_imag);
    cudaFree(buf->d_peak_rm);
    cudaFree(buf->d_peak_pi);
    cudaFree(buf->d_rm_err);
    if (buf->d_residual_real) cudaFree(buf->d_residual_real);
    if (buf->d_residual_imag) cudaFree(buf->d_residual_imag);
    if (buf->d_clean_model_real) cudaFree(buf->d_clean_model_real);
    if (buf->d_clean_model_imag) cudaFree(buf->d_clean_model_imag);
    
    delete buf;
}

// ============================================================================
// PREPROCESSING KERNELS
// ============================================================================

__global__ void kernel_compute_lambda_sq(
    const float* __restrict__ d_freq,
    float* __restrict__ d_lambda_sq,
    int nfreq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nfreq) return;
    
    float freq = d_freq[idx];
    float lambda = C_LIGHT / freq;
    d_lambda_sq[idx] = lambda * lambda;
}

void cu_compute_lambda_sq(
    const float* d_freq,
    float* d_lambda_sq,
    int nfreq,
    cudaStream_t stream
) {
    int blocks = (nfreq + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_compute_lambda_sq<<<blocks, BLOCK_SIZE, 0, stream>>>(d_freq, d_lambda_sq, nfreq);
}

__global__ void kernel_generate_phi_grid(
    float* __restrict__ d_phi,
    float phi_min,
    float dphi,
    int nphi
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nphi) return;
    
    d_phi[idx] = phi_min + idx * dphi;
}

void cu_generate_phi_grid(
    float* d_phi,
    float phi_min,
    float dphi,
    int nphi,
    cudaStream_t stream
) {
    int blocks = (nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_generate_phi_grid<<<blocks, BLOCK_SIZE, 0, stream>>>(d_phi, phi_min, dphi, nphi);
}

// Compute weighted mean of lambda_sq
__global__ void kernel_compute_mean_lambda_sq(
    const float* __restrict__ d_lambda_sq,
    const float* __restrict__ d_weights,
    float* __restrict__ d_result,  // [2]: sum(w*l2), sum(w)
    int nfreq
) {
    __shared__ float s_wl2[BLOCK_SIZE];
    __shared__ float s_w[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float wl2 = 0.0f, w = 0.0f;
    if (idx < nfreq) {
        w = d_weights[idx];
        wl2 = w * d_lambda_sq[idx];
    }
    
    s_wl2[tid] = wl2;
    s_w[tid] = w;
    __syncthreads();
    
    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_wl2[tid] += s_wl2[tid + s];
            s_w[tid] += s_w[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&d_result[0], s_wl2[0]);
        atomicAdd(&d_result[1], s_w[0]);
    }
}

void cu_compute_mean_lambda_sq(
    const float* d_lambda_sq,
    const float* d_weights,
    float* d_mean_lambda_sq,
    int nfreq,
    cudaStream_t stream
) {
    float* d_temp;
    cudaMalloc(&d_temp, 2 * sizeof(float));
    cudaMemsetAsync(d_temp, 0, 2 * sizeof(float), stream);
    
    int blocks = (nfreq + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_compute_mean_lambda_sq<<<blocks, BLOCK_SIZE, 0, stream>>>(
        d_lambda_sq, d_weights, d_temp, nfreq
    );
    
    // Copy back and compute mean
    float h_temp[2];
    cudaMemcpyAsync(h_temp, d_temp, 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    float mean = (h_temp[1] > 0) ? h_temp[0] / h_temp[1] : 0.0f;
    cudaMemcpyAsync(d_mean_lambda_sq, &mean, sizeof(float), cudaMemcpyHostToDevice, stream);
    
    cudaFree(d_temp);
}

// ============================================================================
// NORMALIZATION KERNELS
// ============================================================================

__global__ void kernel_normalize_qu(
    const float* __restrict__ d_Q,
    const float* __restrict__ d_U,
    const float* __restrict__ d_I,
    float* __restrict__ d_Q_norm,
    float* __restrict__ d_U_norm,
    int nfreq,
    int npixels,
    float zero_threshold
) {
    int pix = blockIdx.x;
    if (pix >= npixels) return;
    
    for (int f = threadIdx.x; f < nfreq; f += blockDim.x) {
        int idx = pix * nfreq + f;
        float I_val = d_I[idx];
        float Q_val = d_Q[idx];
        float U_val = d_U[idx];
        
        // Safe division: zero out NaN/Inf or where |I| is too small
        if (isfinite(Q_val) && isfinite(U_val) && isfinite(I_val)
            && fabsf(I_val) > zero_threshold) {
            d_Q_norm[idx] = Q_val / I_val;
            d_U_norm[idx] = U_val / I_val;
        } else {
            d_Q_norm[idx] = 0.0f;
            d_U_norm[idx] = 0.0f;
        }
    }
}

void cu_normalize_qu(
    const float* d_Q,
    const float* d_U,
    const float* d_I,
    float* d_Q_norm,
    float* d_U_norm,
    int nfreq,
    int npixels,
    float zero_threshold,
    cudaStream_t stream
) {
    kernel_normalize_qu<<<npixels, BLOCK_SIZE, 0, stream>>>(
        d_Q, d_U, d_I, d_Q_norm, d_U_norm, nfreq, npixels, zero_threshold
    );
}

__global__ void kernel_copy_qu(
    const float* __restrict__ d_Q,
    const float* __restrict__ d_U,
    float* __restrict__ d_Q_norm,
    float* __restrict__ d_U_norm,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    float Q = d_Q[idx];
    float U = d_U[idx];
    d_Q_norm[idx] = isfinite(Q) ? Q : 0.0f;
    d_U_norm[idx] = isfinite(U) ? U : 0.0f;
}

void cu_copy_qu(
    const float* d_Q,
    const float* d_U,
    float* d_Q_norm,
    float* d_U_norm,
    int nfreq,
    int npixels,
    cudaStream_t stream
) {
    int total = nfreq * npixels;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_copy_qu<<<blocks, BLOCK_SIZE, 0, stream>>>(d_Q, d_U, d_Q_norm, d_U_norm, total);
}

// ============================================================================
// RMSF COMPUTATION
// ============================================================================

// RMSF: R(φ) = K * Σ_i w_i * exp(-2i φ (λ²_i - λ²_0))
// where K = 1 / Σ w_i  (normalization)
__global__ void kernel_compute_rmsf(
    const float* __restrict__ d_lambda_sq,
    const float* __restrict__ d_weights,
    const float* __restrict__ d_phi,
    float* __restrict__ d_rmsf_real,
    float* __restrict__ d_rmsf_imag,
    float mean_lambda_sq,
    int nfreq,
    int nphi
) {
    int phi_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (phi_idx >= nphi) return;
    
    float phi = d_phi[phi_idx];
    float sum_real = 0.0f;
    float sum_imag = 0.0f;
    float sum_weights = 0.0f;
    
    for (int f = 0; f < nfreq; f++) {
        float w = d_weights[f];
        float dl2 = d_lambda_sq[f] - mean_lambda_sq;
        float phase = -2.0f * phi * dl2;
        
        sum_real += w * cosf(phase);
        sum_imag += w * sinf(phase);
        sum_weights += w;
    }
    
    // Normalize
    float norm = (sum_weights > 1e-10f) ? 1.0f / sum_weights : 1.0f;
    d_rmsf_real[phi_idx] = sum_real * norm;
    d_rmsf_imag[phi_idx] = sum_imag * norm;
}

void cu_compute_rmsf(
    const float* d_lambda_sq,
    const float* d_weights,
    const float* d_phi,
    float* d_rmsf_real,
    float* d_rmsf_imag,
    float mean_lambda_sq,
    int nfreq,
    int nphi,
    cudaStream_t stream
) {
    int blocks = (nphi + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_compute_rmsf<<<blocks, BLOCK_SIZE, 0, stream>>>(
        d_lambda_sq, d_weights, d_phi, d_rmsf_real, d_rmsf_imag,
        mean_lambda_sq, nfreq, nphi
    );
}

float cu_compute_rmsf_fwhm(
    const float* d_rmsf_real,
    const float* d_rmsf_imag,
    const float* d_phi,
    int nphi
) {
    // Copy RMSF to host
    float* h_rmsf_real = new float[nphi];
    float* h_rmsf_imag = new float[nphi];
    float* h_phi = new float[nphi];
    
    cudaMemcpy(h_rmsf_real, d_rmsf_real, nphi * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rmsf_imag, d_rmsf_imag, nphi * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phi, d_phi, nphi * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Find peak (should be at center, φ=0)
    float max_amp = 0.0f;
    int peak_idx = nphi / 2;
    for (int i = 0; i < nphi; i++) {
        float amp = sqrtf(h_rmsf_real[i] * h_rmsf_real[i] + h_rmsf_imag[i] * h_rmsf_imag[i]);
        if (amp > max_amp) {
            max_amp = amp;
            peak_idx = i;
        }
    }
    
    float half_max = max_amp * 0.5f;
    
    // Find left crossing
    int left_idx = peak_idx;
    for (int i = peak_idx; i >= 0; i--) {
        float amp = sqrtf(h_rmsf_real[i] * h_rmsf_real[i] + h_rmsf_imag[i] * h_rmsf_imag[i]);
        if (amp < half_max) {
            left_idx = i;
            break;
        }
    }
    
    // Find right crossing
    int right_idx = peak_idx;
    for (int i = peak_idx; i < nphi; i++) {
        float amp = sqrtf(h_rmsf_real[i] * h_rmsf_real[i] + h_rmsf_imag[i] * h_rmsf_imag[i]);
        if (amp < half_max) {
            right_idx = i;
            break;
        }
    }
    
    float fwhm = h_phi[right_idx] - h_phi[left_idx];
    
    delete[] h_rmsf_real;
    delete[] h_rmsf_imag;
    delete[] h_phi;
    
    return fwhm;
}

// ============================================================================
// RM-SYNTHESIS VIA NUFFT (cuFINUFFT)
// ============================================================================
//
// RM-synthesis computes:
//   FDF(φ) = K * Σ_j w_j * P_j * exp(-2i * φ * (λ²_j - λ²_0))
//
// This maps to NUFFT Type-1 (nonuniform to uniform):
//   f[k] = Σ_j c_j * exp(+i * x_j * k)   for k = -N/2, ..., N/2-1
//
// Mapping:
//   - x_j = -2 * (λ²_j - λ²_0)  (nonuniform points in [-π, π))
//   - c_j = w_j * (Q_j + i*U_j) (complex strengths)
//   - f[k] = FDF(φ_k) where φ_k = k * dphi (uniform output grid)
//
// The NUFFT expects x_j in [-π, π), so we need to scale appropriately.
// If φ ranges from phi_min to phi_max with step dphi, and we want nphi points:
//   φ_k = phi_min + k * dphi, for k = 0, 1, ..., nphi-1
//
// We use iflag=+1 (positive exponential) to match our convention.

// Kernel to prepare NUFFT input: weighted complex polarization (single precision)
__global__ void kernel_prepare_nufft_input_f(
    const float* __restrict__ d_Q,
    const float* __restrict__ d_U,
    const float* __restrict__ d_weights,
    cuFloatComplex* __restrict__ d_nufft_c,
    int nfreq,
    int npixels
) {
    int pix = blockIdx.x;
    if (pix >= npixels) return;
    
    for (int f = threadIdx.x; f < nfreq; f += blockDim.x) {
        int idx = pix * nfreq + f;
        float w = d_weights[f];
        float Q = d_Q[idx];
        float U = d_U[idx];

        // Skip NaN/Inf channels — zero weight effectively removes them
        if (!isfinite(Q) || !isfinite(U)) {
            d_nufft_c[idx].x = 0.0f;
            d_nufft_c[idx].y = 0.0f;
        } else {
            d_nufft_c[idx].x = w * Q;
            d_nufft_c[idx].y = w * U;
        }
    }
}

// Kernel to compute NUFFT coordinates from lambda_sq (single precision)
// x_j must be in [-π, π) for cuFINUFFT
__global__ void kernel_compute_nufft_coords_f(
    const float* __restrict__ d_lambda_sq,
    float* __restrict__ d_nufft_x,
    float mean_lambda_sq,
    int nfreq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nfreq) return;
    
    // x_j = -2 * (λ²_j - λ²_0)
    // This needs to be scaled to fit in [-π, π)
    // The actual scaling factor depends on the output grid spacing
    // We'll handle the scaling in the plan setup
    
    float dl2 = d_lambda_sq[idx] - mean_lambda_sq;
    d_nufft_x[idx] = -2.0f * dl2;
}

// Kernel to convert NUFFT output and normalize
__global__ void kernel_finalize_fdf(
    const cuFloatComplex* __restrict__ d_nufft_f,
    float* __restrict__ d_fdf_real,
    float* __restrict__ d_fdf_imag,
    float norm,
    int nphi,
    int npixels
) {
    int pix = blockIdx.x;
    int phi_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (pix >= npixels || phi_idx >= nphi) return;
    
    int idx = pix * nphi + phi_idx;
    
    d_fdf_real[idx] = d_nufft_f[idx].x * norm;
    d_fdf_imag[idx] = d_nufft_f[idx].y * norm;
}

int cu_init_nufft_plan(
    GPUBuffers* buffers,
    float phi_min,
    float dphi,
    float mean_lambda_sq,
    cudaStream_t stream
) {
    int nfreq = buffers->nfreq;
    int nphi = buffers->nphi;
    // Cap NUFFT batch size to avoid blowing up cufinufft workspace memory.
    // The execute function will loop over sub-batches.
    int ntransf = buffers->batch_pixels;
    if (ntransf > NUFFT_MAX_BATCH) ntransf = NUFFT_MAX_BATCH;

    printf("  Setting up NUFFT: nfreq=%d, nphi=%d, ntransf=%d (batch_pixels=%d)\n",
           nfreq, nphi, ntransf, buffers->batch_pixels);
    
    // Ensure prior uploads (on default stream) are visible to this stream
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "  cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    // Debug: verify lambda_sq is on device
    {
        float h_test[4];
        err = cudaMemcpy(h_test, buffers->d_lambda_sq, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("  d_lambda_sq check: [%.6f, %.6f, %.6f, %.6f] (err=%d)\n",
               h_test[0], h_test[1], h_test[2], h_test[3], (int)err);
        printf("  mean_lambda_sq=%.6f\n", mean_lambda_sq);
    }

    // Compute NUFFT coordinates on GPU
    // x_j = -2 * (λ²_j - λ²_0)
    int blocks = (nfreq + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_compute_nufft_coords_f<<<blocks, BLOCK_SIZE, 0, stream>>>(
        buffers->d_lambda_sq, buffers->d_nufft_x, mean_lambda_sq, nfreq
    );
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "  kernel_compute_nufft_coords_f failed: %s\n", cudaGetErrorString(err));
    }

    // Copy coordinates to host to check range and scale
    float* h_x = new float[nfreq];
    cudaMemcpy(h_x, buffers->d_nufft_x, nfreq * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Find min/max to check if we need scaling
    float x_min = h_x[0], x_max = h_x[0];
    for (int i = 1; i < nfreq; i++) {
        if (h_x[i] < x_min) x_min = h_x[i];
        if (h_x[i] > x_max) x_max = h_x[i];
    }
    printf("  NUFFT x range: [%.4f, %.4f]\n", x_min, x_max);
    
    // cuFINUFFT expects x in [-π, π)
    // Our φ grid goes from phi_min to phi_max with spacing dphi
    // The NUFFT output index k corresponds to φ_k = phi_min + k * dphi
    // 
    // For Type-1 NUFFT: f[k] = Σ c_j exp(i x_j k), k = -N/2..N/2-1
    // We want: FDF(φ_k) = Σ c_j exp(-2i φ_k Δλ²_j)
    //                   = Σ c_j exp(-2i (phi_min + k*dphi) Δλ²_j)
    //                   = Σ c_j exp(-2i phi_min Δλ²_j) * exp(-2i k dphi Δλ²_j)
    //
    // So we should set: x_j = -2 * dphi * Δλ²_j
    // And apply the phi_min phase shift separately to c_j
    
    // Recompute with proper scaling
    for (int i = 0; i < nfreq; i++) {
        h_x[i] = h_x[i] * dphi;  // Now x_j = -2 * dphi * Δλ²_j
    }
    cudaMemcpy(buffers->d_nufft_x, h_x, nfreq * sizeof(float), cudaMemcpyHostToDevice);
    
    // Check new range
    x_min = h_x[0]; x_max = h_x[0];
    for (int i = 1; i < nfreq; i++) {
        if (h_x[i] < x_min) x_min = h_x[i];
        if (h_x[i] > x_max) x_max = h_x[i];
    }
    printf("  NUFFT x (scaled) range: [%.6f, %.6f]\n", x_min, x_max);
    
    // Warn if outside [-π, π)
    if (x_min < -M_PI || x_max >= M_PI) {
        printf("  WARNING: x coordinates outside [-π, π), NUFFT will wrap\n");
    }
    
    delete[] h_x;
    
    // Create NUFFT plan (single precision)
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.gpu_stream = stream;
    
    // Output grid size: nphi modes
    int64_t nmodes[3] = {nphi, 1, 1};
    
    // Using single precision: cufinufftf_makeplan
    int ier = cufinufftf_makeplan(
        1,              // type 1: nonuniform to uniform  
        1,              // 1D
        nmodes,         // output grid dimensions [nphi]
        +1,             // iflag: +1 for exp(+i x k)
        ntransf,        // number of simultaneous transforms
        1e-5f,          // tolerance (single precision)
        &buffers->nufft_plan,
        &opts
    );
    
    if (ier != 0) {
        fprintf(stderr, "cufinufftf_makeplan failed with error %d\n", ier);
        return ier;
    }
    
    // Set the nonuniform points (shared across all transforms in batch)
    ier = cufinufftf_setpts(
        buffers->nufft_plan,
        nfreq,              // M: number of nonuniform points
        buffers->d_nufft_x, // x coordinates [nfreq]
        nullptr,            // y (not used for 1D)
        nullptr,            // z (not used for 1D)
        0, nullptr, nullptr, nullptr  // type-3 args (not used)
    );
    
    if (ier != 0) {
        fprintf(stderr, "cufinufftf_setpts failed with error %d\n", ier);
        cufinufftf_destroy(buffers->nufft_plan);
        buffers->nufft_plan = nullptr;
        return ier;
    }
    
    buffers->nufft_ntransf = ntransf;
    printf("  NUFFT plan created successfully\n");
    return 0;
}

void cu_destroy_nufft_plan(GPUBuffers* buffers) {
    if (buffers->nufft_plan) {
        cufinufftf_destroy(buffers->nufft_plan);
        buffers->nufft_plan = nullptr;
    }
}

void cu_rm_synthesis_nufft(
    GPUBuffers* buffers,
    const float* d_Q,
    const float* d_U,
    const float* d_weights,
    float* d_fdf_real,
    float* d_fdf_imag,
    int npixels,
    cudaStream_t stream
) {
    int nfreq = buffers->nfreq;
    int nphi = buffers->nphi;
    int ntransf = buffers->nufft_ntransf;  // NUFFT plan batch size

    // Compute sum of weights for normalization (once)
    float* h_w = new float[nfreq];
    cudaMemcpy(h_w, d_weights, nfreq * sizeof(float), cudaMemcpyDeviceToHost);
    float sum_weights = 0.0f;
    for (int i = 0; i < nfreq; i++) sum_weights += h_w[i];
    delete[] h_w;
    float norm = (sum_weights > 1e-10f) ? 1.0f / sum_weights : 1.0f;

    // Process in sub-batches of ntransf pixels
    for (int offset = 0; offset < npixels; offset += ntransf) {
        int count = npixels - offset;
        if (count > ntransf) count = ntransf;

        // Zero NUFFT input (handles last sub-batch padding)
        cudaMemsetAsync(buffers->d_nufft_c, 0,
                        ntransf * nfreq * sizeof(cuFloatComplex), stream);

        // Prepare weighted complex input for this sub-batch
        kernel_prepare_nufft_input_f<<<count, BLOCK_SIZE, 0, stream>>>(
            d_Q + offset * nfreq, d_U + offset * nfreq, d_weights,
            buffers->d_nufft_c, nfreq, count
        );

        // Execute NUFFT
        int ier = cufinufftf_execute(
            buffers->nufft_plan,
            buffers->d_nufft_c,
            buffers->d_nufft_f
        );

        if (ier != 0) {
            fprintf(stderr, "cufinufftf_execute failed with error %d at offset %d\n", ier, offset);
            return;
        }

        // Convert cuFloatComplex to separate real/imag and normalize
        dim3 blocks(count, (nphi + BLOCK_SIZE - 1) / BLOCK_SIZE);
        kernel_finalize_fdf<<<blocks, BLOCK_SIZE, 0, stream>>>(
            buffers->d_nufft_f,
            d_fdf_real + offset * nphi,
            d_fdf_imag + offset * nphi,
            norm, nphi, count
        );
    }
}

// ============================================================================
// PEAK FINDING WITH PARABOLIC INTERPOLATION
// ============================================================================

__device__ void parabolic_interp(
    float y_prev, float y_peak, float y_next,
    float x_prev, float x_peak, float x_next,
    float* x_refined, float* y_refined
) {
    // Fit parabola through three points and find vertex
    // y = a*x² + b*x + c
    // Using Lagrange interpolation form for numerical stability
    
    float d1 = y_prev - y_peak;
    float d2 = y_next - y_peak;
    float dx = x_next - x_peak;  // Assuming uniform spacing
    
    // Vertex offset from center point
    float offset = 0.5f * dx * (d1 - d2) / (d1 + d2 + 1e-10f);
    
    // Clamp to avoid extrapolating too far
    offset = fmaxf(-dx, fminf(dx, offset));
    
    *x_refined = x_peak + offset;
    
    // Interpolated peak value
    float t = offset / dx;
    *y_refined = y_peak - 0.25f * (d1 + d2) * t * t + 0.5f * (d2 - d1) * t;
}

__global__ void kernel_find_fdf_peak(
    const float* __restrict__ d_fdf_real,
    const float* __restrict__ d_fdf_imag,
    const float* __restrict__ d_phi,
    float* __restrict__ d_peak_rm,
    float* __restrict__ d_peak_pi,
    float* __restrict__ d_rm_err,
    float rmsf_fwhm,
    int nphi,
    int npixels
) {
    int pix = blockIdx.x;
    if (pix >= npixels) return;
    
    // Shared memory for parallel reduction
    __shared__ float s_max_amp[BLOCK_SIZE];
    __shared__ int s_max_idx[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    
    // Find maximum amplitude across all φ channels
    float local_max = 0.0f;
    int local_idx = 0;
    
    for (int i = tid; i < nphi; i += blockDim.x) {
        int idx = pix * nphi + i;
        float re = d_fdf_real[idx];
        float im = d_fdf_imag[idx];
        float amp = sqrtf(re * re + im * im);
        
        if (amp > local_max) {
            local_max = amp;
            local_idx = i;
        }
    }
    
    s_max_amp[tid] = local_max;
    s_max_idx[tid] = local_idx;
    __syncthreads();
    
    // Parallel reduction to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max_amp[tid + s] > s_max_amp[tid]) {
                s_max_amp[tid] = s_max_amp[tid + s];
                s_max_idx[tid] = s_max_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 does the final interpolation and writes output
    if (tid == 0) {
        int peak_idx = s_max_idx[0];
        float peak_amp = s_max_amp[0];
        
        // Parabolic interpolation for sub-channel precision
        float refined_phi = d_phi[peak_idx];
        float refined_amp = peak_amp;
        
        if (peak_idx > 0 && peak_idx < nphi - 1) {
            int idx_prev = pix * nphi + peak_idx - 1;
            int idx_peak = pix * nphi + peak_idx;
            int idx_next = pix * nphi + peak_idx + 1;
            
            float amp_prev = sqrtf(d_fdf_real[idx_prev] * d_fdf_real[idx_prev] + 
                                   d_fdf_imag[idx_prev] * d_fdf_imag[idx_prev]);
            float amp_next = sqrtf(d_fdf_real[idx_next] * d_fdf_real[idx_next] + 
                                   d_fdf_imag[idx_next] * d_fdf_imag[idx_next]);
            
            parabolic_interp(
                amp_prev, peak_amp, amp_next,
                d_phi[peak_idx - 1], d_phi[peak_idx], d_phi[peak_idx + 1],
                &refined_phi, &refined_amp
            );
        }
        
        d_peak_rm[pix] = refined_phi;
        d_peak_pi[pix] = refined_amp;
        
        // RM uncertainty: σ_RM ≈ FWHM / (2 * SNR)
        // Approximate SNR from peak amplitude (assumes noise ~ 1/sqrt(nfreq))
        // More sophisticated error would require noise estimation
        float snr = refined_amp / (1e-6f + 0.01f);  // Placeholder, should use actual noise
        d_rm_err[pix] = rmsf_fwhm / (2.0f * fmaxf(snr, 1.0f));
    }
}

void cu_find_fdf_peak(
    const float* d_fdf_real,
    const float* d_fdf_imag,
    const float* d_phi,
    float* d_peak_rm,
    float* d_peak_pi,
    float* d_rm_err,
    float rmsf_fwhm,
    int nphi,
    int npixels,
    cudaStream_t stream
) {
    kernel_find_fdf_peak<<<npixels, BLOCK_SIZE, 0, stream>>>(
        d_fdf_real, d_fdf_imag, d_phi, d_peak_rm, d_peak_pi, d_rm_err,
        rmsf_fwhm, nphi, npixels
    );
}

// ============================================================================
// RM-CLEAN - PROPER IMPLEMENTATION
// ============================================================================

// RM-CLEAN deconvolution algorithm:
// 1. Initialize: residual = dirty FDF, model = 0
// 2. Loop:
//    a. Find peak in |residual|
//    b. If peak < threshold, stop
//    c. Subtract: residual -= gain * peak_value * RMSF(φ - φ_peak)
//    d. Add to model: model[φ_peak] += gain * peak_value
// 3. Convolve model with clean beam (Gaussian with FWHM of RMSF main lobe)
// 4. Add residuals to get final cleaned FDF

__device__ void find_peak_in_spectrum(
    const float* __restrict__ fdf_real,
    const float* __restrict__ fdf_imag,
    int nphi,
    int* peak_idx,
    float* peak_amp,
    float* peak_real,
    float* peak_imag
) {
    float max_amp = 0.0f;
    int max_idx = 0;
    
    for (int i = 0; i < nphi; i++) {
        float re = fdf_real[i];
        float im = fdf_imag[i];
        float amp = sqrtf(re * re + im * im);
        if (amp > max_amp) {
            max_amp = amp;
            max_idx = i;
        }
    }
    
    *peak_idx = max_idx;
    *peak_amp = max_amp;
    *peak_real = fdf_real[max_idx];
    *peak_imag = fdf_imag[max_idx];
}

__global__ void kernel_rm_clean(
    const float* __restrict__ d_fdf_real,
    const float* __restrict__ d_fdf_imag,
    const float* __restrict__ d_rmsf_real,
    const float* __restrict__ d_rmsf_imag,
    const float* __restrict__ d_phi,
    float* __restrict__ d_clean_fdf_real,
    float* __restrict__ d_clean_fdf_imag,
    float* __restrict__ d_residual_real,
    float* __restrict__ d_residual_imag,
    float* __restrict__ d_model_real,
    float* __restrict__ d_model_imag,
    float* __restrict__ d_clean_peak_rm,
    float* __restrict__ d_clean_peak_pi,
    float gain,
    float threshold,
    float threshold_rel,
    bool use_rel_threshold,
    float rmsf_fwhm,
    float dphi,
    int nphi,
    int npixels,
    int max_iter
) {
    int pix = blockIdx.x;
    if (pix >= npixels) return;
    
    // Only thread 0 does the work (RM-CLEAN is inherently serial per pixel)
    if (threadIdx.x != 0) return;
    
    int offset = pix * nphi;
    float* residual_real = d_residual_real + offset;
    float* residual_imag = d_residual_imag + offset;
    float* model_real = d_model_real + offset;
    float* model_imag = d_model_imag + offset;
    float* clean_real = d_clean_fdf_real + offset;
    float* clean_imag = d_clean_fdf_imag + offset;
    
    // Initialize: residual = dirty FDF, model = 0
    for (int i = 0; i < nphi; i++) {
        residual_real[i] = d_fdf_real[offset + i];
        residual_imag[i] = d_fdf_imag[offset + i];
        model_real[i] = 0.0f;
        model_imag[i] = 0.0f;
    }
    
    // Find initial peak for relative threshold
    int peak_idx;
    float peak_amp, peak_real, peak_imag;
    find_peak_in_spectrum(residual_real, residual_imag, nphi, 
                          &peak_idx, &peak_amp, &peak_real, &peak_imag);
    
    float abs_threshold = use_rel_threshold ? threshold_rel * peak_amp : threshold;
    float initial_peak_amp = peak_amp;
    
    // CLEAN loop
    int center_phi = nphi / 2;  // RMSF is centered
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Find current peak in residual
        find_peak_in_spectrum(residual_real, residual_imag, nphi,
                              &peak_idx, &peak_amp, &peak_real, &peak_imag);
        
        // Check stopping criterion
        if (peak_amp < abs_threshold) break;
        
        // Add scaled peak to model
        float component_real = gain * peak_real;
        float component_imag = gain * peak_imag;
        model_real[peak_idx] += component_real;
        model_imag[peak_idx] += component_imag;
        
        // Subtract shifted RMSF from residual
        // RMSF centered at 0, we need it centered at peak_idx
        int shift = peak_idx - center_phi;
        
        for (int i = 0; i < nphi; i++) {
            int rmsf_idx = i - shift;  // Index into RMSF
            
            // Handle boundaries (RMSF is periodic in principle, but we clip)
            if (rmsf_idx < 0 || rmsf_idx >= nphi) continue;
            
            float rmsf_re = d_rmsf_real[rmsf_idx];
            float rmsf_im = d_rmsf_imag[rmsf_idx];
            
            // Subtract: component * RMSF (complex multiplication)
            // (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
            residual_real[i] -= component_real * rmsf_re - component_imag * rmsf_im;
            residual_imag[i] -= component_real * rmsf_im + component_imag * rmsf_re;
        }
    }
    
    // Convolve model with restoring beam (Gaussian)
    // σ = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.355
    float sigma = rmsf_fwhm / 2.355f;
    float sigma_sq = sigma * sigma;
    
    // Clean beam convolution
    for (int i = 0; i < nphi; i++) {
        float conv_real = 0.0f;
        float conv_imag = 0.0f;
        
        for (int j = 0; j < nphi; j++) {
            if (model_real[j] == 0.0f && model_imag[j] == 0.0f) continue;
            
            float d_phi = (i - j) * dphi;
            float gauss = expf(-0.5f * d_phi * d_phi / sigma_sq);
            
            conv_real += model_real[j] * gauss;
            conv_imag += model_imag[j] * gauss;
        }
        
        // Final cleaned FDF = convolved model + residuals
        clean_real[i] = conv_real + residual_real[i];
        clean_imag[i] = conv_imag + residual_imag[i];
    }
    
    // Find peak in cleaned FDF
    find_peak_in_spectrum(clean_real, clean_imag, nphi,
                          &peak_idx, &peak_amp, &peak_real, &peak_imag);
    
    d_clean_peak_rm[pix] = d_phi[peak_idx];
    d_clean_peak_pi[pix] = peak_amp;
}

void cu_rm_clean(
    const float* d_fdf_real,
    const float* d_fdf_imag,
    const float* d_rmsf_real,
    const float* d_rmsf_imag,
    const float* d_phi,
    float* d_clean_fdf_real,
    float* d_clean_fdf_imag,
    float* d_residual_real,
    float* d_residual_imag,
    float* d_model_real,
    float* d_model_imag,
    float* d_clean_peak_rm,
    float* d_clean_peak_pi,
    const CleanConfig& config,
    float rmsf_fwhm,
    int nphi,
    int npixels,
    cudaStream_t stream
) {
    // Get dphi from phi grid (assume uniform)
    float h_phi[2];
    cudaMemcpy(h_phi, d_phi, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    float dphi = h_phi[1] - h_phi[0];
    
    // One block per pixel, single thread does the work
    // (RM-CLEAN is inherently serial, parallelism is across pixels)
    kernel_rm_clean<<<npixels, 32, 0, stream>>>(
        d_fdf_real, d_fdf_imag, d_rmsf_real, d_rmsf_imag, d_phi,
        d_clean_fdf_real, d_clean_fdf_imag,
        d_residual_real, d_residual_imag,
        d_model_real, d_model_imag,
        d_clean_peak_rm, d_clean_peak_pi,
        config.gain, config.threshold, config.threshold_rel,
        config.use_threshold_rel, rmsf_fwhm, dphi,
        nphi, npixels, config.max_iter
    );
}

// Simplified version - just get cleaned peaks without full cube
void cu_rm_clean_peaks_only(
    const float* d_fdf_real,
    const float* d_fdf_imag,
    const float* d_rmsf_real,
    const float* d_rmsf_imag,
    const float* d_phi,
    float* d_workspace,
    float* d_clean_peak_rm,
    float* d_clean_peak_pi,
    const CleanConfig& config,
    int nphi,
    int npixels,
    cudaStream_t stream
) {
    // Workspace layout: [residual_real | residual_imag | model_real | model_imag]
    // Each is npixels * nphi
    size_t chunk = npixels * nphi;
    float* d_residual_real = d_workspace;
    float* d_residual_imag = d_workspace + chunk;
    float* d_model_real = d_workspace + 2 * chunk;
    float* d_model_imag = d_workspace + 3 * chunk;
    
    // We don't need the full cleaned FDF, just use model buffer for output
    float* d_clean_fdf_real = d_model_real;  // Reuse
    float* d_clean_fdf_imag = d_model_imag;
    
    float h_phi[2];
    cudaMemcpy(h_phi, d_phi, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    float dphi = h_phi[1] - h_phi[0];
    
    // Need RMSF FWHM - compute it
    float rmsf_fwhm = cu_compute_rmsf_fwhm(d_rmsf_real, d_rmsf_imag, d_phi, nphi);
    
    kernel_rm_clean<<<npixels, 32, 0, stream>>>(
        d_fdf_real, d_fdf_imag, d_rmsf_real, d_rmsf_imag, d_phi,
        d_clean_fdf_real, d_clean_fdf_imag,
        d_residual_real, d_residual_imag,
        d_model_real, d_model_imag,
        d_clean_peak_rm, d_clean_peak_pi,
        config.gain, config.threshold, config.threshold_rel,
        config.use_threshold_rel, rmsf_fwhm, dphi,
        nphi, npixels, config.max_iter
    );
}
