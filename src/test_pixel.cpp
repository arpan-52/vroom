// Single-pixel RM-synthesis test
// Loads one pixel from Q/U cubes, runs RM synthesis, prints the FDF
#include "vroom.h"
#include "cuda_kernels.h"
#include "io.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s Q.fits U.fits freq.txt pixel_index [phi_min phi_max dphi]\n", argv[0]);
        printf("  pixel_index: flat index (y * nx + x), or use x,y format\n");
        return 1;
    }

    const char* q_file = argv[1];
    const char* u_file = argv[2];
    const char* freq_file = argv[3];
    int pixel_idx = atoi(argv[4]);

    float phi_min = -500.0f, phi_max = 500.0f, dphi = 1.0f;
    if (argc > 5) phi_min = atof(argv[5]);
    if (argc > 6) phi_max = atof(argv[6]);
    if (argc > 7) dphi = atof(argv[7]);

    // Read frequencies
    float* h_freq = nullptr;
    int nfreq;
    if (read_frequencies(freq_file, &h_freq, &nfreq) != 0) {
        fprintf(stderr, "Failed to read frequencies\n");
        return 1;
    }
    printf("Frequencies: %d channels (%.1f - %.1f MHz)\n",
           nfreq, h_freq[0] / 1e6, h_freq[nfreq - 1] / 1e6);

    // Get cube dimensions
    CubeInfo cube;
    float* dummy = nullptr;
    if (read_fits_cube(q_file, &dummy, &cube) != 0) {
        fprintf(stderr, "Failed to read Q cube\n");
        return 1;
    }
    free(dummy);
    printf("Cube: %d x %d x %d\n", cube.nx, cube.ny, cube.nfreq);

    int px = pixel_idx % cube.nx;
    int py = pixel_idx / cube.nx;
    printf("Testing pixel %d (x=%d, y=%d)\n\n", pixel_idx, px, py);

    // Read one pixel's worth of Q and U
    float* h_Q = (float*)malloc(nfreq * sizeof(float));
    float* h_U = (float*)malloc(nfreq * sizeof(float));

    if (read_fits_cube_batch(q_file, h_Q, cube, pixel_idx, 1) != 0) {
        fprintf(stderr, "Failed to read Q pixel\n");
        return 1;
    }
    if (read_fits_cube_batch(u_file, h_U, cube, pixel_idx, 1) != 0) {
        fprintf(stderr, "Failed to read U pixel\n");
        return 1;
    }

    // Print input spectrum, count NaNs
    int nan_count = 0;
    printf("--- Input spectrum ---\n");
    printf("%5s %12s %12s %12s\n", "Chan", "Freq(MHz)", "Q", "U");
    for (int i = 0; i < nfreq; i++) {
        bool bad = !std::isfinite(h_Q[i]) || !std::isfinite(h_U[i]);
        if (bad) nan_count++;
        printf("%5d %12.3f %12.6f %12.6f%s\n",
               i, h_freq[i] / 1e6, h_Q[i], h_U[i], bad ? " *** NaN" : "");
    }
    printf("\nNaN channels: %d / %d\n\n", nan_count, nfreq);

    // Compute lambda squared
    const float c = 299792458.0f;
    float* h_lambda_sq = (float*)malloc(nfreq * sizeof(float));
    float* h_weights = (float*)malloc(nfreq * sizeof(float));
    double sum_wl2 = 0, sum_w = 0;
    for (int i = 0; i < nfreq; i++) {
        float lam = c / h_freq[i];
        h_lambda_sq[i] = lam * lam;
        h_weights[i] = 1.0f;
        // Only count valid channels for mean
        if (std::isfinite(h_Q[i]) && std::isfinite(h_U[i])) {
            sum_wl2 += h_lambda_sq[i];
            sum_w += 1.0;
        }
    }
    float mean_l2 = (float)(sum_wl2 / sum_w);
    printf("Lambda^2 range: %.6f - %.6f m^2 (mean: %.6f)\n",
           h_lambda_sq[nfreq - 1], h_lambda_sq[0], mean_l2);

    // Faraday grid
    int nphi = (int)((phi_max - phi_min) / dphi) + 1;
    float* h_phi = (float*)malloc(nphi * sizeof(float));
    for (int i = 0; i < nphi; i++) h_phi[i] = phi_min + i * dphi;
    printf("Faraday grid: %.1f to %.1f, dphi=%.1f, nphi=%d\n\n", phi_min, phi_max, dphi, nphi);

    // ============================================================
    // CPU DFT reference (brute force, NaN-safe)
    // ============================================================
    printf("--- CPU DFT reference ---\n");
    float* cpu_fdf_real = (float*)calloc(nphi, sizeof(float));
    float* cpu_fdf_imag = (float*)calloc(nphi, sizeof(float));
    float cpu_sum_w = 0;
    for (int i = 0; i < nfreq; i++) {
        if (!std::isfinite(h_Q[i]) || !std::isfinite(h_U[i])) continue;
        cpu_sum_w += h_weights[i];
    }
    float cpu_norm = (cpu_sum_w > 0) ? 1.0f / cpu_sum_w : 1.0f;

    for (int p = 0; p < nphi; p++) {
        float re = 0, im = 0;
        for (int f = 0; f < nfreq; f++) {
            if (!std::isfinite(h_Q[f]) || !std::isfinite(h_U[f])) continue;
            float w = h_weights[f];
            float dl2 = h_lambda_sq[f] - mean_l2;
            float phase = -2.0f * h_phi[p] * dl2;
            float cos_p = cosf(phase);
            float sin_p = sinf(phase);
            // FDF = sum w * (Q + iU) * exp(-2i phi dl2)
            //     = sum w * [(Q cos - U sin) + i(Q sin + U cos)]
            re += w * (h_Q[f] * cos_p - h_U[f] * sin_p);
            im += w * (h_Q[f] * sin_p + h_U[f] * cos_p);
        }
        cpu_fdf_real[p] = re * cpu_norm;
        cpu_fdf_imag[p] = im * cpu_norm;
    }

    // Find CPU peak
    float cpu_peak_amp = 0;
    int cpu_peak_idx = 0;
    for (int p = 0; p < nphi; p++) {
        float amp = sqrtf(cpu_fdf_real[p] * cpu_fdf_real[p] +
                          cpu_fdf_imag[p] * cpu_fdf_imag[p]);
        if (amp > cpu_peak_amp) {
            cpu_peak_amp = amp;
            cpu_peak_idx = p;
        }
    }
    printf("CPU peak: phi=%.2f rad/m^2, |FDF|=%.6f\n\n",
           h_phi[cpu_peak_idx], cpu_peak_amp);

    // ============================================================
    // GPU NUFFT (using only valid channels, like the pipeline does)
    // ============================================================
    printf("--- GPU NUFFT ---\n");
    int npixels = 1;

    // Compact to valid channels only
    int nfreq_valid = nfreq - nan_count;
    float* v_Q = (float*)malloc(nfreq_valid * sizeof(float));
    float* v_U = (float*)malloc(nfreq_valid * sizeof(float));
    float* v_freq = (float*)malloc(nfreq_valid * sizeof(float));
    float* v_lambda_sq = (float*)malloc(nfreq_valid * sizeof(float));
    float* v_weights = (float*)malloc(nfreq_valid * sizeof(float));
    {
        int vi = 0;
        double vsum_wl2 = 0, vsum_w = 0;
        for (int i = 0; i < nfreq; i++) {
            if (!std::isfinite(h_Q[i]) || !std::isfinite(h_U[i])) continue;
            v_Q[vi] = h_Q[i];
            v_U[vi] = h_U[i];
            v_freq[vi] = h_freq[i];
            v_lambda_sq[vi] = h_lambda_sq[i];
            v_weights[vi] = h_weights[i];
            vsum_wl2 += v_weights[vi] * v_lambda_sq[vi];
            vsum_w += v_weights[vi];
            vi++;
        }
        mean_l2 = (float)(vsum_wl2 / vsum_w);
    }
    printf("Valid channels for GPU: %d (mean_l2=%.6f)\n", nfreq_valid, mean_l2);

    GPUBuffers* buf = allocate_gpu_buffers(nfreq_valid, nphi, npixels, false, false);

    // Upload compacted constants
    cudaMemcpy(buf->d_freq, v_freq, nfreq_valid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buf->d_lambda_sq, v_lambda_sq, nfreq_valid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buf->d_weights, v_weights, nfreq_valid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buf->d_phi, h_phi, nphi * sizeof(float), cudaMemcpyHostToDevice);

    // Init NUFFT plan
    int ier = cu_init_nufft_plan(buf, phi_min, dphi, mean_l2, 0);
    if (ier != 0) {
        fprintf(stderr, "NUFFT plan failed: %d\n", ier);
        return 1;
    }

    // Upload compacted Q/U
    cudaMemcpy(buf->d_Q, v_Q, nfreq_valid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buf->d_U, v_U, nfreq_valid * sizeof(float), cudaMemcpyHostToDevice);

    // Copy Q/U to normalized (no model I)
    cu_copy_qu(buf->d_Q, buf->d_U, buf->d_Q_norm, buf->d_U_norm, nfreq_valid, npixels, 0);

    // Run RM synthesis
    cu_rm_synthesis_nufft(buf, buf->d_Q_norm, buf->d_U_norm, buf->d_weights,
                          buf->d_fdf_real, buf->d_fdf_imag, npixels, 0);
    cudaDeviceSynchronize();

    // Download FDF
    float* gpu_fdf_real = (float*)malloc(nphi * sizeof(float));
    float* gpu_fdf_imag = (float*)malloc(nphi * sizeof(float));
    cudaMemcpy(gpu_fdf_real, buf->d_fdf_real, nphi * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_fdf_imag, buf->d_fdf_imag, nphi * sizeof(float), cudaMemcpyDeviceToHost);

    // Find GPU peak
    float gpu_peak_amp = 0;
    int gpu_peak_idx = 0;
    int gpu_nan_count = 0;
    for (int p = 0; p < nphi; p++) {
        if (!std::isfinite(gpu_fdf_real[p]) || !std::isfinite(gpu_fdf_imag[p])) {
            gpu_nan_count++;
            continue;
        }
        float amp = sqrtf(gpu_fdf_real[p] * gpu_fdf_real[p] +
                          gpu_fdf_imag[p] * gpu_fdf_imag[p]);
        if (amp > gpu_peak_amp) {
            gpu_peak_amp = amp;
            gpu_peak_idx = p;
        }
    }
    printf("GPU peak: phi=%.2f rad/m^2, |FDF|=%.6f\n", h_phi[gpu_peak_idx], gpu_peak_amp);
    if (gpu_nan_count > 0) printf("WARNING: %d NaN values in GPU FDF!\n", gpu_nan_count);

    // Compare around peaks
    printf("\n--- FDF comparison around CPU peak (phi=%.1f) ---\n", h_phi[cpu_peak_idx]);
    printf("%8s %12s %12s %12s %12s\n", "phi", "CPU_amp", "GPU_amp", "CPU_re", "GPU_re");
    int start = (cpu_peak_idx - 5 > 0) ? cpu_peak_idx - 5 : 0;
    int end = (cpu_peak_idx + 5 < nphi) ? cpu_peak_idx + 5 : nphi - 1;
    for (int p = start; p <= end; p++) {
        float cpu_amp = sqrtf(cpu_fdf_real[p] * cpu_fdf_real[p] +
                              cpu_fdf_imag[p] * cpu_fdf_imag[p]);
        float gpu_amp = sqrtf(gpu_fdf_real[p] * gpu_fdf_real[p] +
                              gpu_fdf_imag[p] * gpu_fdf_imag[p]);
        printf("%8.1f %12.6f %12.6f %12.6f %12.6f\n",
               h_phi[p], cpu_amp, gpu_amp, cpu_fdf_real[p], gpu_fdf_real[p]);
    }

    // Cleanup
    free_gpu_buffers(buf);
    free(h_Q); free(h_U); free(h_freq); free(h_lambda_sq); free(h_weights); free(h_phi);
    free(v_Q); free(v_U); free(v_freq); free(v_lambda_sq); free(v_weights);
    free(cpu_fdf_real); free(cpu_fdf_imag); free(gpu_fdf_real); free(gpu_fdf_imag);

    return 0;
}
