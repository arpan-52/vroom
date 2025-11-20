#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fitsio.h>
#include <cuda_runtime.h>
#include "cuda_rm.h"

typedef struct {
    int NFREQ;
    int NY, NX;
    float *freq;
    float *Q, *U, *I;
    int *mask;
} DataCube;

typedef struct {
    float *rm_map;
    float *rm_clean_map;
    float *rm_err_map;
    float *peak_rm_map;
    float2 *rmsf_cube;
} OutputMaps;

typedef struct {
    float f_min;
    float f_max;
    float f_res;
} FaradayParams;

typedef struct {
    int chunk_size;
    float clean_gain;
    float clean_threshold;
    int clean_max_iter;
} ProcessParams;

// ============================================================================
// GPU MEMORY DETECTION
// ============================================================================

size_t get_available_vram() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

void print_vram_info() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory: %.1f GB free / %.1f GB total\n", 
           free_mem / (1024.0f * 1024.0f * 1024.0f),
           total_mem / (1024.0f * 1024.0f * 1024.0f));
}

// ============================================================================
// AUTO-DETECT CHUNK SIZE BASED ON VRAM AND CUBE SIZE
// ============================================================================

int auto_detect_chunk_size(int nfreq, int ny, int nx, size_t available_vram) {
    // Memory per pixel per channel (float)
    size_t bytes_per_pixel_per_freq = 4;  // float32
    
    // Estimate memory for one chunk:
    // - Q, U, I input: 3 * chunk_size^2 * nfreq * 4 bytes
    // - Intermediate: Q_norm, U_norm, I_fit: 3 * chunk_size^2 * nfreq * 4 bytes
    // - RMSF output: chunk_size^2 * nfara * 8 bytes (float2)
    // - Peak, RM, Err, Clean: 4 * chunk_size^2 * 4 bytes
    // Total rough estimate: ~10 * chunk_size^2 * nfreq * 4 bytes
    
    // Reserve 20% of VRAM for safety
    size_t usable_vram = (available_vram * 80) / 100;
    
    // Estimate: 10 * chunk^2 * nfreq * 4 < usable_vram
    // chunk^2 < usable_vram / (10 * nfreq * 4)
    float chunk_float = sqrtf((float)usable_vram / (10.0f * nfreq * 4.0f));
    int chunk_size = (int)chunk_float;
    
    // Clamp to reasonable values
    if (chunk_size < 64) chunk_size = 64;
    if (chunk_size > 1024) chunk_size = 1024;
    
    // Round to nearest power of 2 for efficiency
    int powers[] = {64, 128, 256, 512, 1024};
    int best = 64;
    for (int i = 0; i < 5; i++) {
        if (powers[i] <= chunk_size) {
            best = powers[i];
        }
    }
    
    printf("Auto-detected chunk size: %d×%d (based on %.1f GB available VRAM)\n",
           best, best, usable_vram / (1024.0f * 1024.0f * 1024.0f));
    
    return best;
}

// ============================================================================
// FITS I/O
// ============================================================================

int read_fits_cube(const char *filename, float **data, int *nfreq, int *ny, int *nx) {
    fitsfile *fptr = NULL;
    int status = 0, naxis;
    long naxes[3];
    
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fprintf(stderr, "Error opening FITS file: %s\n", filename);
        return status;
    }
    
    fits_get_img_dim(fptr, &naxis, &status);
    fits_get_img_size(fptr, 3, naxes, &status);
    
    *nfreq = naxes[0];
    *nx = naxes[1];
    *ny = naxes[2];
    
    long nelements = (*nfreq) * (*nx) * (*ny);
    *data = (float *)malloc(nelements * sizeof(float));
    
    long fpixel = 1;
    fits_read_img(fptr, TFLOAT, fpixel, nelements, NULL, *data, NULL, &status);
    fits_close_file(fptr, &status);
    
    return status;
}

int read_fits_image(const char *filename, int **data, int *ny, int *nx) {
    fitsfile *fptr = NULL;
    int status = 0, naxis;
    long naxes[2];
    
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) return status;
    
    fits_get_img_dim(fptr, &naxis, &status);
    fits_get_img_size(fptr, 2, naxes, &status);
    
    *nx = naxes[0];
    *ny = naxes[1];
    
    long nelements = (*nx) * (*ny);
    *data = (int *)malloc(nelements * sizeof(int));
    
    long fpixel = 1;
    fits_read_img(fptr, TINT, fpixel, nelements, NULL, *data, NULL, &status);
    fits_close_file(fptr, &status);
    
    return status;
}

int read_frequencies(const char *filename, float **freq, int *nfreq) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening frequency file: %s\n", filename);
        return 1;
    }
    
    int count = 0;
    float dummy;
    while (fscanf(f, "%f\n", &dummy) == 1) count++;
    
    *nfreq = count;
    *freq = (float *)malloc(count * sizeof(float));
    
    rewind(f);
    for (int i = 0; i < count; i++) {
        fscanf(f, "%f\n", &(*freq)[i]);
    }
    fclose(f);
    return 0;
}

int read_weights_file(const char *filename, float **weights, int nfreq) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        return 1;
    }
    
    *weights = (float *)malloc(nfreq * sizeof(float));
    
    for (int i = 0; i < nfreq; i++) {
        if (fscanf(f, "%f\n", &(*weights)[i]) != 1) {
            fprintf(stderr, "Error reading weights at channel %d\n", i);
            fclose(f);
            free(*weights);
            return 1;
        }
    }
    
    fclose(f);
    return 0;
}

int compute_rms_weights(float **weights, float *I, int *mask, int nfreq, int ny, int nx) {
    *weights = (float *)malloc(nfreq * sizeof(float));
    
    printf("Computing per-channel RMS-based weights from I spectrum...\n");
    
    for (int f = 0; f < nfreq; f++) {
        double sum_sq = 0.0;
        int count = 0;
        
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                if (mask[y * nx + x]) {
                    float I_val = I[f * (ny * nx) + y * nx + x];
                    sum_sq += (double)I_val * I_val;
                    count++;
                }
            }
        }
        
        if (count > 0) {
            float rms = sqrtf((float)(sum_sq / count));
            (*weights)[f] = (rms > 1e-10f) ? 1.0f / (rms * rms) : 1.0f;
        } else {
            (*weights)[f] = 1.0f;
        }
    }
    
    printf("RMS weights computed.\n");
    return 0;
}

int write_fits_image(const char *filename, float *data, int ny, int nx) {
    fitsfile *fptr = NULL;
    int status = 0;
    long naxes[2] = {nx, ny};
    
    fits_create_file(&fptr, filename, &status);
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    
    long fpixel = 1;
    long nelements = nx * ny;
    fits_write_img(fptr, TFLOAT, fpixel, nelements, data, &status);
    fits_close_file(fptr, &status);
    
    return status;
}

int write_fits_cube(const char *filename, float *data, int ny, int nx, int nfara) {
    fitsfile *fptr = NULL;
    int status = 0;
    long naxes[3] = {nx, ny, nfara};
    
    fits_create_file(&fptr, filename, &status);
    fits_create_img(fptr, FLOAT_IMG, 3, naxes, &status);
    
    long fpixel = 1;
    long nelements = nx * ny * nfara;
    fits_write_img(fptr, TFLOAT, fpixel, nelements, data, &status);
    fits_close_file(fptr, &status);
    
    return status;
}

// ============================================================================
// FARADAY GRID
// ============================================================================

void create_faraday_grid(
    float f_min, float f_max, float f_res,
    float **fara_grid, int *nfara
) {
    if (f_res <= 0) {
        fprintf(stderr, "Error: Faraday resolution must be > 0\n");
        exit(1);
    }
    
    if (f_max <= f_min) {
        fprintf(stderr, "Error: f_max must be > f_min\n");
        exit(1);
    }
    
    *nfara = (int)((f_max - f_min) / f_res) + 1;
    *fara_grid = (float *)malloc(*nfara * sizeof(float));
    
    for (int i = 0; i < *nfara; i++) {
        (*fara_grid)[i] = f_min + i * f_res;
    }
    
    printf("Faraday grid: %.1f to %.1f rad/m² (resolution %.1f, NFARA=%d)\n",
           f_min, f_max, f_res, *nfara);
}

// ============================================================================
// COMMAND-LINE PARSING
// ============================================================================

void parse_user_args(int argc, char **argv, FaradayParams *fara_params, ProcessParams *proc_params) {
    // Defaults
    fara_params->f_min = -500.0f;
    fara_params->f_max = 500.0f;
    fara_params->f_res = 5.0f;
    
    proc_params->chunk_size = 0;  // 0 = auto-detect
    proc_params->clean_gain = 0.1f;
    proc_params->clean_threshold = 1e-3f;
    proc_params->clean_max_iter = 100;
    
    // Parse arguments
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-f_min") == 0) {
            fara_params->f_min = atof(argv[++i]);
        } else if (strcmp(argv[i], "-f_max") == 0) {
            fara_params->f_max = atof(argv[++i]);
        } else if (strcmp(argv[i], "-f_res") == 0) {
            fara_params->f_res = atof(argv[++i]);
        } else if (strcmp(argv[i], "-chunk") == 0) {
            proc_params->chunk_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-gain") == 0) {
            proc_params->clean_gain = atof(argv[++i]);
        } else if (strcmp(argv[i], "-threshold") == 0) {
            proc_params->clean_threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-max_iter") == 0) {
            proc_params->clean_max_iter = atoi(argv[++i]);
        }
    }
}

void print_usage(const char *prog) {
    printf("Usage: %s Q.fits U.fits I.fits mask.fits frequencies.txt output_prefix\n", prog);
    printf("       [weights.txt] [options]\n\n");
    printf("Required:\n");
    printf("  Q.fits, U.fits, I.fits          - Stokes cubes (NFREQ, NY, NX)\n");
    printf("  mask.fits                       - Source mask (1=source, 0=noise)\n");
    printf("  frequencies.txt                 - One frequency per line (Hz)\n");
    printf("  output_prefix                   - Prefix for output files\n\n");
    printf("Optional arguments:\n");
    printf("  weights.txt                     - Per-channel weights (auto-computed if missing)\n");
    printf("  -f_min MIN                      - Min Faraday depth [rad/m²] (default: -500)\n");
    printf("  -f_max MAX                      - Max Faraday depth [rad/m²] (default: 500)\n");
    printf("  -f_res RES                      - Faraday resolution [rad/m²] (default: 5)\n");
    printf("  -chunk SIZE                     - Chunk size in pixels (default: auto-detect)\n");
    printf("  -gain GAIN                      - RM-CLEAN gain [0-1] (default: 0.1)\n");
    printf("  -threshold THR                  - RM-CLEAN threshold (default: 1e-3)\n");
    printf("  -max_iter ITER                  - Max RM-CLEAN iterations (default: 100)\n\n");
    printf("Examples:\n");
    printf("  %s Q.fits U.fits I.fits mask.fits freq.txt output\n", prog);
    printf("  %s Q.fits U.fits I.fits mask.fits freq.txt output -f_min -1000 -f_max 1000\n", prog);
    printf("  %s Q.fits U.fits I.fits mask.fits freq.txt output weights.txt -chunk 256 -gain 0.05\n", prog);
}

// ============================================================================
// CHUNK PROCESSING
// ============================================================================

void process_chunk(
    DataCube *data,
    OutputMaps *output,
    float *d_freq,
    float *d_lambda_sq,
    float *d_fara_grid,
    float *d_weights,
    int nfara,
    ProcessParams *proc_params,
    int y_start, int x_start,
    int chunk_ny, int chunk_nx
) {
    int nfreq = data->NFREQ;
    
    int *valid_indices = (int *)malloc(chunk_ny * chunk_nx * sizeof(int));
    int n_valid = 0;
    
    for (int y = 0; y < chunk_ny; y++) {
        for (int x = 0; x < chunk_nx; x++) {
            int gy = y_start + y;
            int gx = x_start + x;
            if (gy < data->NY && gx < data->NX && data->mask[gy * data->NX + gx]) {
                valid_indices[n_valid++] = gy * data->NX + gx;
            }
        }
    }
    
    if (n_valid == 0) {
        free(valid_indices);
        return;
    }
    
    printf("Processing chunk [%d,%d): found %d valid pixels\n", y_start, x_start, n_valid);
    
    float *d_Q, *d_U, *d_I, *d_I_fit;
    float *d_Q_norm, *d_U_norm;
    float2 *d_rmsf;
    float *d_peak_rm, *d_rm_value, *d_rm_err, *d_rm_clean;
    
    size_t chunk_data_size = n_valid * nfreq * sizeof(float);
    size_t rmsf_size = n_valid * nfara * sizeof(float2);
    
    cudaMalloc(&d_Q, chunk_data_size);
    cudaMalloc(&d_U, chunk_data_size);
    cudaMalloc(&d_I, chunk_data_size);
    cudaMalloc(&d_I_fit, chunk_data_size);
    cudaMalloc(&d_Q_norm, chunk_data_size);
    cudaMalloc(&d_U_norm, chunk_data_size);
    cudaMalloc(&d_rmsf, rmsf_size);
    cudaMalloc(&d_peak_rm, n_valid * sizeof(float));
    cudaMalloc(&d_rm_value, n_valid * sizeof(float));
    cudaMalloc(&d_rm_err, n_valid * sizeof(float));
    cudaMalloc(&d_rm_clean, n_valid * sizeof(float));
    
    for (int i = 0; i < n_valid; i++) {
        int global_idx = valid_indices[i];
        cudaMemcpy(d_Q + i * nfreq, data->Q + global_idx * nfreq, nfreq * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U + i * nfreq, data->U + global_idx * nfreq, nfreq * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_I + i * nfreq, data->I + global_idx * nfreq, nfreq * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    cu_spline_fit(d_freq, d_I, d_I_fit, NULL, nfreq, n_valid);
    
    cudaMemcpy(d_Q_norm, d_Q, chunk_data_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_U_norm, d_U, chunk_data_size, cudaMemcpyDeviceToDevice);
    cu_normalize_qu(d_I_fit, d_Q_norm, d_U_norm, nfreq, n_valid);
    
    cu_compute_rmsf(d_Q_norm, d_U_norm, d_lambda_sq, d_weights, d_fara_grid, d_rmsf, nfreq, nfara, n_valid);
    cu_find_peak_rm(d_rmsf, d_fara_grid, d_peak_rm, d_rm_value, d_rm_err, nfara, n_valid);
    cu_rm_clean(d_rmsf, d_rm_value, d_peak_rm, d_fara_grid, d_rm_clean, nfara, n_valid, 
                proc_params->clean_gain, proc_params->clean_threshold, proc_params->clean_max_iter);
    
    float *h_peak_rm = (float *)malloc(n_valid * sizeof(float));
    float *h_rm_value = (float *)malloc(n_valid * sizeof(float));
    float *h_rm_err = (float *)malloc(n_valid * sizeof(float));
    float *h_rm_clean = (float *)malloc(n_valid * sizeof(float));
    float2 *h_rmsf = (float2 *)malloc(rmsf_size);
    
    cudaMemcpy(h_peak_rm, d_peak_rm, n_valid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rm_value, d_rm_value, n_valid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rm_err, d_rm_err, n_valid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rm_clean, d_rm_clean, n_valid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rmsf, d_rmsf, rmsf_size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n_valid; i++) {
        int global_idx = valid_indices[i];
        int gy = global_idx / data->NX;
        int gx = global_idx % data->NX;
        
        output->rm_map[gy * data->NX + gx] = h_rm_value[i];
        output->rm_clean_map[gy * data->NX + gx] = h_rm_clean[i];
        output->rm_err_map[gy * data->NX + gx] = h_rm_err[i];
        output->peak_rm_map[gy * data->NX + gx] = h_peak_rm[i];
        
        for (int f = 0; f < nfara; f++) {
            output->rmsf_cube[(gy * data->NX + gx) * nfara + f] = h_rmsf[i * nfara + f];
        }
    }
    
    cudaFree(d_Q); cudaFree(d_U); cudaFree(d_I); cudaFree(d_I_fit);
    cudaFree(d_Q_norm); cudaFree(d_U_norm);
    cudaFree(d_rmsf);
    cudaFree(d_peak_rm); cudaFree(d_rm_value); cudaFree(d_rm_err); cudaFree(d_rm_clean);
    
    free(h_peak_rm); free(h_rm_value); free(h_rm_err); free(h_rm_clean); free(h_rmsf);
    free(valid_indices);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
    if (argc < 7) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char *q_file = argv[1];
    const char *u_file = argv[2];
    const char *i_file = argv[3];
    const char *mask_file = argv[4];
    const char *freq_file = argv[5];
    const char *output_prefix = argv[6];
    const char *weights_file = NULL;
    
    if (argc > 7 && argv[7][0] != '-') {
        weights_file = argv[7];
    }
    
    FaradayParams fara_params;
    ProcessParams proc_params;
    parse_user_args(argc, argv, &fara_params, &proc_params);
    
    printf("=== RM Estimation + RM-CLEAN ===\n");
    print_vram_info();
    printf("\n");
    printf("Loading data...\n");
    
    DataCube data = {0};
    read_fits_cube(q_file, &data.Q, &data.NFREQ, &data.NY, &data.NX);
    read_fits_cube(u_file, &data.U, &data.NFREQ, &data.NY, &data.NX);
    read_fits_cube(i_file, &data.I, &data.NFREQ, &data.NY, &data.NX);
    read_fits_image(mask_file, &data.mask, &data.NY, &data.NX);
    read_frequencies(freq_file, &data.freq, &data.NFREQ);
    
    printf("Data loaded: NFREQ=%d, NY=%d, NX=%d\n", data.NFREQ, data.NY, data.NX);
    printf("Cube size: %.2f GB\n", (data.NFREQ * data.NY * data.NX * 4.0 * 3) / (1024*1024*1024));
    
    // Auto-detect chunk size if not provided
    if (proc_params.chunk_size == 0) {
        size_t available = get_available_vram();
        proc_params.chunk_size = auto_detect_chunk_size(data.NFREQ, data.NY, data.NX, available);
    } else {
        printf("Using user-specified chunk size: %d×%d\n", proc_params.chunk_size, proc_params.chunk_size);
    }
    
    float *weights = NULL;
    if (weights_file) {
        printf("Attempting to read weights from: %s\n", weights_file);
        if (read_weights_file(weights_file, &weights, data.NFREQ) == 0) {
            printf("Weights loaded from file.\n");
        } else {
            printf("Weight file not found. Computing RMS weights instead.\n");
            compute_rms_weights(&weights, data.I, data.mask, data.NFREQ, data.NY, data.NX);
        }
    } else {
        printf("No weight file specified. Computing RMS-based weights...\n");
        compute_rms_weights(&weights, data.I, data.mask, data.NFREQ, data.NY, data.NX);
    }
    
    float *fara_grid;
    int nfara;
    create_faraday_grid(fara_params.f_min, fara_params.f_max, fara_params.f_res, &fara_grid, &nfara);
    
    printf("RM-CLEAN parameters: gain=%.3f, threshold=%.2e, max_iter=%d\n",
           proc_params.clean_gain, proc_params.clean_threshold, proc_params.clean_max_iter);
    printf("\n");
    
    OutputMaps output;
    output.rm_map = (float *)malloc(data.NY * data.NX * sizeof(float));
    output.rm_clean_map = (float *)malloc(data.NY * data.NX * sizeof(float));
    output.rm_err_map = (float *)malloc(data.NY * data.NX * sizeof(float));
    output.peak_rm_map = (float *)malloc(data.NY * data.NX * sizeof(float));
    output.rmsf_cube = (float2 *)malloc(data.NY * data.NX * nfara * sizeof(float2));
    
    float nan_val = NAN;
    float2 nan_complex = {NAN, NAN};
    for (int i = 0; i < data.NY * data.NX; i++) {
        output.rm_map[i] = nan_val;
        output.rm_clean_map[i] = nan_val;
        output.rm_err_map[i] = nan_val;
        output.peak_rm_map[i] = nan_val;
        for (int f = 0; f < nfara; f++) {
            output.rmsf_cube[i * nfara + f] = nan_complex;
        }
    }
    
    float *d_freq, *d_lambda_sq, *d_fara_grid, *d_weights;
    cudaMalloc(&d_freq, data.NFREQ * sizeof(float));
    cudaMalloc(&d_lambda_sq, data.NFREQ * sizeof(float));
    cudaMalloc(&d_fara_grid, nfara * sizeof(float));
    cudaMalloc(&d_weights, data.NFREQ * sizeof(float));
    
    cudaMemcpy(d_freq, data.freq, data.NFREQ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fara_grid, fara_grid, nfara * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, data.NFREQ * sizeof(float), cudaMemcpyHostToDevice);
    cu_compute_lambda_sq(d_freq, d_lambda_sq, data.NFREQ);
    
    printf("Processing chunks...\n");
    
    for (int y = 0; y < data.NY; y += proc_params.chunk_size) {
        for (int x = 0; x < data.NX; x += proc_params.chunk_size) {
            int chunk_ny = (y + proc_params.chunk_size > data.NY) ? (data.NY - y) : proc_params.chunk_size;
            int chunk_nx = (x + proc_params.chunk_size > data.NX) ? (data.NX - x) : proc_params.chunk_size;
            
            process_chunk(&data, &output, d_freq, d_lambda_sq, d_fara_grid, d_weights, nfara, 
                         &proc_params, y, x, chunk_ny, chunk_nx);
        }
    }
    
    printf("Writing output files...\n");
    
    char filename[256];
    snprintf(filename, 256, "%s_rm_map.fits", output_prefix);
    write_fits_image(filename, output.rm_map, data.NY, data.NX);
    printf("  %s\n", filename);
    
    snprintf(filename, 256, "%s_rm_clean_map.fits", output_prefix);
    write_fits_image(filename, output.rm_clean_map, data.NY, data.NX);
    printf("  %s\n", filename);
    
    snprintf(filename, 256, "%s_rm_err_map.fits", output_prefix);
    write_fits_image(filename, output.rm_err_map, data.NY, data.NX);
    printf("  %s\n", filename);
    
    snprintf(filename, 256, "%s_peak_rm_map.fits", output_prefix);
    write_fits_image(filename, output.peak_rm_map, data.NY, data.NX);
    printf("  %s\n", filename);
    
    float *rmsf_real = (float *)malloc(data.NY * data.NX * nfara * sizeof(float));
    float *rmsf_imag = (float *)malloc(data.NY * data.NX * nfara * sizeof(float));
    
    for (int i = 0; i < data.NY * data.NX * nfara; i++) {
        rmsf_real[i] = output.rmsf_cube[i].x;
        rmsf_imag[i] = output.rmsf_cube[i].y;
    }
    
    snprintf(filename, 256, "%s_rmsf_real.fits", output_prefix);
    write_fits_cube(filename, rmsf_real, data.NY, data.NX, nfara);
    printf("  %s\n", filename);
    
    snprintf(filename, 256, "%s_rmsf_imag.fits", output_prefix);
    write_fits_cube(filename, rmsf_imag, data.NY, data.NX, nfara);
    printf("  %s\n", filename);
    
    free(rmsf_real);
    free(rmsf_imag);
    
    printf("\nDone! All output files written with prefix: %s\n", output_prefix);
    
    free(data.Q); free(data.U); free(data.I); free(data.mask); free(data.freq);
    free(output.rm_map); free(output.rm_clean_map); free(output.rm_err_map); free(output.peak_rm_map); free(output.rmsf_cube);
    free(weights); free(fara_grid);
    cudaFree(d_freq); cudaFree(d_lambda_sq); cudaFree(d_fara_grid); cudaFree(d_weights);
    
    return 0;
}