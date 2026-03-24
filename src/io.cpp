#include "io.h"
#include <fitsio.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// ERROR HANDLING
// ============================================================================

static void fits_report_error(int status) {
    if (status) {
        fits_report_error(stderr, status);
    }
}

// ============================================================================
// FITS CUBE I/O
// ============================================================================

int read_fits_cube(
    const char* filename,
    float** data,
    CubeInfo* info
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    int naxis;
    long naxes[3];
    
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fprintf(stderr, "Error opening FITS file: %s\n", filename);
        fits_report_error(status);
        return status;
    }
    
    fits_get_img_dim(fptr, &naxis, &status);
    if (naxis != 3) {
        fprintf(stderr, "Error: Expected 3D cube, got %d dimensions\n", naxis);
        fits_close_file(fptr, &status);
        return -1;
    }
    
    fits_get_img_size(fptr, 3, naxes, &status);
    
    // FITS is FORTRAN order: naxes[0]=NX, naxes[1]=NY, naxes[2]=NFREQ
    info->nx = (int)naxes[0];
    info->ny = (int)naxes[1];
    info->nfreq = (int)naxes[2];
    info->npix = (size_t)info->nx * info->ny;
    info->nelems = info->npix * info->nfreq;
    
    *data = (float*)malloc(info->nelems * sizeof(float));
    if (!*data) {
        fprintf(stderr, "Error: Failed to allocate %.2f GB for cube\n",
                info->nelems * sizeof(float) / 1e9);
        fits_close_file(fptr, &status);
        return -1;
    }
    
    long fpixel = 1;
    fits_read_img(fptr, TFLOAT, fpixel, info->nelems, nullptr, *data, nullptr, &status);
    fits_close_file(fptr, &status);
    
    if (status) {
        fits_report_error(status);
        free(*data);
        *data = nullptr;
    }
    
    return status;
}

int read_fits_cube_batch(
    const char* filename,
    float* data,
    const CubeInfo& info,
    int start_pixel,
    int npixels
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fits_report_error(status);
        return status;
    }
    
    // Read pixel by pixel across all frequencies
    // FITS stores as [NX, NY, NFREQ], we want [npixels, NFREQ]
    // where pixel index = y * NX + x
    
    long fpixel[3];
    long inc[3] = {1, 1, 1};
    
    for (int p = 0; p < npixels; p++) {
        int global_pix = start_pixel + p;
        int y = global_pix / info.nx;
        int x = global_pix % info.nx;
        
        // Read all frequencies for this pixel
        fpixel[0] = x + 1;  // FITS is 1-indexed
        fpixel[1] = y + 1;
        fpixel[2] = 1;
        
        long lpixel[3] = {x + 1, y + 1, info.nfreq};
        
        // Output goes to data[p * nfreq]
        fits_read_subset(fptr, TFLOAT, fpixel, lpixel, inc,
                         nullptr, data + p * info.nfreq, nullptr, &status);
        
        if (status) break;
    }
    
    fits_close_file(fptr, &status);
    if (status) fits_report_error(status);
    
    return status;
}

int read_fits_cube_batch_fast(
    const char* filename,
    float* data,
    int nx, int ny,
    int start_pixel, int npixels,
    const int* chan_map,
    int nchannels_out,
    int nfreq_orig
) {
    fitsfile* fptr = nullptr;
    int status = 0;

    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fits_report_error(status);
        return status;
    }

    // Compute row range covering all batch pixels
    int start_row = start_pixel / nx;
    int end_row = (start_pixel + npixels - 1) / nx;
    int num_rows = end_row - start_row + 1;
    int slab_size = nx * num_rows;

    // Slab buffer for one frequency channel's spatial region
    float* slab = (float*)malloc((size_t)slab_size * sizeof(float));

    // Offset of first batch pixel within slab
    int slab_offset_start = start_pixel - start_row * nx;

    long inc[3] = {1, 1, 1};

    for (int v = 0; v < nchannels_out; v++) {
        int orig_chan = chan_map[v];

        // Read spatial slab [1:nx, start_row+1:end_row+1] for this channel
        long fpixel[3] = {1, start_row + 1, orig_chan + 1};
        long lpixel[3] = {(long)nx, end_row + 1, orig_chan + 1};

        fits_read_subset(fptr, TFLOAT, fpixel, lpixel, inc,
                         nullptr, slab, nullptr, &status);
        if (status) break;

        // Extract batch pixels from slab
        // Batch pixels are contiguous: global_pix = start_pixel + p
        // slab index for global_pix = (global_pix - start_row * nx) = slab_offset_start + p
        for (int p = 0; p < npixels; p++) {
            data[p * nchannels_out + v] = slab[slab_offset_start + p];
        }
    }

    free(slab);
    fits_close_file(fptr, &status);
    if (status) fits_report_error(status);
    return status;
}

int write_fits_image(
    const char* filename,
    const float* data,
    int ny,
    int nx,
    const char* extname
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    long naxes[2] = {nx, ny};
    
    // Remove existing file if present
    remove(filename);
    
    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(status);
        return status;
    }
    
    fits_create_img(fptr, FLOAT_IMG, 2, naxes, &status);
    
    if (extname) {
        fits_write_key(fptr, TSTRING, "EXTNAME", (void*)extname, nullptr, &status);
    }
    
    long fpixel = 1;
    long nelements = (long)nx * ny;
    fits_write_img(fptr, TFLOAT, fpixel, nelements, (void*)data, &status);
    fits_close_file(fptr, &status);
    
    if (status) fits_report_error(status);
    return status;
}

int write_fits_cube(
    const char* filename,
    const float* data,
    int nz,
    int ny,
    int nx,
    const char* extname
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    long naxes[3] = {nx, ny, nz};
    
    remove(filename);
    
    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(status);
        return status;
    }
    
    fits_create_img(fptr, FLOAT_IMG, 3, naxes, &status);
    
    if (extname) {
        fits_write_key(fptr, TSTRING, "EXTNAME", (void*)extname, nullptr, &status);
    }
    
    long fpixel = 1;
    long nelements = (long)nx * ny * nz;
    fits_write_img(fptr, TFLOAT, fpixel, nelements, (void*)data, &status);
    fits_close_file(fptr, &status);
    
    if (status) fits_report_error(status);
    return status;
}

int write_fdf_cube(
    const char* filename,
    const float* data,
    int nphi,
    int ny,
    int nx,
    const FaradayConfig& faraday
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    long naxes[3] = {nx, ny, nphi};
    
    remove(filename);
    
    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(status);
        return status;
    }
    
    fits_create_img(fptr, FLOAT_IMG, 3, naxes, &status);
    
    // WCS for Faraday depth axis
    float crpix3 = 1.0f;
    fits_write_key(fptr, TSTRING, "CTYPE3", (void*)"FDEP", 
                   "Faraday depth", &status);
    fits_write_key(fptr, TFLOAT, "CRPIX3", (void*)&crpix3, 
                   "Reference pixel", &status);
    fits_write_key(fptr, TFLOAT, "CRVAL3", (void*)&faraday.phi_min, 
                   "Reference value [rad/m^2]", &status);
    fits_write_key(fptr, TFLOAT, "CDELT3", (void*)&faraday.dphi, 
                   "Increment [rad/m^2]", &status);
    fits_write_key(fptr, TSTRING, "CUNIT3", (void*)"rad/m^2", 
                   "Units", &status);
    
    long fpixel = 1;
    long nelements = (long)nx * ny * nphi;
    fits_write_img(fptr, TFLOAT, fpixel, nelements, (void*)data, &status);
    fits_close_file(fptr, &status);
    
    if (status) fits_report_error(status);
    return status;
}

// ============================================================================
// FREQUENCY FILE I/O
// ============================================================================

int read_frequencies(
    const char* filename,
    float** freq,
    int* nfreq
) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening frequency file: %s\n", filename);
        return -1;
    }
    
    // Count lines
    int count = 0;
    float dummy;
    while (fscanf(f, "%f", &dummy) == 1) count++;
    
    if (count == 0) {
        fprintf(stderr, "Error: Empty frequency file\n");
        fclose(f);
        return -1;
    }
    
    *nfreq = count;
    *freq = (float*)malloc(count * sizeof(float));
    
    rewind(f);
    for (int i = 0; i < count; i++) {
        if (fscanf(f, "%f", &(*freq)[i]) != 1) {
            fprintf(stderr, "Error reading frequency at line %d\n", i + 1);
            free(*freq);
            fclose(f);
            return -1;
        }
    }
    
    fclose(f);
    return 0;
}

int read_weights(
    const char* filename,
    float** weights,
    int nfreq
) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        return -1;  // Not an error, weights are optional
    }
    
    *weights = (float*)malloc(nfreq * sizeof(float));
    
    for (int i = 0; i < nfreq; i++) {
        if (fscanf(f, "%f", &(*weights)[i]) != 1) {
            fprintf(stderr, "Error reading weight at line %d\n", i + 1);
            free(*weights);
            *weights = nullptr;
            fclose(f);
            return -1;
        }
    }
    
    fclose(f);
    return 0;
}

// ============================================================================
// RMSF I/O
// ============================================================================

int save_rmsf(
    const char* filename,
    const RMSF& rmsf,
    const FaradayConfig& faraday
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    remove(filename);
    
    fits_create_file(&fptr, filename, &status);
    if (status) {
        fits_report_error(status);
        return status;
    }
    
    // Primary HDU: RMSF amplitude
    long naxes[1] = {faraday.nphi};
    float crpix1 = 1.0f;
    fits_create_img(fptr, FLOAT_IMG, 1, naxes, &status);
    fits_write_key(fptr, TSTRING, "EXTNAME", (void*)"RMSF_AMP", nullptr, &status);
    fits_write_key(fptr, TFLOAT, "CRPIX1", (void*)&crpix1, nullptr, &status);
    fits_write_key(fptr, TFLOAT, "CRVAL1", (void*)&faraday.phi_min, nullptr, &status);
    fits_write_key(fptr, TFLOAT, "CDELT1", (void*)&faraday.dphi, nullptr, &status);
    fits_write_key(fptr, TFLOAT, "FWHM", (void*)&rmsf.fwhm, "RMSF FWHM [rad/m^2]", &status);
    fits_write_key(fptr, TFLOAT, "MAXSCALE", (void*)&rmsf.max_scale, 
                   "Max recoverable scale [rad/m^2]", &status);
    
    fits_write_img(fptr, TFLOAT, 1, faraday.nphi, (void*)rmsf.rmsf_amp, &status);
    
    // Extension: RMSF real
    fits_create_img(fptr, FLOAT_IMG, 1, naxes, &status);
    fits_write_key(fptr, TSTRING, "EXTNAME", (void*)"RMSF_REAL", nullptr, &status);
    fits_write_img(fptr, TFLOAT, 1, faraday.nphi, (void*)rmsf.rmsf_real, &status);
    
    // Extension: RMSF imag
    fits_create_img(fptr, FLOAT_IMG, 1, naxes, &status);
    fits_write_key(fptr, TSTRING, "EXTNAME", (void*)"RMSF_IMAG", nullptr, &status);
    fits_write_img(fptr, TFLOAT, 1, faraday.nphi, (void*)rmsf.rmsf_imag, &status);
    
    fits_close_file(fptr, &status);
    if (status) fits_report_error(status);
    return status;
}

int load_rmsf(
    const char* filename,
    RMSF* rmsf,
    FaradayConfig* faraday
) {
    fitsfile* fptr = nullptr;
    int status = 0;
    
    fits_open_file(&fptr, filename, READONLY, &status);
    if (status) {
        fits_report_error(status);
        return status;
    }
    
    // Read dimensions
    int naxis;
    long naxes[1];
    fits_get_img_dim(fptr, &naxis, &status);
    fits_get_img_size(fptr, 1, naxes, &status);
    faraday->nphi = (int)naxes[0];
    
    // Read WCS
    fits_read_key(fptr, TFLOAT, "CRVAL1", &faraday->phi_min, nullptr, &status);
    fits_read_key(fptr, TFLOAT, "CDELT1", &faraday->dphi, nullptr, &status);
    faraday->phi_max = faraday->phi_min + (faraday->nphi - 1) * faraday->dphi;
    
    // Read FWHM
    fits_read_key(fptr, TFLOAT, "FWHM", &rmsf->fwhm, nullptr, &status);
    fits_read_key(fptr, TFLOAT, "MAXSCALE", &rmsf->max_scale, nullptr, &status);
    
    // Allocate and read data
    rmsf->rmsf_amp = (float*)malloc(faraday->nphi * sizeof(float));
    rmsf->rmsf_real = (float*)malloc(faraday->nphi * sizeof(float));
    rmsf->rmsf_imag = (float*)malloc(faraday->nphi * sizeof(float));
    
    fits_read_img(fptr, TFLOAT, 1, faraday->nphi, nullptr, rmsf->rmsf_amp, nullptr, &status);
    
    // Move to RMSF_REAL extension
    fits_movnam_hdu(fptr, IMAGE_HDU, (char*)"RMSF_REAL", 0, &status);
    fits_read_img(fptr, TFLOAT, 1, faraday->nphi, nullptr, rmsf->rmsf_real, nullptr, &status);
    
    // Move to RMSF_IMAG extension
    fits_movnam_hdu(fptr, IMAGE_HDU, (char*)"RMSF_IMAG", 0, &status);
    fits_read_img(fptr, TFLOAT, 1, faraday->nphi, nullptr, rmsf->rmsf_imag, nullptr, &status);
    
    fits_close_file(fptr, &status);
    if (status) fits_report_error(status);
    return status;
}
