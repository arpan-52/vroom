#ifndef VROOM_IO_H
#define VROOM_IO_H

#include "vroom.h"

// ============================================================================
// FITS CUBE I/O
// ============================================================================

// Read a 3D FITS cube (NFREQ x NY x NX)
// Returns 0 on success, non-zero on error
int read_fits_cube(
    const char* filename,
    float** data,
    CubeInfo* info
);

// Read specific rows of a FITS cube (for batched loading)
// Reads pixels [start_pixel, start_pixel + npixels) across all frequencies
// Output layout: [npixels * nfreq] with frequency as fast axis
int read_fits_cube_batch(
    const char* filename,
    float* data,
    const CubeInfo& info,
    int start_pixel,
    int npixels
);

// Fast batch read: channel-major strategy (reads spatial slabs per channel)
// Reads selected channels for a contiguous pixel range, output: [npixels * nchannels_out]
// chan_map[i] = original channel index for output channel i
int read_fits_cube_batch_fast(
    const char* filename,
    float* data,
    int nx, int ny,
    int start_pixel, int npixels,
    const int* chan_map,
    int nchannels_out,
    int nfreq_orig
);

// Write a 2D FITS image
int write_fits_image(
    const char* filename,
    const float* data,
    int ny,
    int nx,
    const char* extname = nullptr
);

// Write a 3D FITS cube
int write_fits_cube(
    const char* filename,
    const float* data,
    int nz,
    int ny,
    int nx,
    const char* extname = nullptr
);

// Write FITS cube with proper WCS for Faraday depth axis
int write_fdf_cube(
    const char* filename,
    const float* data,
    int nphi,
    int ny,
    int nx,
    const FaradayConfig& faraday
);

// ============================================================================
// FREQUENCY FILE I/O
// ============================================================================

// Read frequencies from text file (one per line, in Hz)
int read_frequencies(
    const char* filename,
    float** freq,
    int* nfreq
);

// Read optional weights file
int read_weights(
    const char* filename,
    float** weights,
    int nfreq
);

// ============================================================================
// RMSF I/O
// ============================================================================

// Save RMSF to FITS
int save_rmsf(
    const char* filename,
    const RMSF& rmsf,
    const FaradayConfig& faraday
);

// Load RMSF from FITS (for separate RM-CLEAN step)
int load_rmsf(
    const char* filename,
    RMSF* rmsf,
    FaradayConfig* faraday
);

#endif // VROOM_IO_H
