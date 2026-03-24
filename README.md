# VROOM - GPU RM-Synthesis + RM-CLEAN

<p align="center">
  <img src="assets/cat-driving-serious.gif" width="400"/>
</p>

*Taken from [Tenor](https://tenor.com/view/cat-driving-serious-cat-driving-driving-fast-gif-12051647047998874147)*

Fast GPU-accelerated Rotation Measure synthesis and RM-CLEAN deconvolution for radio polarimetry, using **NUFFT** for optimal performance with non-uniform λ² sampling.

## Features

- **NUFFT-accelerated RM-Synthesis**: Uses cuFINUFFT for O(N log N) complexity instead of O(N²) direct DFT
- **Proper RM-CLEAN**: Full deconvolution with RMSF subtraction and Gaussian restoring beam
- **Optional I normalization**: Divide Q/U by model I cube (with zero-division protection)
- **Async pipeline**: Overlaps CPU I/O with GPU computation for maximum throughput
- **Memory-aware batching**: Automatically sizes batches based on available RAM and VRAM
- **User-provided weights**: Accepts per-channel weights for proper noise handling

## Requirements

- CUDA Toolkit (11.0+)
- **cuFINUFFT** library (https://github.com/flatironinstitute/cufinufft)
- CFITSIO library
- C++14 compiler

### Installation

#### Ubuntu/Debian
```bash
# CFITSIO
sudo apt install libcfitsio-dev

# cuFINUFFT (build from source or use conda)
git clone https://github.com/flatironinstitute/cufinufft.git
cd cufinufft
make -j
sudo make install

# Or via conda:
conda install -c conda-forge cufinufft
```

## Building

```bash
make                                    # Default (sm_70 for V100)
make CUDA_ARCH=sm_80                    # For A100
make CUDA_ARCH=sm_86                    # For RTX 3090  
make CUFINUFFT_DIR=/opt/cufinufft       # Custom cuFINUFFT path
```

## Usage

### Basic RM-Synthesis + RM-CLEAN
```bash
./vroom -q Q.fits -u U.fits -f frequencies.txt -o output
```

### With model I normalization
```bash
./vroom -q Q.fits -u U.fits -i I_model.fits -f frequencies.txt -o output
```

### Full options
```bash
./vroom -q Q.fits -u U.fits -f frequencies.txt -o output \
        --phi-min -1000 --phi-max 1000 --dphi 0.5 \
        --clean-gain 0.1 --clean-maxiter 5000 \
        --save-fdf --save-rmsf
```

### RM-Synthesis only (no CLEAN)
```bash
./vroom -q Q.fits -u U.fits -f frequencies.txt -o output --no-clean
```

## Input Files

| File | Description |
|------|-------------|
| Q.fits | Stokes Q cube [NX × NY × NFREQ] |
| U.fits | Stokes U cube [NX × NY × NFREQ] |
| I.fits | (Optional) Model I cube for normalization |
| frequencies.txt | Frequencies in Hz, one per line |
| weights.txt | (Optional) Per-channel weights |

## Output Files

| File | Description |
|------|-------------|
| `{prefix}_peak_rm.fits` | Peak RM value per pixel [rad/m²] |
| `{prefix}_peak_pi.fits` | Peak polarized intensity |
| `{prefix}_rm_err.fits` | RM uncertainty estimate |
| `{prefix}_clean_peak_rm.fits` | Peak RM after CLEAN (if enabled) |
| `{prefix}_clean_peak_pi.fits` | Peak PI after CLEAN (if enabled) |
| `{prefix}_rmsf.fits` | RMSF (if --save-rmsf) |
| `{prefix}_fdf_real.fits` | FDF real part cube (if --save-fdf) |
| `{prefix}_fdf_imag.fits` | FDF imaginary part cube (if --save-fdf) |

## Command Line Options

```
Required:
  -q, --q-cube FILE        Q Stokes cube (FITS)
  -u, --u-cube FILE        U Stokes cube (FITS)
  -f, --freq FILE          Frequency file (Hz, one per line)
  -o, --output PREFIX      Output file prefix

Optional:
  -i, --i-cube FILE        Model I cube for normalization (Q/I, U/I)
  -w, --weights FILE       Per-channel weights file
  --phi-min VALUE          Min Faraday depth [rad/m²] (default: -500)
  --phi-max VALUE          Max Faraday depth [rad/m²] (default: +500)
  --dphi VALUE             Faraday depth step [rad/m²] (default: 1)
  --no-clean               Skip RM-CLEAN, only do RM-synthesis
  --clean-gain VALUE       RM-CLEAN loop gain (default: 0.1)
  --clean-cutoff VALUE     RM-CLEAN absolute threshold
  --clean-maxiter VALUE    Max CLEAN iterations (default: 1000)
  --save-fdf               Save full FDF cube
  --save-rmsf              Save RMSF to file
  -h, --help               Show help
```

## Algorithm

### RM-Synthesis via NUFFT
Computes the Faraday Dispersion Function using Type-1 NUFFT:

```
F(φ) = K ∑ᵢ wᵢ Pᵢ exp(-2iφ(λ²ᵢ - λ²₀))
```

where P = Q + iU is the complex polarization, λ² is wavelength squared, and φ is Faraday depth.

**Why NUFFT?** Radio observations have non-uniform λ² spacing. Direct DFT costs O(N_freq × N_phi) per pixel. NUFFT reduces this to O(N_freq log N_freq + N_phi log N_phi), providing significant speedup for large Faraday depth ranges.

### RM-CLEAN
1. Initialize: residual = dirty FDF, model = 0
2. Find peak in |residual|
3. Subtract: residual -= gain × peak × RMSF(φ - φ_peak)
4. Add to model: model[φ_peak] += gain × peak
5. Repeat until threshold or max iterations
6. Convolve model with Gaussian restoring beam (FWHM = RMSF FWHM)
7. Add residuals: clean_FDF = convolved_model + residual

## Performance

The async pipeline overlaps:
- **CPU**: Loading next batch from FITS files
- **GPU**: Processing current batch (RM-synthesis + CLEAN)
- **CPU**: Writing previous batch results

Batch size is automatically computed based on available RAM and VRAM to maximize throughput while avoiding out-of-memory errors.

## Testing

```bash
make test    # Generate synthetic data and run
```

## License

MIT
