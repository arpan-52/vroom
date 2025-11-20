#!/usr/bin/env python3
"""
Generate example RM data for testing the CUDA RM estimation code.

Requires: numpy, astropy

Usage:
    python3 generate_test_data.py
"""

import numpy as np
from astropy.io import fits

def generate_test_data(nfreq=32, ny=512, nx=512, output_prefix='test'):
    """
    Generate synthetic RM test cube.
    
    Parameters:
    - nfreq: Number of frequency channels
    - ny, nx: Spatial dimensions
    - output_prefix: Output filename prefix
    """
    
    print(f"Generating test data: {nfreq} freq, {ny}x{nx} spatial")
    
    # Frequency array (800-900 MHz)
    freq = np.linspace(800e6, 900e6, nfreq, dtype=np.float32)
    np.savetxt(f'{output_prefix}_frequencies.txt', freq)
    print(f"  Frequencies: {freq[0]/1e6:.1f} - {freq[-1]/1e6:.1f} MHz")
    
    # Generate source mask (20% of pixels are sources)
    mask = np.random.rand(ny, nx) > 0.8
    mask = mask.astype(np.int32)
    print(f"  Source pixels: {mask.sum()}")
    
    # Write mask FITS
    hdu = fits.PrimaryHDU(mask)
    hdu.writeto(f'{output_prefix}_mask.fits', overwrite=True)
    
    # Generate synthetic RM cube
    # RM varies spatially, constant per pixel across frequencies
    rm_map = np.random.normal(0, 100, (ny, nx))  # rad/m²
    
    # Synthetic Stokes I: power-law spectrum I(ν) = I0 * (ν/ν_ref)^α
    I0 = 1.0
    alpha = -2.5
    nu_ref = freq[nfreq // 2]
    
    I_cube = np.zeros((nfreq, ny, nx), dtype=np.float32)
    Q_cube = np.zeros((nfreq, ny, nx), dtype=np.float32)
    U_cube = np.zeros((nfreq, ny, nx), dtype=np.float32)
    
    c = 299792458.0
    
    for f in range(nfreq):
        # Power-law spectrum
        I_spec = I0 * (freq[f] / nu_ref) ** alpha
        
        # Q, U depend on RM
        wavelength = c / freq[f]
        lambda_sq = wavelength ** 2
        
        for y in range(ny):
            for x in range(nx):
                if mask[y, x]:
                    # Add some intrinsic polarization with RM rotation
                    q_intrinsic = 0.1 * I_spec * np.cos(2 * rm_map[y, x] * lambda_sq)
                    u_intrinsic = 0.1 * I_spec * np.sin(2 * rm_map[y, x] * lambda_sq)
                    
                    # Add noise
                    noise = np.random.normal(0, 0.01 * I_spec, 2)
                    Q_cube[f, y, x] = q_intrinsic + noise[0]
                    U_cube[f, y, x] = u_intrinsic + noise[1]
                    I_cube[f, y, x] = I_spec + np.random.normal(0, 0.01 * I_spec)
                else:
                    # Noise only
                    I_cube[f, y, x] = np.random.normal(0, 0.05)
                    Q_cube[f, y, x] = np.random.normal(0, 0.01)
                    U_cube[f, y, x] = np.random.normal(0, 0.01)
    
    # Write FITS cubes
    hdu = fits.PrimaryHDU(I_cube)
    hdu.writeto(f'{output_prefix}_I.fits', overwrite=True)
    print(f"  Wrote {output_prefix}_I.fits")
    
    hdu = fits.PrimaryHDU(Q_cube)
    hdu.writeto(f'{output_prefix}_Q.fits', overwrite=True)
    print(f"  Wrote {output_prefix}_Q.fits")
    
    hdu = fits.PrimaryHDU(U_cube)
    hdu.writeto(f'{output_prefix}_U.fits', overwrite=True)
    print(f"  Wrote {output_prefix}_U.fits")
    
    # Generate RMS weights (inverse variance per channel)
    weights = np.ones(nfreq, dtype=np.float32)
    # Add some channel variation
    weights[5:10] *= 0.8  # Some channels noisier
    np.savetxt(f'{output_prefix}_weights.txt', weights)
    print(f"  Wrote {output_prefix}_weights.txt")
    
    print("\nGenerated files:")
    print(f"  {output_prefix}_I.fits (I cube)")
    print(f"  {output_prefix}_Q.fits (Q cube)")
    print(f"  {output_prefix}_U.fits (U cube)")
    print(f"  {output_prefix}_mask.fits (source mask)")
    print(f"  {output_prefix}_frequencies.txt (frequency array)")
    print(f"  {output_prefix}_weights.txt (per-channel weights)")
    print("\nUsage:")
    print(f"  ./rm_clean {output_prefix}_Q.fits {output_prefix}_U.fits {output_prefix}_I.fits \\")
    print(f"             {output_prefix}_mask.fits {output_prefix}_frequencies.txt {output_prefix}_out \\")
    print(f"             {output_prefix}_weights.txt")

if __name__ == '__main__':
    generate_test_data(nfreq=32, ny=256, nx=256, output_prefix='test')