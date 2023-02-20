import numpy as np
import pywt

def denoise_signal(signal):
    # Decompose signal into wavelet coefficients
    coeffs = pywt.wavedec(signal, 'db4', level=6)

    # Threshold the coefficients using universal threshold
    threshold = np.sqrt(2*np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # Reconstruct the signal from the denoised coefficients
    denoised_signal = pywt.waverec(denoised_coeffs, 'db4')

    # Calculate the signal-to-noise ratio (SNR)
    noise = signal - denoised_signal
    snr = 20*np.log10(np.linalg.norm(signal)/np.linalg.norm(noise))

    return denoised_signal, snr
