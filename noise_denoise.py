import numpy as np
import pywt

def add_noise(signal, noise_std):
    noise = np.random.normal(0, noise_std, size=len(signal))
    noisy_signal = signal + noise
    return noisy_signal

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

# Generate a signal
signal = np.sin(2*np.pi*5*np.linspace(0, 1, num=1000))

# Add Gaussian noise to the signal
noise_std = 0.1
noisy_signal = add_noise(signal, noise_std)

# Denoise the signal and print the SNR
denoised_signal, snr = denoise_signal(noisy_signal)
print('SNR:', snr)
