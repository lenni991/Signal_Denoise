import numpy as np
import pywt
import matplotlib.pyplot as plt

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

# Generate a noisy signal
signal = np.random.randn(1000)
noise = 0.1*np.random.randn(1000)
noisy_signal = signal + noise

# Denoise the signal and plot the denoised signal
denoised_signal, snr = denoise_signal(noisy_signal)
plt.plot(signal, label='Original signal')
plt.plot(noisy_signal, label='Noisy signal')
plt.plot(denoised_signal, label='Denoised signal')
plt.legend()
plt.show()

# Print the SNR
print('SNR:', snr)
