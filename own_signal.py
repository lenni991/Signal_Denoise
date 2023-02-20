# Generate a noisy signal
signal = np.random.randn(1000)
noise = 0.1np.random.randn(1000)
noisy_signal = signal + noise

# Denoise the signal and print the SNR
denoised_signal, snr = denoise_signal(noisy_signal)
print('SNR', snr)
