import numpy as np

def low_pass_filter(adata: np.ndarray, bandlimit_index=1000, order=4) -> np.ndarray:
    n = adata.shape[1]
    fsig = np.fft.fft(adata, axis=1)

    freq_indices = np.fft.fftfreq(n) * n

    # Butterworth for low pass filter:
    response = 1.0 / np.sqrt(1.0 + (freq_indices / bandlimit_index) ** (2 * order))

    fsig_filtered = fsig * response  # broadcasting sur l'axe fréquence

    adata_filtered = np.fft.ifft(fsig_filtered, axis=1)
    return np.real(adata_filtered)

def band_pass_filter(adata: np.ndarray, low_frequency: float, high_frequency: float, sampling_frequency: float, order=4) -> np.ndarray:
    n = adata.shape[1]
    fft_spectrum = np.fft.fft(adata, axis=1)
    freq_indices = np.abs(np.fft.fftfreq(n, d = 1 / sampling_frequency))

    center_freq = np.sqrt(low_frequency * high_frequency)
    bandwidth = (high_frequency - low_frequency)

    non_zeros_mask = freq_indices != 0
    non_zero = freq_indices[non_zeros_mask]

    band_pass = np.zeros_like(freq_indices)

    # butterworth for band pass filter
    ratio = (non_zero ** 2 - center_freq ** 2) / (non_zero * bandwidth)
    band_pass[non_zeros_mask] = 1.0 / np.sqrt(1.0 + ratio ** (2 * order))


    spectrum_filtered = fft_spectrum * band_pass

    adata_filtered = np.fft.ifft(spectrum_filtered, axis=1)
    return np.real(adata_filtered)