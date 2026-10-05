import numpy as np

def low_pass_filter(adata: np.ndarray, bandlimit_index=1000, order=4) -> np.ndarray:
    n = adata.shape[1]
    fsig = np.fft.fft(adata, axis=1)

    freq_indices = np.fft.fftfreq(n) * n

    # Réponse de Butterworth (douce, pas de coupure nette)
    response = 1.0 / np.sqrt(1.0 + (freq_indices / bandlimit_index) ** (2 * order))

    fsig_filtered = fsig * response  # broadcasting sur l'axe fréquence

    adata_filtered = np.fft.ifft(fsig_filtered, axis=1)
    return np.real(adata_filtered)