import numpy as np
from .fft import rfft_block, irfft_block

def get_window(window_type: str, window_length: int) -> np.ndarray:
    """
    Returns a window function of specified type and sample length.
    
    Supported window types:
    - 'hann' / 'hanning': Raised cosine (Hann) window.
    - 'hamming': Hamming window.
    - 'rectangular' / 'boxcar' / None: Rectangular window (all ones).
    """
    if window_type in ('hann', 'hanning'):
        return np.hanning(window_length)
    elif window_type == 'hamming':
        return np.hamming(window_length)
    elif window_type in ('rectangular', 'boxcar') or window_type is None:
        return np.ones(window_length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

def stft(sig: np.ndarray, window_length: int, hop_length: int, window_type: str = 'hann', fft_length: int = None) -> np.ndarray:
    """
    Computes Short-Time Fourier Transform (STFT) of a 1D (mono) or 2D (stereo/multichannel) signal
    by computing the whole block of frames at once using compute shader block processing.
    
    Parameters:
        sig (np.ndarray): Input signal array of shape (samples,) or (samples, channels).
        window_length (int): Length of the analysis window in samples.
        hop_length (int): Step size between consecutive frames in samples.
        window_type (str): Type of analysis window function ('hann', 'hamming', 'rectangular').
        fft_length (int, optional): FFT size. Defaults to next power of 2 >= window_length.
        
    Returns:
        np.ndarray: Complex STFT coefficients matrix with shape (freq_bins, num_frames) for 1D
                    or (freq_bins, num_frames, channels) for multi-channel signals.
    """
    if fft_length is None:
        fft_length = 1 << (window_length - 1).bit_length()
        
    if fft_length < window_length:
        raise ValueError("fft_length must be greater than or equal to window_length")

    w_a = get_window(window_type, window_length)
    
    # Handle multi-channel signals
    is_multichannel = sig.ndim == 2
    if is_multichannel:
        num_channels = sig.shape[1]
        sig_len = sig.shape[0]
    else:
        num_channels = 1
        sig_len = len(sig)
        sig = sig[:, np.newaxis]

    # Boundary padding: zero-pad by window_length at both ends to ensure steady-state overlap-add
    pad_width = window_length
    padded_sig_len = pad_width + sig_len + pad_width
    num_frames = (padded_sig_len - window_length) // hop_length + 1
    if num_frames < 1:
        num_frames = 1
        
    required_padded_length = (num_frames - 1) * hop_length + window_length
    if required_padded_length < padded_sig_len:
        num_frames += 1
        required_padded_length = (num_frames - 1) * hop_length + window_length

    end_pad = required_padded_length - (pad_width + sig_len)

    # Pad signal with zeros at both ends
    sig_padded = np.pad(sig, ((pad_width, end_pad), (0, 0)), mode='constant')

    # Construct frame sample index matrix: shape (num_frames, window_length)
    frame_starts = np.arange(num_frames) * hop_length
    sample_offsets = np.arange(window_length)
    idx = frame_starts[:, np.newaxis] + sample_offsets[np.newaxis, :]

    # Extract all windowed frame segments: shape (num_frames, window_length, num_channels)
    segments = sig_padded[idx, :] * w_a[np.newaxis, :, np.newaxis]

    # Reshape for whole block FFT processing: shape (num_frames * num_channels, window_length)
    flat_segments = np.transpose(segments, (0, 2, 1)).reshape(num_frames * num_channels, window_length)

    # Compute real FFT for the whole block of frames at once
    flat_stft = rfft_block(flat_segments, n=fft_length)

    # Reshape back to STFT matrix: shape (freq_bins, num_frames, num_channels)
    num_bins = fft_length // 2 + 1
    stft_matrix = flat_stft.reshape(num_frames, num_channels, num_bins)
    stft_matrix = np.transpose(stft_matrix, (2, 0, 1))

    if not is_multichannel:
        stft_matrix = stft_matrix[:, :, 0]

    return stft_matrix

def istft(stft_matrix: np.ndarray, window_length: int, hop_length: int, fft_length: int, window_type: str = 'hann', original_length: int = None) -> np.ndarray:
    """
    Performs Inverse Short-Time Fourier Transform (ISTFT) with whole-block IFFT and
    overlap-add normalization to guarantee exact signal reconstruction.
    
    Parameters:
        stft_matrix (np.ndarray): Complex STFT coefficient matrix.
        window_length (int): Length of the window in samples.
        hop_length (int): Step size between consecutive frames in samples.
        fft_length (int): FFT size.
        window_type (str): Type of analysis window used in forward STFT.
        original_length (int, optional): Truncates the output to original sample length.
        
    Returns:
        np.ndarray: Reconstructed time-domain signal array.
    """
    pad_width = window_length

    if stft_matrix.ndim == 3:
        num_channels = stft_matrix.shape[2]
        recon_channels = []
        for c in range(num_channels):
            recon_c = _istft_single_channel(stft_matrix[:, :, c], window_length, hop_length, fft_length, window_type)
            recon_channels.append(recon_c)
        recon = np.column_stack(recon_channels)
    else:
        recon = _istft_single_channel(stft_matrix, window_length, hop_length, fft_length, window_type)

    # Discard front zero-padding
    recon = recon[pad_width:]

    if original_length is not None:
        recon = recon[:original_length]

    return recon

def _istft_single_channel(stft_matrix: np.ndarray, window_length: int, hop_length: int, fft_length: int, window_type: str) -> np.ndarray:
    num_bins, num_frames = stft_matrix.shape
    total_length = (num_frames - 1) * hop_length + window_length

    float_dtype = np.float64 if np.issubdtype(stft_matrix.dtype, np.complex128) else np.float32
    reconstructed = np.zeros(total_length, dtype=float_dtype)
    norm_buffer = np.zeros(total_length, dtype=float_dtype)

    w_a = get_window(window_type, window_length).astype(float_dtype)
    w_s = np.ones(window_length, dtype=float_dtype)

    # Whole-block inverse real FFT computation
    stft_frames = stft_matrix.T  # Shape: (num_frames, num_bins)
    segments = irfft_block(stft_frames, n=fft_length)  # Shape: (num_frames, fft_length)
    if fft_length > window_length:
        segments = segments[:, :window_length]

    # Vectorized Overlap-Add across all frames
    frame_starts = np.arange(num_frames) * hop_length
    sample_offsets = np.arange(window_length)
    idx = frame_starts[:, np.newaxis] + sample_offsets[np.newaxis, :]

    np.add.at(reconstructed, idx.ravel(), (segments * w_s[np.newaxis, :]).ravel())
    np.add.at(norm_buffer, idx.ravel(), (np.tile(w_a * w_s, num_frames)).ravel())

    # Normalize to compensate for windowing overlap sum
    norm_buffer[norm_buffer < 1e-12] = 1.0
    reconstructed /= norm_buffer

    return reconstructed
