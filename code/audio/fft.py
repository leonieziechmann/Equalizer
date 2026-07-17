import numpy as np

def get_window(window_type, window_length):
    """
    Returns a window function of the specified type and length.
    """
    if window_type in ('hann', 'hanning'):
        return np.hanning(window_length)
    elif window_type == 'hamming':
        return np.hamming(window_length)
    elif window_type in ('rectangular', 'boxcar') or window_type is None:
        return np.ones(window_length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

def stft(sig, window_length, hop_length, window_type='hann', fft_length=None):
    """
    Computes the Short-Time Fourier Transform of a 1D or 2D signal.
    
    Parameters:
        sig (np.ndarray): Input signal, shape (samples,) or (samples, channels).
        window_length (int): Length of the window in samples.
        hop_length (int): Step size between successive windows in samples.
        window_type (str): Type of window to use ('hann', 'hamming', 'rectangular').
        fft_length (int, optional): FFT size. Defaults to the next power of 2 >= window_length.
        
    Returns:
        np.ndarray: Complex STFT coefficients, shape (freq_bins, num_frames) or (freq_bins, num_frames, channels).
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

    # Boundary padding: pad by window_length at both ends to ensure steady-state overlap-add
    # for all original samples, eliminating edge effects entirely.
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

    num_bins = fft_length // 2 + 1
    complex_dtype = np.complex128 if np.issubdtype(sig.dtype, np.float64) else np.complex64
    stft_matrix = np.zeros((num_bins, num_frames, num_channels), dtype=complex_dtype)

    for i in range(num_frames):
        start = i * hop_length
        end = start + window_length
        # Multiply each channel by the analysis window
        segment = sig_padded[start:end, :] * w_a[:, np.newaxis]
        # FFT along the sample axis (axis 0)
        stft_matrix[:, i, :] = np.fft.rfft(segment, n=fft_length, axis=0)

    if not is_multichannel:
        stft_matrix = stft_matrix[:, :, 0]

    return stft_matrix

def istft(stft_matrix, window_length, hop_length, fft_length, window_type='hann', original_length=None):
    """
    Performs the Inverse Short-Time Fourier Transform with overlap-add normalization
    to guarantee perfect reconstruction.
    
    Parameters:
        stft_matrix (np.ndarray): Complex STFT coefficients.
        window_length (int): Length of the window in samples.
        hop_length (int): Step size between successive windows in samples.
        fft_length (int): FFT size.
        window_type (str): Type of window that was used.
        original_length (int, optional): Truncates the output to this length.
        
    Returns:
        np.ndarray: Reconstructed signal.
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

    # Discard front padding
    recon = recon[pad_width:]

    if original_length is not None:
        recon = recon[:original_length]

    return recon

def _istft_single_channel(stft_matrix, window_length, hop_length, fft_length, window_type):
    num_bins, num_frames = stft_matrix.shape
    total_length = (num_frames - 1) * hop_length + window_length

    float_dtype = np.float64 if np.issubdtype(stft_matrix.dtype, np.complex128) else np.float32
    reconstructed = np.zeros(total_length, dtype=float_dtype)
    norm_buffer = np.zeros(total_length, dtype=float_dtype)

    # Get analysis window (used to compute the normalization denominator)
    w_a = get_window(window_type, window_length).astype(float_dtype)
    # Synthesis window: we use rectangular (all ones) for standard overlap-add reconstruction
    # to match original behavior, but normalized correctly by the analysis window.
    w_s = np.ones(window_length, dtype=float_dtype)

    for i in range(num_frames):
        start = i * hop_length
        end = start + window_length
        
        spec = stft_matrix[:, i]
        # Perform inverse FFT
        segment = np.fft.irfft(spec, n=fft_length)
        # Truncate if FFT length is larger than the window length
        segment = segment[:window_length]
        
        # Accumulate overlap-add
        reconstructed[start:end] += segment * w_s
        norm_buffer[start:end] += w_a * w_s

    # Normalize to compensate for windowing overlap
    # Protect against division by zero at boundary points where norm is 0
    norm_buffer[norm_buffer < 1e-12] = 1.0
    reconstructed /= norm_buffer

    return reconstructed
