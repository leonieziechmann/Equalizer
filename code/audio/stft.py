import numpy as np
from .fft import fft

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
    Computes Short-Time Fourier Transform (STFT) of a 1D (mono) or (stereo/multichannel) signal.
    
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

    num_bins = fft_length // 2 + 1
    complex_dtype = np.complex128 if np.issubdtype(sig.dtype, np.float64) else np.complex64
    stft_matrix = np.zeros((num_bins, num_frames, num_channels), dtype=complex_dtype)

    for i in range(num_frames):
        start = i * hop_length
        end = start + window_length
        # Multiply each channel by analysis window
        segment = sig_padded[start:end, :] * w_a[:, np.newaxis]
        
        # Zero-pad segment to fft_length if necessary
        if fft_length > window_length:
            seg_padded = np.pad(segment, ((0, fft_length - window_length), (0, 0)), mode='constant')
        else:
            seg_padded = segment[:fft_length, :]

        for ch in range(num_channels):
            full_fft = fft(seg_padded[:, ch])
            stft_matrix[:, i, ch] = full_fft[:num_bins]

    if not is_multichannel:
        stft_matrix = stft_matrix[:, :, 0]

    return stft_matrix

def istft(stft_matrix: np.ndarray, window_length: int, hop_length: int, fft_length: int, window_type: str = 'hann', original_length: int = None) -> np.ndarray:
    """
    Performs Inverse Short-Time Fourier Transform (ISTFT) using AudioEngine.next_block
    to guarantee exact signal reconstruction and unified processing.
    
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
    from AudioEngine import AudioEngine

    prev_instance = getattr(AudioEngine, 'instance', None)
    engine = AudioEngine(
        windowLength=window_length,
        step=hop_length,
        windowType=window_type,
        fft_length=fft_length
    )
    engine.load_stft(stft_matrix)

    recon = engine.istft()
    if prev_instance is not None:
        AudioEngine.instance = prev_instance

    pad_width = window_length

    is_mono = stft_matrix.ndim == 2 or (stft_matrix.ndim == 3 and stft_matrix.shape[2] == 1)
    if is_mono:
        recon = recon[:, 0]

    # Discard front zero-padding
    recon = recon[pad_width:]

    if original_length is not None:
        recon = recon[:original_length]

    return recon
