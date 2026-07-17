import os
import numpy as np
import soundfile as sf
from .fft import stft, istft

class Spectrum:
    """
    A class to hold spectral representation data and the metadata 
    needed for exact reconstruction.
    """
    def __init__(self, data, sample_rate, window_length, hop_length, fft_length, original_length, window_type, num_channels):
        self.data = data  # Shape: (freq_bins, num_frames) or (freq_bins, num_frames, channels)
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.fft_length = fft_length
        self.original_length = original_length
        self.window_type = window_type
        self.num_channels = num_channels

class SpectralTransformer:
    """
    A class to analyze and synthesize audio signals in the frequency domain,
    ensuring that configurations for forward and inverse transforms are matched.
    """
    def __init__(self, baseFrequency=None, windowLength=None, hopLength=None, sampleRate=None, windowType='hann'):
        """
        Parameters:
            baseFrequency (float, optional): Target frequency resolution (bin spacing) in Hz.
            windowLength (float or int, optional): Window duration in seconds (if float) or samples (if int).
            hopLength (float or int, optional): Hop duration in seconds (if float) or samples (if int).
            sampleRate (int, optional): Default sample rate to assume if not provided in analyze.
            windowType (str): Type of window to use ('hann', 'hamming', 'rectangular').
        """
        self.baseFrequency = baseFrequency
        self.windowLength = windowLength
        self.hopLength = hopLength
        self.sampleRate = sampleRate
        self.windowType = windowType

    def analyze(self, audio_source):
        """
        Performs STFT analysis on the audio source.
        
        Parameters:
            audio_source (str or tuple or np.ndarray): 
                - String/Path: Path to an audio file to read.
                - Tuple: (signal, sample_rate)
                - np.ndarray: Signal array (uses default sample rate).
                
        Returns:
            Spectrum: The spectral representation of the audio signal.
        """
        # Resolve audio signal and sample rate
        if isinstance(audio_source, (str, bytes, os.PathLike)):
            sig, sr = sf.read(audio_source)
        elif isinstance(audio_source, (tuple, list)) and len(audio_source) == 2:
            sig, sr = audio_source
        elif isinstance(audio_source, np.ndarray):
            sig = audio_source
            sr = self.sampleRate if self.sampleRate is not None else 44100
        else:
            raise TypeError("Unsupported audio source type. Must be file path, (signal, sample_rate) tuple, or numpy array.")

        # Ensure signal is floating-point
        if not np.issubdtype(sig.dtype, np.floating):
            sig = sig.astype(np.float32)

        # Resolve window length in samples
        if self.windowLength is None:
            n_win = 512
        elif isinstance(self.windowLength, float):
            n_win = int(self.windowLength * sr)
        else:
            n_win = int(self.windowLength)

        # Resolve FFT length in samples
        if self.baseFrequency is not None:
            # We want frequency spacing fs / N_fft <= baseFrequency
            target_n_fft = int(sr / self.baseFrequency)
            n_fft = 1 << max(n_win, target_n_fft - 1).bit_length()
        else:
            n_fft = 1 << (n_win - 1).bit_length()

        # Resolve hop length in samples
        if self.hopLength is None:
            n_hop = n_win // 2
        elif isinstance(self.hopLength, float):
            n_hop = int(self.hopLength * sr)
        else:
            n_hop = int(self.hopLength)

        n_hop = min(n_hop, n_win)  # Hop length cannot exceed window length
        
        # Keep track of original length and number of channels
        original_length = sig.shape[0]
        num_channels = sig.shape[1] if sig.ndim == 2 else 1

        # Perform STFT
        data = stft(
            sig=sig,
            window_length=n_win,
            hop_length=n_hop,
            window_type=self.windowType,
            fft_length=n_fft
        )

        return Spectrum(
            data=data,
            sample_rate=sr,
            window_length=n_win,
            hop_length=n_hop,
            fft_length=n_fft,
            original_length=original_length,
            window_type=self.windowType,
            num_channels=num_channels
        )

    def synthesize(self, spectrum: Spectrum, output_path=None):
        """
        Reconstructs the audio signal from its spectral representation.
        
        Parameters:
            spectrum (Spectrum): The spectrum object.
            output_path (str, optional): File path to write the output wav file.
            
        Returns:
            np.ndarray or str: Reconstructed signal array, or output path if output_path was provided.
        """
        recon = istft(
            stft_matrix=spectrum.data,
            window_length=spectrum.window_length,
            hop_length=spectrum.hop_length,
            fft_length=spectrum.fft_length,
            window_type=spectrum.window_type,
            original_length=spectrum.original_length
        )

        if output_path is not None:
            sf.write(output_path, recon, spectrum.sample_rate)
            return output_path

        return recon
