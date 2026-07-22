from __future__ import annotations
import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from audio.spectral_transformer import SpectralTransformer
from audio.eq_curve import EqCurve
from audio.fft import ifft
from audio.metrics import evaluate_reconstruction_metrics, compute_energy

class AudioEngine:

    playing = False
    audio_loaded = False
    instance: 'AudioEngine'
    windowLength = 1024
    step = 512      # 50 % Overlap
    overlap = windowLength - step
    
    def __init__(self, eq_source=None, windowLength: int = 1024, step: int = 512, windowType: str = 'hann', fft_length: int = None):
        if getattr(AudioEngine, 'instance', None) is None:
            AudioEngine.instance = self
        self.eq_source = eq_source
        self.windowLength = windowLength
        self.step = step
        self.overlap = self.windowLength - self.step
        self.windowType = windowType
        self.fft_length = fft_length if fft_length is not None else self.windowLength

        self.stream = None
        from audio.stft import get_window
        self.w_a = get_window(self.windowType, self.windowLength)
        
        self.transformer = SpectralTransformer(
            windowLength=self.windowLength,
            hopLength=self.step,
            windowType=self.windowType
        )

        self.bufferL = np.zeros(self.windowLength)
        self.bufferR = np.zeros(self.windowLength)
        self.norm_buffer = np.zeros(self.windowLength)

        self.frame = 0
        self.sample_rate = 44100
        
        self.num_bins = self.fft_length // 2 + 1
        self.gains = np.ones(self.num_bins, dtype=np.float32)
        self.current_mag_pre = np.zeros(self.num_bins, dtype=np.float32)
        self.current_mag_post = np.zeros(self.num_bins, dtype=np.float32)
        self.waveform_max = None
        self.waveform_min = None
        self.mag_peak = 1.0
        self.db_envelope_orig = None
        self.db_envelope_recon = None
        self.loop = False
        self.sig = None
        self.spectrum = None
        self.eq_spectrum = None
        self.ZxxL = None
        self.ZxxR = None
        self.eq_ZxxL = None
        self.eq_ZxxR = None
        
        # Populate gains with initial EQ curve configuration
        self.update_gains()

    def update_gains(self, eq_curve: EqCurve = None):
        """
        Updates the frequency bin gains using the provided EqCurve or existing eq_source.
        """
        num_bins = self.windowLength // 2 + 1
        if eq_curve is not None:
            self.gains = eq_curve.evaluate_gains(num_bins, self.sample_rate)
        elif self.eq_source is not None:
            if isinstance(self.eq_source, EqCurve):
                self.gains = self.eq_source.evaluate_gains(num_bins, self.sample_rate)
            elif hasattr(self.eq_source, 'interpolate'):
                self.gains = np.empty(num_bins, dtype=np.float32)
                for i in range(num_bins):
                    f = i / (num_bins - 1) if num_bins > 1 else 0.0
                    self.gains[i] = self.eq_source.interpolate(f, self.sample_rate)
        else:
            self.gains = np.ones(num_bins, dtype=np.float32)

        if self.audio_loaded and self.ZxxL is not None:
            gains_col = self.gains[:, np.newaxis]
            self.eq_ZxxL = self.ZxxL * gains_col
            self.eq_ZxxR = self.ZxxR * gains_col
            if self.spectrum is not None:
                eq_data = np.stack([self.eq_ZxxL, self.eq_ZxxR], axis=-1)
                from audio.spectral_transformer import Spectrum
                self.eq_spectrum = Spectrum(
                    data=eq_data,
                    sample_rate=self.spectrum.sample_rate,
                    window_length=self.spectrum.window_length,
                    hop_length=self.spectrum.hop_length,
                    fft_length=self.spectrum.fft_length,
                    original_length=self.spectrum.original_length,
                    window_type=self.spectrum.window_type,
                    num_channels=self.spectrum.num_channels
                )

        self.update_equalized_envelope()

    def compare_energy(self):
        if not self.audio_loaded:
            return
        
        energy_orig = compute_energy(self.sig)
        print("Energy Original:", energy_orig)

        print("RMS Original:", np.sqrt(np.mean(self.sig ** 2)))
        

    def get_energy_comparison(self) -> str:
        if not self.audio_loaded:
            return "RMS Original: - | RMS Equalized: -"
        
        rms_orig = np.sqrt(np.mean(self.sig ** 2))
        if self.eq_ZxxL is not None:
            # Estimate via frequency domain
            total_energy = np.sum(np.abs(self.eq_ZxxL)**2) + np.sum(np.abs(self.eq_ZxxR)**2)
            # Scale approximation
            scale = 2.0 / (self.windowLength * self.ZxxL.shape[1])
            rms_recon = np.sqrt(total_energy * scale)
        else:
            rms_recon = rms_orig
        
        return f"RMS Original: {rms_orig:.4f} | RMS Equalized: {rms_recon:.4f}"
        
    def get_comparison_metrics(self) -> dict | None:
        if not self.audio_loaded:
            return None
        
        if self.eq_spectrum is not None:
            # Fast snippet synthesis (first 2 seconds max) to avoid lagging UI with full track synthesis
            frames_to_test = min(self.ZxxL.shape[1], int(2.0 * self.sample_rate / self.step))
            from audio.spectral_transformer import Spectrum
            short_eq_spec = Spectrum(
                data=self.eq_spectrum.data[:, :frames_to_test, :],
                sample_rate=self.eq_spectrum.sample_rate,
                window_length=self.eq_spectrum.window_length,
                hop_length=self.eq_spectrum.hop_length,
                fft_length=self.eq_spectrum.fft_length,
                original_length=frames_to_test * self.step,
                window_type=self.eq_spectrum.window_type,
                num_channels=self.eq_spectrum.num_channels
            )
            reconstructed = self.transformer.synthesize(short_eq_spec)
            sig_chunk = self.sig[:len(reconstructed)]
            return evaluate_reconstruction_metrics(sig_chunk, reconstructed)
        else:
            return evaluate_reconstruction_metrics(self.sig, self.sig)

    def compute_energy(self, signal):
        return compute_energy(signal)

    def load_audio(self, sig: np.ndarray, sr: int):
        self.audio_loaded = True
        
        # Ensure audio signal is stereo; duplicate if mono
        if sig.ndim == 1:
            sig = np.column_stack((sig, sig))
            
        self.sig = sig
        self.sample_rate = sr

        # Analyze using SpectralTransformer ONCE on track load
        self.spectrum = self.transformer.analyze((sig, sr))
        
        # Extract left and right channel spectra
        self.ZxxL = self.spectrum.data[:, :, 0]
        self.ZxxR = self.spectrum.data[:, :, 1]

        # Compute mag_peak for normalization
        mag_pre = (np.abs(self.ZxxL) + np.abs(self.ZxxR)) / 2.0
        self.mag_peak = float(np.max(mag_pre))
        if self.mag_peak < 1e-9:
            self.mag_peak = 1.0

        self.db_envelope_orig = self.compute_db_envelope(self.sig)

        self.overlapL = np.zeros(self.windowLength - self.step)
        self.overlapR = np.zeros(self.windowLength - self.step)        
        self.stream = None

        # Generate downsampled waveform (1000 points) for visualization
        mono_sig = np.mean(self.sig, axis=1)
        num_samples = len(mono_sig)
        num_points = 1000
        block_size = max(1, num_samples // num_points)
        self.waveform_max = np.zeros(num_points, dtype=np.float32)
        self.waveform_min = np.zeros(num_points, dtype=np.float32)
        for i in range(num_points):
            start = i * block_size
            end = min(num_samples, (i + 1) * block_size)
            if start < end:
                self.waveform_max[i] = np.max(mono_sig[start:end])
                self.waveform_min[i] = np.min(mono_sig[start:end])

        self.update_gains()

    def compute_db_envelope(self, sig, num_points=1000):
        if sig is None:
            return np.zeros(num_points, dtype=np.float32)
        mono = np.mean(sig, axis=1) if sig.ndim > 1 else sig
        num_samples = len(mono)
        block_size = max(1, num_samples // num_points)
        envelope = np.zeros(num_points, dtype=np.float32)
        for i in range(num_points):
            start = i * block_size
            end = min(num_samples, (i + 1) * block_size)
            if start < end:
                block = mono[start:end]
                rms = np.sqrt(np.mean(block ** 2))
                envelope[i] = 20 * np.log10(rms + 1e-6)
        return envelope

    def update_equalized_envelope(self):
        if not self.audio_loaded or self.eq_ZxxL is None:
            return
        
        # Fast frequency-domain envelope approximation to avoid synthesizing the entire track
        energy_mono = (np.sum(np.abs(self.eq_ZxxL)**2, axis=0) + np.sum(np.abs(self.eq_ZxxR)**2, axis=0)) / 2.0
        
        # Scale to rough time domain amplitude via Parseval mapping
        scale = 2.0 / self.windowLength
        rms_frames = np.sqrt(energy_mono / self.windowLength) * scale
        
        num_frames = len(rms_frames)
        num_points = min(1000, num_frames)
        block_size = max(1, num_frames // num_points)
        
        envelope = np.zeros(1000, dtype=np.float32)
        for i in range(1000):
            start = i * block_size
            end = min(num_frames, (i + 1) * block_size)
            if start < end:
                block_rms = np.mean(rms_frames[start:end])
                envelope[i] = 20 * np.log10(block_rms + 1e-6)
                
        # Align roughly with original track dB range
        if self.db_envelope_orig is not None and np.max(self.db_envelope_orig) > -80:
             diff = np.max(envelope) - np.max(self.db_envelope_orig)
             envelope -= diff
             
        self.db_envelope_recon = envelope

    def set_gain(self, start_bin: int, end_bin: int, gain: float):
        self.gains[start_bin:end_bin] = gain

    def set_gain_hz(self, low_hz: float, high_hz: float, gain: float):
        freq_resolution = self.sample_rate / self.windowLength
        start = int(low_hz / freq_resolution)
        end = int(high_hz / freq_resolution)
        self.gains[start:end] = gain
    
    def play_audio(self, pos=None) -> bool:
        if not self.audio_loaded:
            return False
        
        target_frame = self.frame if pos is None else int(pos * self.ZxxL.shape[1])
        if target_frame >= self.ZxxL.shape[1]:
            target_frame = 0
            
        self.stop()
        self.frame = target_frame

        self.overlapL[:] = 0
        self.overlapR[:] = 0

        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                blocksize=self.step,
                callback=self.callback
            )
            self.stream.start()
            self.playing = True
            return True
        except Exception as e:
            print(f"Error starting audio stream: {e}", file=sys.stderr)
            self.stream = None
            self.playing = False
            return False

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.playing = False
        if hasattr(self, 'current_mag_pre') and self.current_mag_pre is not None:
            self.current_mag_pre.fill(0)
            self.current_mag_post.fill(0)
        self.bufferL = np.zeros(self.windowLength)
        self.bufferR = np.zeros(self.windowLength)
        self.norm_buffer = np.zeros(self.windowLength)
        if hasattr(self, 'buffers_ch'):
            for b in self.buffers_ch:
                b.fill(0)

    def load_stft(self, stft_matrix: np.ndarray):
        """
        Loads pre-computed STFT matrix into AudioEngine for synthesis / streaming.
        """
        if stft_matrix.ndim == 3:
            self.num_channels = stft_matrix.shape[2]
            self.Zxx_channels = [stft_matrix[:, :, c] for c in range(self.num_channels)]
            self.ZxxL = stft_matrix[:, :, 0]
            self.ZxxR = stft_matrix[:, :, 1] if self.num_channels > 1 else stft_matrix[:, :, 0]
        else:
            self.num_channels = 1
            self.Zxx_channels = [stft_matrix]
            self.ZxxL = stft_matrix
            self.ZxxR = stft_matrix

        num_bins = stft_matrix.shape[0]
        self.num_bins = num_bins
        self.gains = np.ones(num_bins, dtype=np.float32)
        self.eq_ZxxL = None
        self.eq_ZxxR = None
        self.frame = 0
        self.bufferL = np.zeros(self.windowLength)
        self.bufferR = np.zeros(self.windowLength)
        self.norm_buffer = np.zeros(self.windowLength)
        if self.num_channels > 2:
            self.buffers_ch = [np.zeros(self.windowLength) for _ in range(self.num_channels)]
        self.audio_loaded = True

    def istft(self) -> np.ndarray:
        """
        Reconstructs time-domain signal by iterating next_block() until completion.
        """
        self.frame = 0
        self.bufferL = np.zeros(self.windowLength)
        self.bufferR = np.zeros(self.windowLength)
        self.norm_buffer = np.zeros(self.windowLength)
        if hasattr(self, 'num_channels') and self.num_channels > 2:
            self.buffers_ch = [np.zeros(self.windowLength) for _ in range(self.num_channels)]

        reconstructed = []
        nxt = self.next_block()

        while nxt is not None:
            reconstructed.append(nxt)
            nxt = self.next_block()

        if len(reconstructed) == 0:
            return np.zeros((0, getattr(self, 'num_channels', 2)))

        return np.concatenate(reconstructed, axis=0)

    def stdtft(self, sig):
        spectrum = self.transformer.analyze((sig, self.sample_rate))
        if spectrum.data.ndim == 3:
            return spectrum.data[:, :, 0]
        return spectrum.data

    def callback(self, outdata, frames, time, status):
        block = self.next_block()
        if block is None:
            outdata[:] = 0
            raise sd.CallbackStop()
        outdata[:] = block

    def _ifft_window(self, spec: np.ndarray) -> np.ndarray:
        full_spec = np.concatenate([spec, np.conjugate(spec[1:-1][::-1])])
        return np.real(ifft(full_spec))[:self.windowLength]

    def next_block(self):
        total_frames = self.ZxxL.shape[1]
        num_ch = getattr(self, 'num_channels', 2)

        # 1. End-of-signal / looping check
        if self.frame >= total_frames:
            if self.loop:
                self.frame = 0
                self.bufferL.fill(0)
                self.bufferR.fill(0)
                if hasattr(self, 'norm_buffer'):
                    self.norm_buffer.fill(0)
                if hasattr(self, 'buffers_ch'):
                    for b in self.buffers_ch:
                        b.fill(0)
            else:
                if hasattr(self, 'current_mag_pre') and self.current_mag_pre is not None:
                    self.current_mag_pre.fill(0)
                    self.current_mag_post.fill(0)
                tail_has_data = np.any(np.abs(self.buffers_ch[0]) > 1e-12) if num_ch > 2 else (np.any(np.abs(self.bufferL) > 1e-12) or np.any(np.abs(self.bufferR) > 1e-12))
                if not tail_has_data:
                    self.playing = False
                    return None

        # 2. Accumulate current frame if available
        if self.frame < total_frames:
            if num_ch > 2:
                for c in range(num_ch):
                    spec_c = self.Zxx_channels[c][:, self.frame] * (1.0 if self.eq_ZxxL is not None else self.gains)
                    self.buffers_ch[c] += self._ifft_window(spec_c)
            else:
                if self.eq_ZxxL is not None and self.eq_ZxxR is not None:
                    specL, specR = self.eq_ZxxL[:, self.frame], self.eq_ZxxR[:, self.frame]
                else:
                    specL = self.ZxxL[:, self.frame] * self.gains
                    specR = self.ZxxR[:, self.frame] * self.gains

                if hasattr(self, 'current_mag_pre') and self.current_mag_pre is not None:
                    self.current_mag_pre = (np.abs(self.ZxxL[:, self.frame]) + np.abs(self.ZxxR[:, self.frame])) / 2.0
                    self.current_mag_post = (np.abs(specL) + np.abs(specR)) / 2.0

                self.bufferL += self._ifft_window(specL)
                self.bufferR += self._ifft_window(specR)

            self.frame += 1

        # 3. Extract unnormalized hop & shift buffers
        if num_ch > 2:
            out_channels = []
            for c in range(num_ch):
                out_c = self.buffers_ch[c][:self.step].copy()
                self.buffers_ch[c][:-self.step] = self.buffers_ch[c][self.step:]
                self.buffers_ch[c][-self.step:] = 0
                out_channels.append(out_c)
            out_hop = np.column_stack(out_channels)
        else:
            outL = self.bufferL[:self.step].copy()
            outR = self.bufferR[:self.step].copy()
            out_hop = np.column_stack((outL, outR))

            self.bufferL[:-self.step] = self.bufferL[self.step:]
            self.bufferL[-self.step:] = 0
            self.bufferR[:-self.step] = self.bufferR[self.step:]
            self.bufferR[-self.step:] = 0

        return out_hop

    def export_audio(self, output_path: str) -> bool:
        if not self.audio_loaded or self.eq_spectrum is None:
            return False
        try:
            reconstructed = self.transformer.synthesize(self.eq_spectrum)
            sf.write(output_path, reconstructed, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error exporting audio: {e}", file=sys.stderr)
            return False

instance: AudioEngine
