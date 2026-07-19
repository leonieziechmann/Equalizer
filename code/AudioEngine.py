from __future__ import annotations
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QSlider
import numpy as np
import sounddevice as sd
from audio.spectral_transformer import SpectralTransformer
from audio import irfft

if TYPE_CHECKING:
    from gui.EqWindow import EqWindow

class AudioEngine():

    playing = False
    audio_loaded=False
    instance: 'AudioEngine'
    windowLength = 512
    step = 256      # 50 % Overlap
    overlap = windowLength - step
    positionSlider: QSlider
    

    def __init__(self, eqWindow: EqWindow):
        self.eqWindow = eqWindow
        
        self.transformer = SpectralTransformer(
            windowLength=self.windowLength,
            hopLength=self.step,
            windowType='hann'
        )

        self.bufferL = np.zeros(self.windowLength)
        self.bufferR = np.zeros(self.windowLength)

        self.frame = 0
        self.sample_rate = 44100
        
        self.num_bins = self.windowLength // 2 + 1
        self.gains = np.ones(self.num_bins, dtype=np.float32)
        self.current_mag_pre = np.zeros(self.num_bins, dtype=np.float32)
        self.current_mag_post = np.zeros(self.num_bins, dtype=np.float32)
        self.waveform_max = None
        self.waveform_min = None
        self.mag_peak = 1.0
        self.db_envelope_orig = None
        self.db_envelope_recon = None
        
        # Populate gains with initial EQ curve configuration
        self.update_gains()
        self.loop = False


    def compare_energy(self):
        if not self.audio_loaded:
            return
        
        # Analyze and synthesize directly using SpectralTransformer to verify reconstruction
        spectrum = self.transformer.analyze((self.sig, self.sample_rate))
        reconstructed = self.transformer.synthesize(spectrum)
        
        energyOriginal = self.compute_energy(self.sig)
        energyRecons = self.compute_energy(reconstructed)
        print("Energy Original:", energyOriginal)
        print("Energy Reconstructed:", energyRecons)
        print()

        print("RMS Original:", np.sqrt(np.mean(self.sig ** 2)))
        print("RMS Reconstructed:", np.sqrt(np.mean(reconstructed ** 2)))
        

    def get_energy_comparison(self) -> str:
        if not self.audio_loaded:
            return "RMS Original: - | RMS Equalized: -"
        
        # Analyze using SpectralTransformer
        spectrum = self.transformer.analyze((self.sig, self.sample_rate))
        
        # Apply the current gains
        gains_expanded = self.gains[:, np.newaxis, np.newaxis]
        spectrum.data = spectrum.data * gains_expanded
        
        # Synthesize equalized audio
        reconstructed = self.transformer.synthesize(spectrum)
        
        rms_orig = np.sqrt(np.mean(self.sig ** 2))
        rms_recon = np.sqrt(np.mean(reconstructed ** 2))
        
        return f"RMS Original: {rms_orig:.4f} | RMS Equalized: {rms_recon:.4f}"
        
    def get_comparison_metrics(self):
        if not self.audio_loaded:
            return None
        
        # Analyze using SpectralTransformer
        spectrum = self.transformer.analyze((self.sig, self.sample_rate))
        
        # Apply the current gains
        if spectrum.data.ndim == 3:
            gains_expanded = self.gains[:, np.newaxis, np.newaxis]
        else:
            gains_expanded = self.gains[:, np.newaxis]
        
        spectrum.data = spectrum.data * gains_expanded
        
        # Synthesize equalized audio
        reconstructed = self.transformer.synthesize(spectrum)
        
        # Mix to mono for metrics calculation if multi-channel
        if self.sig.ndim > 1:
            orig_mono = np.mean(self.sig, axis=1)
        else:
            orig_mono = self.sig
            
        if reconstructed.ndim > 1:
            recon_mono = np.mean(reconstructed, axis=1)
        else:
            recon_mono = reconstructed
            
        # Align lengths in case of minor overlap/synthesis windowing differences
        min_len = min(len(orig_mono), len(recon_mono))
        orig_mono = orig_mono[:min_len]
        recon_mono = recon_mono[:min_len]
        
        # 1. RMS
        rms_orig = float(np.sqrt(np.mean(orig_mono ** 2)))
        rms_recon = float(np.sqrt(np.mean(recon_mono ** 2)))
        
        # 2. Energy
        energy_orig = float(self.compute_energy(orig_mono))
        energy_recon = float(self.compute_energy(recon_mono))
        
        # 3. Peak Amplitude
        peak_orig = float(np.max(np.abs(orig_mono)))
        peak_recon = float(np.max(np.abs(recon_mono)))
        
        # 4. Crest Factor
        crest_orig = float(peak_orig / rms_orig) if rms_orig > 0 else 0.0
        crest_recon = float(peak_recon / rms_recon) if rms_recon > 0 else 0.0
        
        # 5. Correlation (Pearson correlation coefficient)
        std_orig = np.std(orig_mono)
        std_recon = np.std(recon_mono)
        if std_orig > 0 and std_recon > 0:
            correlation = float(np.corrcoef(orig_mono, recon_mono)[0, 1])
        else:
            correlation = 0.0
            
        # 6. MSE
        mse = float(np.mean((orig_mono - recon_mono) ** 2))
        
        # 7. MAE
        mae = float(np.mean(np.abs(orig_mono - recon_mono)))
        
        # 8. SDR (Signal to Distortion Ratio)
        noise_power = np.sum((orig_mono - recon_mono) ** 2)
        signal_power = np.sum(orig_mono ** 2)
        if noise_power > 0 and signal_power > 0:
            sdr = float(10 * np.log10(signal_power / noise_power))
        else:
            sdr = float('inf') if noise_power == 0 else -float('inf')
            
        return {
            'rms_orig': rms_orig,
            'rms_recon': rms_recon,
            'energy_orig': energy_orig,
            'energy_recon': energy_recon,
            'peak_orig': peak_orig,
            'peak_recon': peak_recon,
            'crest_orig': crest_orig,
            'crest_recon': crest_recon,
            'correlation': correlation,
            'mse': mse,
            'mae': mae,
            'sdr': sdr
        }

    def compute_energy(self, signal):
        squared = signal ** 2
        return np.sum(squared)


    def load_audio(self, sig, sr):
        self.audio_loaded = True
        
        # Ensure audio signal is stereo; duplicate if mono
        if sig.ndim == 1:
            sig = np.column_stack((sig, sig))
            
        self.sig = sig
        self.sample_rate = sr

        # Analyze using the configured SpectralTransformer
        spectrum = self.transformer.analyze((sig, sr))
        
        # Extract left and right channel spectra
        self.ZxxL = spectrum.data[:, :, 0]
        self.ZxxR = spectrum.data[:, :, 1]

        # Compute mag_peak for normalization
        mag_pre = (np.abs(self.ZxxL) + np.abs(self.ZxxR)) / 2.0
        self.mag_peak = np.max(mag_pre)
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

        # Ensure gains are aligned with the audio sample rate configuration
        self.update_gains()

    def update_gains(self):
        num_bins = self.windowLength//2 + 1

        self.gains = np.empty(num_bins, dtype=np.float32)

        for i in range(num_bins):
            f = i / (num_bins - 1)
            self.gains[i] = self.eqWindow.interpolate(f, self.sample_rate)
        
        print(self.gains)
        self.update_equalized_envelope()

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
        if not self.audio_loaded:
            return
        
        # Analyze using SpectralTransformer
        spectrum = self.transformer.analyze((self.sig, self.sample_rate))
        
        # Apply the current gains
        if spectrum.data.ndim == 3:
            gains_expanded = self.gains[:, np.newaxis, np.newaxis]
        else:
            gains_expanded = self.gains[:, np.newaxis]
        
        spectrum.data = spectrum.data * gains_expanded
        
        # Synthesize equalized audio
        reconstructed = self.transformer.synthesize(spectrum)
        
        # Compute the equalized envelope
        self.db_envelope_recon = self.compute_db_envelope(reconstructed)


    def set_gain(self, start_bin, end_bin, gain):
        self.gains[start_bin:end_bin] = gain

    def set_gain_hz(self, low_hz, high_hz, gain):
        freq_resolution = self.sample_rate / self.windowLength

        start = int(low_hz / freq_resolution)
        end = int(high_hz / freq_resolution)

        self.gains[start:end] = gain
      
    
    def play_audio(self, pos=None):
        if not self.audio_loaded:
            return False
        import sys
        
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

    def stdtft(self, sig):
        # Fallback to SpectralTransformer for consistency/backward compatibility
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




    def next_block(self):       
        if self.frame >= self.ZxxL.shape[1]:
            if self.loop:
                self.frame = 0
                self.bufferL.fill(0)
                self.bufferR.fill(0)
            else:
                if hasattr(self, 'current_mag_pre') and self.current_mag_pre is not None:
                    self.current_mag_pre.fill(0)
                    self.current_mag_post.fill(0)
                if len(self.bufferL) < 256: 
                    self.playing = False
                    return None
                
                else:
                    hop =  np.column_stack((self.bufferL[:256], self.bufferR[:256]))
                    self.bufferL = self.bufferL[256:]
                    self.bufferR = self.bufferR[256:]


                    return hop


        # Apply equalizer
        specL = self.ZxxL[:, self.frame] * self.gains
        specR = self.ZxxR[:, self.frame] * self.gains

        # Save magnitude spectra for visualizer
        self.current_mag_pre = (np.abs(self.ZxxL[:, self.frame]) + np.abs(self.ZxxR[:, self.frame])) / 2.0
        self.current_mag_post = (np.abs(specL) + np.abs(specR)) / 2.0

        # Back to time domain
        winL = irfft(specL, self.windowLength)
        winR = irfft(specR, self.windowLength)


        # add into synthesis buffer
        self.bufferL += winL
        self.bufferR += winR        

        # output first hop
        outL = self.bufferL[:self.step].copy()
        outR = self.bufferR[:self.step].copy()      

        # shift buffer
        self.bufferL[:-self.step] = self.bufferL[self.step:]
        self.bufferR[:-self.step] = self.bufferR[self.step:]        
        self.bufferL[-self.step:] = 0
        self.bufferR[-self.step:] = 0       
        self.frame += 1     

        return np.column_stack((outL, outR))

    def export_audio(self, output_path):
        if not self.audio_loaded:
            return False
        try:
            spectrum = self.transformer.analyze((self.sig, self.sample_rate))
            
            if spectrum.data.ndim == 3:
                gains_expanded = self.gains[:, np.newaxis, np.newaxis]
            else:
                gains_expanded = self.gains[:, np.newaxis]
                
            spectrum.data = spectrum.data * gains_expanded
            reconstructed = self.transformer.synthesize(spectrum)
            
            import soundfile as sf
            sf.write(output_path, reconstructed, self.sample_rate)
            return True
        except Exception as e:
            import sys
            print(f"Error exporting audio: {e}", file=sys.stderr)
            return False




instance: AudioEngine

