from PySide6.QtWidgets import QSlider
import numpy as np
import sounddevice as sd
import gui.EqWindow
from audio.spectral_transformer import SpectralTransformer

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
        self.gains = np.ones(5, dtype=np.float32)

        self.transformer = SpectralTransformer(
            windowLength=self.windowLength,
            hopLength=self.step,
            windowType='hann'
        )

        self.bufferL = np.zeros(self.windowLength)
        self.bufferR = np.zeros(self.windowLength)

        self.frame = 0


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

        self.overlapL = np.zeros(self.windowLength - self.step)
        self.overlapR = np.zeros(self.windowLength - self.step)        
        self.stream = None

    def update_gains(self):
        num_bins = self.windowLength//2 + 1

        self.gains = np.empty(num_bins, dtype=np.float32)

        for i in range(num_bins):
            f = i / (num_bins - 1)
            self.gains[i] = self.eqWindow.interpolate(f)
        
        print(self.gains)


    def set_gain(self, start_bin, end_bin, gain):
        self.gains[start_bin:end_bin] = gain

    def set_gain_hz(self, low_hz, high_hz, gain):
        freq_resolution = self.sample_rate / self.windowLength

        start = int(low_hz / freq_resolution)
        end = int(high_hz / freq_resolution)

        self.gains[start:end] = gain
      
    
    def play_audio(self, pos=0.0):
        self.stop()
        self.frame = int(pos * self.ZxxL.shape[1])

        self.overlapL[:] = 0
        self.overlapR[:] = 0

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            blocksize=self.step,
            callback=self.callback
        )
        self.playing = True
        self.stream.start()


    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def stdtft(self, sig):
        # Fallback to SpectralTransformer for consistency/backward compatibility
        spectrum = self.transformer.analyze((sig, self.sample_rate))
        if spectrum.data.ndim == 3:
            return spectrum.data[:, :, 0]
        return spectrum.data


    



    def callback(self, outdata, frames, time, status):
        if self.frame < self.ZxxL.shape[1]:
            self.eqWindow.update_frequencies(self.ZxxL[:, self.frame])
            if not self.positionSlider.isSliderDown():
                self.positionSlider.setValue(int(self.frame / self.ZxxL.shape[1] * 100))


        block = self.next_block()

        if block is None:
            outdata[:] = 0
            playing = False
            self.positionSlider.setValue(0)
            raise sd.CallbackStop()

        outdata[:] = block




    def next_block(self):       
        if self.frame >= self.ZxxL.shape[1]:
            if len(self.bufferL) < 256: 
                return None
            
            else:
                hop =  np.column_stack((self.bufferL[:256], self.bufferR[:256]))
                self.bufferL = self.bufferL[256:]
                self.bufferR = self.bufferR[256:]


                return hop


        # Apply equalizer
        specL = self.ZxxL[:, self.frame] * self.gains
        specR = self.ZxxR[:, self.frame] * self.gains

        # Back to time domain
        winL = np.fft.irfft(specL, self.windowLength)
        winR = np.fft.irfft(specR, self.windowLength)


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




instance: AudioEngine

