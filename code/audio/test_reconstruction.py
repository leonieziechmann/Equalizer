import os
import sys
import numpy as np
import soundfile as sf

# Add the parent directory (code) to the python path so we can import audio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio import (
    SpectralTransformer,
    EqCurve,
    stft,
    istft,
    fft,
    ifft,
    rfft,
    irfft,
    evaluate_reconstruction_metrics
)
from audio.fft import fft_block, rfft_block, irfft_block

def test_fft_primitives():
    print("Testing FFT and IFFT primitives...")
    x = np.random.uniform(-1.0, 1.0, 64)
    x_complex = x + 0.5j * np.roll(x, 1)
    
    # Complex FFT & IFFT roundtrip
    X = fft(x_complex)
    x_recon = ifft(X)
    assert np.allclose(x_complex, x_recon), "Complex FFT/IFFT roundtrip failed"
    
    # Real FFT & IRFFT roundtrip
    X_real = rfft(x)
    x_real_recon = irfft(X_real, len(x))
    assert np.allclose(x, x_real_recon), "Real FFT/IRFFT roundtrip failed"
    print("FFT and IFFT primitives: PASSED")

def test_fft_block_primitives():
    print("Testing Compute Shader / Vectorized Block FFT primitives...")
    num_signals = 16
    N = 128
    x_block = np.random.uniform(-1.0, 1.0, (num_signals, N))
    x_complex_block = x_block + 0.5j * np.roll(x_block, 1, axis=1)

    # Complex Block FFT & IFFT roundtrip
    X_block = fft_block(x_complex_block, is_inverse=False)
    x_recon_block = fft_block(X_block, is_inverse=True)
    assert np.allclose(x_complex_block, x_recon_block), "Complex Block FFT/IFFT roundtrip failed"

    # Real Block FFT & IRFFT roundtrip
    X_real_block = rfft_block(x_block, n=N)
    x_real_recon_block = irfft_block(X_real_block, n=N)
    assert np.allclose(x_block, x_real_recon_block), "Real Block FFT/IRFFT roundtrip failed"

    # Verify against numpy reference
    np_X_real = np.fft.rfft(x_block, n=N, axis=-1)
    assert np.allclose(X_real_block, np_X_real), "Block Real FFT mismatch with numpy reference"

    print("Compute Shader / Vectorized Block FFT primitives: PASSED")

def test_roundtrip_1d():
    print("Testing 1D signal perfect reconstruction...")
    fs = 44100
    t = np.linspace(0, 5, 5 * fs, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    
    st = SpectralTransformer(baseFrequency=5, windowLength=0.1) # 100ms window, 5Hz resolution
    spectrum = st.analyze((sig, fs))
    recon = st.synthesize(spectrum)
    
    assert sig.shape == recon.shape, f"Shape mismatch: {sig.shape} vs {recon.shape}"
    
    mse = np.mean((sig - recon) ** 2)
    print(f"1D MSE: {mse:.2e}")
    assert mse < 1e-12, f"Perfect reconstruction failed, MSE = {mse}"
    print("1D Signal Perfect Reconstruction: PASSED")

def test_roundtrip_2d():
    print("Testing 2D (stereo) signal perfect reconstruction...")
    fs = 44100
    t = np.linspace(0, 5, 5 * fs, endpoint=False)
    sig_l = 0.5 * np.sin(2 * np.pi * 440 * t)
    sig_r = 0.5 * np.cos(2 * np.pi * 440 * t)
    sig = np.column_stack((sig_l, sig_r))
    
    st = SpectralTransformer(windowLength=1024, hopLength=512)
    spectrum = st.analyze((sig, fs))
    recon = st.synthesize(spectrum)
    
    assert sig.shape == recon.shape, f"Shape mismatch: {sig.shape} vs {recon.shape}"
    
    mse = np.mean((sig - recon) ** 2)
    print(f"2D MSE: {mse:.2e}")
    assert mse < 1e-12, f"Perfect reconstruction failed, MSE = {mse}"
    print("2D Signal Perfect Reconstruction: PASSED")

def test_file_roundtrip():
    git_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    mp3_path = os.path.join(git_root, 'test-track.mp3')
    
    if not os.path.exists(mp3_path):
        print(f"Could not find test track at {mp3_path}, skipping file test.")
        return

    print(f"Testing roundtrip on file: {mp3_path}")
    st = SpectralTransformer(baseFrequency=5, windowLength=0.1)
    spectrum = st.analyze(mp3_path)
    
    sig, sr = sf.read(mp3_path)
    recon = st.synthesize(spectrum)
    
    if sig.ndim == 2:
        assert recon.ndim == 2, "Reconstructed signal should be stereo"
        assert sig.shape[1] == recon.shape[1], "Channel mismatch"
    
    assert sig.shape[0] == recon.shape[0], f"Length mismatch: {sig.shape[0]} vs {recon.shape[0]}"
    
    mse = np.mean((sig - recon) ** 2)
    print(f"File MSE: {mse:.2e}")
    assert mse < 1e-12, f"File perfect reconstruction failed, MSE = {mse}"
    print("File Roundtrip: PASSED")

def test_eq_curve():
    print("Testing EqCurve interpolation and bin gain evaluation...")
    curve = EqCurve()
    
    f_low = 50.0 / 22050.0
    val_low = curve.interpolate(f_low)
    expected_low = 10.0 ** (-34.0 / 20.0)
    assert abs(val_low - expected_low) < 1e-6, f"Expected {expected_low}, got {val_low}"
    
    f_high = 1000.0 / 22050.0
    val_high = curve.interpolate(f_high)
    expected_high = 10.0 ** (-6.4 / 20.0)
    assert abs(val_high - expected_high) < 1e-6, f"Expected {expected_high}, got {val_high}"

    custom_points = [(0.0, 1.0), (1.0, 0.0)]
    curve_custom = EqCurve(custom_points)
    
    f_20 = 20.0 / 22050.0
    assert abs(curve_custom.interpolate(f_20) - 0.0001) < 1e-6
    
    f_20k = 20000.0 / 22050.0
    assert abs(curve_custom.interpolate(f_20k) - 10.0**(12.0/20.0)) < 1e-6
    
    gains = curve_custom.evaluate_gains(257, 44100)
    assert len(gains) == 257
    assert isinstance(gains, np.ndarray)
    
    print("EqCurve Interpolation & Evaluation: PASSED")

def test_metrics():
    print("Testing audio metrics computation...")
    sig = np.sin(np.linspace(0, 10, 1000))
    recon = sig * 0.99
    
    metrics = evaluate_reconstruction_metrics(sig, recon)
    assert 'rms_orig' in metrics
    assert 'sdr' in metrics
    assert metrics['sdr'] > 30.0  # High SDR for tiny gain difference
    print("Audio Metrics Computation: PASSED")

def test_headless_eq_processing():
    print("Testing headless EQ processing logic...")
    fs = 44100
    t = np.linspace(0, 1, fs, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    transformer = SpectralTransformer(windowLength=512, hopLength=256, windowType='hann')
    spectrum = transformer.analyze((sig, fs))
    
    y_raw = 1.0 - (20.0 * np.log10(0.5) + 80.0) / 92.0
    curve = EqCurve([(0.5, y_raw)])
    
    spectrum = transformer.apply_equalizer(spectrum, curve)
    recon = transformer.synthesize(spectrum)
    
    expected_recon = sig * 0.5
    mse = np.mean((expected_recon - recon) ** 2)
    print(f"Headless EQ Processing MSE: {mse:.2e}")
    assert mse < 1e-12, f"Headless EQ processing failed, MSE = {mse}"
    print("Headless EQ Processing: PASSED")

def test_audio_engine():
    print("Testing AudioEngine seek, restart, and loop behaviors...")
    from AudioEngine import AudioEngine
    import unittest.mock as mock
    
    class MockStream:
        def __init__(self, *args, **kwargs):
            self.active = True
        def start(self):
            pass
        def stop(self):
            self.active = False
        def close(self):
            pass

    engine = AudioEngine(EqCurve([(0.0, 0.5)]))
    
    fs = 44100
    sig = np.random.uniform(-0.1, 0.1, (fs * 5, 2))
    
    engine.load_audio(sig, fs)
    assert engine.audio_loaded
    assert engine.ZxxL.shape[1] > 0
    
    with mock.patch('sounddevice.OutputStream', side_effect=MockStream):
        success = engine.play_audio(pos=0.5)
        assert success
        expected_seek_frame = int(0.5 * engine.ZxxL.shape[1])
        assert engine.frame == expected_seek_frame
        
        block = engine.next_block()
        assert block is not None
        assert block.shape == (engine.step, 2)
        assert engine.frame == expected_seek_frame + 1
        engine.stop()
        
        engine.play_audio(pos=0.99)
        engine.frame = engine.ZxxL.shape[1]
        
        for _ in range(10):
            blk = engine.next_block()
            if blk is None:
                break
                
        assert not engine.playing
        
        success = engine.play_audio()
        assert success
        assert engine.playing
        assert engine.frame == 0
        
        block = engine.next_block()
        assert block is not None
        assert block.shape == (engine.step, 2)
        engine.stop()
        
        engine.loop = True
        engine.play_audio()
        engine.frame = engine.ZxxL.shape[1]
        
        block = engine.next_block()
        assert block is not None
        assert block.shape == (engine.step, 2)
        assert engine.frame == 1
        
        engine.stop()
        
    print("AudioEngine seek, restart, and loop behaviors: PASSED")

if __name__ == '__main__':
    test_fft_primitives()
    print("-" * 40)
    test_fft_block_primitives()
    print("-" * 40)
    test_roundtrip_1d()
    print("-" * 40)
    test_roundtrip_2d()
    print("-" * 40)
    test_file_roundtrip()
    print("-" * 40)
    test_eq_curve()
    print("-" * 40)
    test_metrics()
    print("-" * 40)
    test_headless_eq_processing()
    print("-" * 40)
    test_audio_engine()
    print("-" * 40)
    print("All audio analysis, EQ, and reconstruction tests passed successfully!")
