import os
import sys
import numpy as np
import soundfile as sf

# Add the parent directory (code) to the python path so we can import audio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio.spectral_transformer import SpectralTransformer

def test_roundtrip_1d():
    print("Testing 1D signal perfect reconstruction...")
    # Generate 1D test signal: 5 seconds of sines at 44100 Hz
    fs = 44100
    t = np.linspace(0, 5, 5 * fs, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    
    st = SpectralTransformer(baseFrequency=5, windowLength=0.1) # 100ms window, 5Hz resolution
    spectrum = st.analyze((sig, fs))
    recon = st.synthesize(spectrum)
    
    # Check shape
    assert sig.shape == recon.shape, f"Shape mismatch: {sig.shape} vs {recon.shape}"
    
    # Check MSE
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
    
    st = SpectralTransformer(windowLength=1024, hopLength=512) # Standard sample-based config
    spectrum = st.analyze((sig, fs))
    recon = st.synthesize(spectrum)
    
    assert sig.shape == recon.shape, f"Shape mismatch: {sig.shape} vs {recon.shape}"
    
    mse = np.mean((sig - recon) ** 2)
    print(f"2D MSE: {mse:.2e}")
    assert mse < 1e-12, f"Perfect reconstruction failed, MSE = {mse}"
    print("2D Signal Perfect Reconstruction: PASSED")

def test_file_roundtrip():
    # Check if test-track.mp3 exists in the root directory
    git_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    mp3_path = os.path.join(git_root, 'test-track.mp3')
    
    if not os.path.exists(mp3_path):
        print(f"Could not find test track at {mp3_path}, skipping file test.")
        return

    print(f"Testing roundtrip on file: {mp3_path}")
    st = SpectralTransformer(baseFrequency=5, windowLength=0.1)
    spectrum = st.analyze(mp3_path)
    
    # Load original for comparison
    sig, sr = sf.read(mp3_path)
    
    recon = st.synthesize(spectrum)
    
    # If the original file has multiple channels, ensure the reconstruction matches
    if sig.ndim == 2:
        assert recon.ndim == 2, "Reconstructed signal should be stereo"
        assert sig.shape[1] == recon.shape[1], "Channel mismatch"
    
    # The reconstruction might have slightly padded length if not truncated,
    # but SpectralTransformer.synthesize should truncate it to original_length.
    assert sig.shape[0] == recon.shape[0], f"Length mismatch: {sig.shape[0]} vs {recon.shape[0]}"
    
    mse = np.mean((sig - recon) ** 2)
    print(f"File MSE: {mse:.2e}")
    assert mse < 1e-12, f"File perfect reconstruction failed, MSE = {mse}"
    print("File Roundtrip: PASSED")

def test_eq_curve():
    print("Testing EqCurve interpolation...")
    from audio.eq_curve import EqCurve
    
    # Test empty / default points
    curve = EqCurve()
    # default points are [(0.3, 0.5), (0.3, 0.2)]
    # since both x are 0.3, it returns 1.0 - 0.5 = 0.5 for f <= 0.3, and 1.0 - 0.2 = 0.8 for f >= 0.3
    assert abs(curve.interpolate(0.1) - 0.5) < 1e-6
    assert abs(curve.interpolate(0.3) - 0.5) < 1e-6
    assert abs(curve.interpolate(0.4) - 0.8) < 1e-6

    # Test custom points (e.g. linear ramp)
    # y range in GUI is 0 to 1, gain is 1 - y.
    # points: (0.2, 0.8) => gain = 0.2
    #         (0.8, 0.2) => gain = 0.8
    custom_points = [(0.2, 0.8), (0.8, 0.2)]
    curve = EqCurve(custom_points)
    
    # Boundary: f <= 0.2
    assert abs(curve.interpolate(0.1) - 0.2) < 1e-6
    # Boundary: f >= 0.8
    assert abs(curve.interpolate(0.9) - 0.8) < 1e-6
    # Midpoint: f = 0.5
    # expected gain is 0.2 + (0.8 - 0.2) * 0.5 = 0.5
    assert abs(curve.interpolate(0.5) - 0.5) < 1e-6
    
    print("EqCurve Interpolation: PASSED")

def test_headless_eq_processing():
    print("Testing headless EQ processing logic...")
    from audio.eq_curve import EqCurve
    from audio.spectral_transformer import SpectralTransformer
    
    # Create simple 1D signal (1 second of 440Hz sine at 44100Hz)
    fs = 44100
    t = np.linspace(0, 1, fs, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    transformer = SpectralTransformer(windowLength=512, hopLength=256, windowType='hann')
    spectrum = transformer.analyze((sig, fs))
    
    # Apply flat EQ (gain = 0.5 everywhere)
    curve = EqCurve([(0.5, 0.5)])
    
    num_bins = spectrum.fft_length // 2 + 1
    gains = np.empty(num_bins, dtype=np.float32)
    for i in range(num_bins):
        f = i / (num_bins - 1)
        gains[i] = curve.interpolate(f)
        
    assert np.allclose(gains, 0.5), "Flat EQ gains should all be 0.5"
    
    if spectrum.data.ndim == 3:
        gains_expanded = gains[:, np.newaxis, np.newaxis]
    else:
        gains_expanded = gains[:, np.newaxis]
        
    spectrum.data = spectrum.data * gains_expanded
    recon = transformer.synthesize(spectrum)
    
    expected_recon = sig * 0.5
    mse = np.mean((expected_recon - recon) ** 2)
    print(f"Headless EQ Processing MSE: {mse:.2e}")
    assert mse < 1e-12, f"Headless EQ processing failed, MSE = {mse}"
    print("Headless EQ Processing: PASSED")

if __name__ == '__main__':
    test_roundtrip_1d()
    print("-" * 40)
    test_roundtrip_2d()
    print("-" * 40)
    test_file_roundtrip()
    print("-" * 40)
    test_eq_curve()
    print("-" * 40)
    test_headless_eq_processing()
    print("-" * 40)
    print("All reconstruction and EQ tests passed successfully!")
