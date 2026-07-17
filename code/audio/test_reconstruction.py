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

if __name__ == '__main__':
    test_roundtrip_1d()
    print("-" * 40)
    test_roundtrip_2d()
    print("-" * 40)
    test_file_roundtrip()
    print("All reconstruction tests passed successfully!")
