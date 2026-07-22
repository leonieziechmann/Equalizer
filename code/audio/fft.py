"""
Fast Fourier Transform (FFT) Implementation

This module contains a pure Python implementation of the Cooley-Tukey Radix-2
recursive Fast Fourier Transform algorithm, twiddle factor caching, and frequency bin helpers.
"""

import numpy as np

# Cache precomputed twiddle factors W_N^k = exp(-2j * pi * k / N)
_twiddles = {}

def get_twiddle(N: int) -> np.ndarray:
    """
    Returns precomputed twiddle factors exp(-2j * pi * k / N) for k in 0..N/2-1.
    """
    if N not in _twiddles:
        _twiddles[N] = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return _twiddles[N]

def fft(x: np.ndarray) -> np.ndarray:
    """
    Computes 1D Discrete Fourier Transform using Cooley-Tukey Radix-2 recursive algorithm.
    Input length len(x) must be a power of 2.
    """
    N = len(x)
    if N <= 4:
        if N <= 1:
            return x
        elif N == 2:
            return np.array([x[0] + x[1], x[0] - x[1]], dtype=complex)
        elif N == 4:
            e0, e1 = x[0] + x[2], x[0] - x[2]
            o0, o1 = x[1] + x[3], x[1] - x[3]
            return np.array([
                e0 + o0,
                e1 - 1j * o1,
                e0 - o0,
                e1 + 1j * o1
            ], dtype=complex)
            
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = get_twiddle(N) * odd
    return np.concatenate([even + T, even - T])

def ifft(x: np.ndarray) -> np.ndarray:
    """
    Computes 1D Inverse Discrete Fourier Transform via conjugate trick:
    ifft(x) = conj(fft(conj(x))) / N
    """
    N = len(x)
    return np.conjugate(fft(np.conjugate(x))) / N

def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Returns Discrete Fourier Transform sample bin frequencies for real input.
    """
    n = int(n)
    val = 1.0 / (n * d)
    N = n // 2 + 1
    return np.arange(0, N, dtype=np.float64) * val



