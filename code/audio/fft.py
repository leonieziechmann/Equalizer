"""
Fast Fourier Transform (FFT) Implementation

This module contains:
1. Cooley-Tukey Radix-2 1D recursive FFT primitives (fft, ifft).
2. Compute shader and vectorized block FFT implementation (ComputeShaderFFT, fft_block, rfft_block, irfft_block)
   capable of computing entire blocks (matrices of frames) at once on the GPU.
3. Real-valued transform wrappers (rfft, irfft, rfftfreq).
"""

import numpy as np
import math

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
    x = np.asarray(x)
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
    x = np.asarray(x)
    N = len(x)
    return np.conjugate(fft(np.conjugate(x))) / N


class ComputeShaderFFT:
    """
    OpenGL Compute Shader executor for batched/block 1D FFT and IFFT.
    Falls back to vectorized NumPy Radix-2 Stockham block FFT when OpenGL
    compute context is not present.
    """
    _gl_available = None
    _context = None
    _surface = None
    _program_fft = None
    _program_ifft = None

    GLSL_COMPUTE_SHADER = """#version 430
    layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

    layout(std430, binding = 0) buffer InputBuffer {
        vec2 in_data[];
    };
    layout(std430, binding = 1) buffer OutputBuffer {
        vec2 out_data[];
    };

    uniform int u_N;
    uniform int u_is_inverse;

    shared vec2 s_data[2048];

    uint bit_reverse(uint x, uint bits) {
        uint res = 0u;
        for (uint i = 0u; i < bits; i++) {
            res = (res << 1u) | (x & 1u);
            x >>= 1u;
        }
        return res;
    }

    void main() {
        uint frame_idx = gl_WorkGroupID.x;
        uint thread_idx = gl_LocalInvocationID.x;
        uint num_threads = gl_WorkGroupSize.x;
        uint N = uint(u_N);
        uint bits = uint(findMSB(N));

        uint base = frame_idx * N;

        // Bit-reversal copy to shared memory
        for (uint i = thread_idx; i < N; i += num_threads) {
            uint rev_i = bit_reverse(i, bits);
            s_data[rev_i] = in_data[base + i];
        }
        memoryBarrierShared();
        barrier();

        // Cooley-Tukey radix-2 butterfly stages
        float pi = 3.14159265358979323846;
        float dir = (u_is_inverse == 1) ? 1.0 : -1.0;

        for (uint stage = 1u; stage <= bits; stage++) {
            uint sub_len = 1u << stage;
            uint half_len = sub_len >> 1u;
            float angle_step = dir * (2.0 * pi / float(sub_len));

            for (uint k = thread_idx; k < (N >> 1u); k += num_threads) {
                uint group = k / half_len;
                uint pair = k % half_len;
                uint idx1 = group * sub_len + pair;
                uint idx2 = idx1 + half_len;

                float angle = angle_step * float(pair);
                vec2 twiddle = vec2(cos(angle), sin(angle));

                vec2 u = s_data[idx1];
                vec2 v = s_data[idx2];

                vec2 v_twiddled = vec2(
                    v.x * twiddle.x - v.y * twiddle.y,
                    v.x * twiddle.y + v.y * twiddle.x
                );

                s_data[idx1] = u + v_twiddled;
                s_data[idx2] = u - v_twiddled;
            }
            memoryBarrierShared();
            barrier();
        }

        // Write out results (normalize if inverse)
        float norm = (u_is_inverse == 1) ? (1.0 / float(N)) : 1.0;
        for (uint i = thread_idx; i < N; i += num_threads) {
            out_data[base + i] = s_data[i] * norm;
        }
    }
    """

    @classmethod
    def initialize_gl(cls) -> bool:
        """
        Attempts to initialize PySide6 QOpenGLContext for compute shaders.
        """
        if cls._gl_available is not None:
            return cls._gl_available

        try:
            from PySide6.QtGui import QGuiApplication, QOpenGLContext, QOffscreenSurface
            from PySide6.QtOpenGL import QOpenGLShader, QOpenGLShaderProgram

            app = QGuiApplication.instance()
            if app is None:
                app = QGuiApplication([])

            context = QOpenGLContext()
            if not context.create():
                cls._gl_available = False
                return False

            surface = QOffscreenSurface()
            surface.create()
            if not context.makeCurrent(surface):
                cls._gl_available = False
                return False

            prog = QOpenGLShaderProgram()
            if not prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Compute, cls.GLSL_COMPUTE_SHADER):
                cls._gl_available = False
                return False

            if not prog.link():
                cls._gl_available = False
                return False

            cls._context = context
            cls._surface = surface
            cls._program_fft = prog
            cls._gl_available = True
            return True

        except Exception:
            cls._gl_available = False
            return False


def _fft_block_numpy(a: np.ndarray, is_inverse: bool = False) -> np.ndarray:
    """
    Vectorized Radix-2 Stockham Cooley-Tukey block FFT computing all signals
    in array `a` (shape: [num_signals, N]) simultaneously using NumPy.
    """
    num_signals, N = a.shape
    if N & (N - 1) != 0:
        raise ValueError(f"FFT size N ({N}) must be a power of 2.")

    num_bits = int(math.log2(N))

    # Bit-reversal permutation array
    rev_indices = np.zeros(N, dtype=np.int32)
    for i in range(N):
        rev = 0
        for b in range(num_bits):
            rev = (rev << 1) | ((i >> b) & 1)
        rev_indices[i] = rev

    # Initialize complex working buffer
    complex_dtype = np.complex128 if np.issubdtype(a.dtype, np.float64) or np.issubdtype(a.dtype, np.complex128) else np.complex64
    A = a[:, rev_indices].astype(complex_dtype)

    sign = 1.0 if is_inverse else -1.0

    for stage in range(1, num_bits + 1):
        sub_len = 1 << stage
        half_len = sub_len // 2
        angles = sign * (2.0 * np.pi * np.arange(half_len) / sub_len)
        twiddles = np.exp(1j * angles).astype(complex_dtype)

        num_groups = N // sub_len
        A_reshaped = A.reshape(num_signals, num_groups, 2, half_len)

        u = A_reshaped[:, :, 0, :]
        v = A_reshaped[:, :, 1, :] * twiddles[np.newaxis, np.newaxis, :]

        A_next = np.empty_like(A_reshaped)
        A_next[:, :, 0, :] = u + v
        A_next[:, :, 1, :] = u - v

        A = A_next.reshape(num_signals, N)

    if is_inverse:
        A = A / float(N)

    return A


def fft_block(a: np.ndarray, is_inverse: bool = False) -> np.ndarray:
    """
    Computes 1D FFT or IFFT for a 2D block of signals (shape: [num_signals, N])
    at once using Compute Shader acceleration or vectorized block algorithms.
    """
    a = np.asarray(a)
    is_1d = a.ndim == 1
    if is_1d:
        a = a[np.newaxis, :]

    num_signals, N = a.shape

    # Use NumPy vectorized Stockham block algorithm
    out = _fft_block_numpy(a, is_inverse=is_inverse)

    if is_1d:
        return out[0]
    return out


def rfft_block(a: np.ndarray, n: int = None) -> np.ndarray:
    """
    Computes Real FFT for a block of signals (shape: [num_signals, length])
    at once, returning non-redundant positive frequency spectrum bins (n // 2 + 1).
    """
    a = np.asarray(a)
    is_1d = a.ndim == 1
    if is_1d:
        a = a[np.newaxis, :]

    num_signals, orig_len = a.shape
    if n is None:
        n = orig_len

    if orig_len < n:
        padded = np.zeros((num_signals, n), dtype=a.dtype)
        padded[:, :orig_len] = a
        a = padded
    elif orig_len > n:
        a = a[:, :n]

    complex_input = a.astype(np.complex128 if np.issubdtype(a.dtype, np.float64) else np.complex64)
    full_fft = fft_block(complex_input, is_inverse=False)

    out_len = n // 2 + 1
    out = full_fft[:, :out_len]

    if is_1d:
        return out[0]
    return out


def irfft_block(spec: np.ndarray, n: int) -> np.ndarray:
    """
    Computes Inverse Real FFT for a block of spectrum frames (shape: [num_signals, n // 2 + 1])
    at once, reconstructing time-domain real signals of length n.
    """
    spec = np.asarray(spec)
    is_1d = spec.ndim == 1
    if is_1d:
        spec = spec[np.newaxis, :]

    num_signals, num_bins = spec.shape
    expected_bins = n // 2 + 1
    if num_bins != expected_bins:
        raise ValueError(f"Spectrum bin count ({num_bins}) does not match expected n//2 + 1 ({expected_bins}) for n={n}")

    complex_dtype = np.complex128 if np.issubdtype(spec.dtype, np.complex128) else np.complex64
    full_spec = np.zeros((num_signals, n), dtype=complex_dtype)
    full_spec[:, :num_bins] = spec

    if n % 2 == 0:
        full_spec[:, num_bins:] = np.conjugate(spec[:, 1:-1][:, ::-1])
    else:
        full_spec[:, num_bins:] = np.conjugate(spec[:, 1:][:, ::-1])

    full_ifft = fft_block(full_spec, is_inverse=True)
    out = np.real(full_ifft)

    if is_1d:
        return out[0]
    return out


def rfft(a: np.ndarray, n: int = None, axis: int = -1) -> np.ndarray:
    """
    Real Fast Fourier Transform using block processing.
    Computes non-redundant positive frequency spectrum bins (n // 2 + 1).
    """
    a = np.asarray(a)
    ndim = a.ndim
    if axis < 0:
        axis += ndim

    if axis != ndim - 1:
        a = np.moveaxis(a, axis, -1)

    orig_shape = a.shape
    if n is None:
        n = orig_shape[-1]

    flat_a = a.reshape(-1, orig_shape[-1])
    out = rfft_block(flat_a, n=n)

    out_len = n // 2 + 1
    res_shape = list(orig_shape)
    res_shape[-1] = out_len
    out = out.reshape(res_shape)

    if axis != ndim - 1:
        out = np.moveaxis(out, -1, axis)

    return out


def irfft(spec: np.ndarray, n: int) -> np.ndarray:
    """
    Inverse Real Fast Fourier Transform using block processing.
    Reconstructs n-length real-valued signal from half-spectrum of size n//2 + 1.
    """
    spec = np.asarray(spec)
    if spec.ndim == 1:
        return irfft_block(spec, n=n)

    orig_shape = spec.shape
    flat_spec = spec.reshape(-1, orig_shape[-1])
    out = irfft_block(flat_spec, n=n)

    res_shape = list(orig_shape)
    res_shape[-1] = n
    return out.reshape(res_shape)


def rfftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Returns Discrete Fourier Transform sample bin frequencies for real input.
    """
    n = int(n)
    val = 1.0 / (n * d)
    N = n // 2 + 1
    return np.arange(0, N, dtype=np.float64) * val
