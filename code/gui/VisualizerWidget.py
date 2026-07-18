import sys
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from AudioEngine import AudioEngine

class VisualizerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create matplotlib Figure and Canvas
        self.figure = Figure(facecolor='#282828')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Style variables
        self.dark_bg = '#1e1e1e'
        self.text_color = '#e0e0e0'
        self.grid_color = '#444444'
        
        # Create subplots
        self.ax_spec_pre = self.figure.add_subplot(3, 1, 1, facecolor=self.dark_bg)
        self.ax_spec_post = self.figure.add_subplot(3, 1, 2, facecolor=self.dark_bg)
        self.ax_curve = self.figure.add_subplot(3, 1, 3, facecolor=self.dark_bg)
        
        # Title and Labels styling
        self.ax_spec_pre.set_title("Pre-EQ Spectrogram (Original Track)", color=self.text_color, fontsize=10, pad=5)
        self.ax_spec_pre.set_ylabel("Frequency (Hz)", color=self.text_color, fontsize=8)
        
        self.ax_spec_post.set_title("Post-EQ Spectrogram (Equalized Track)", color=self.text_color, fontsize=10, pad=5)
        self.ax_spec_post.set_ylabel("Frequency (Hz)", color=self.text_color, fontsize=8)
        
        self.ax_curve.set_title("Equalizer Filter Response Curve", color=self.text_color, fontsize=10, pad=5)
        self.ax_curve.set_ylabel("Gain (dB)", color=self.text_color, fontsize=8)
        self.ax_curve.set_xlabel("Frequency (Hz)", color=self.text_color, fontsize=8)
        
        # Configure subplots layout margins
        self.figure.subplots_adjust(hspace=0.55, top=0.92, bottom=0.10, left=0.10, right=0.95)
        
        # Setup playhead vertical lines on both spectrograms
        self.line_playhead_pre = self.ax_spec_pre.axvline(0, color='#ff5252', linewidth=1.5, zorder=5)
        self.line_playhead_post = self.ax_spec_post.axvline(0, color='#ff5252', linewidth=1.5, zorder=5)
        
        # Initial dummy data for lines
        self.sample_rate = 44100
        self.num_bins = 257
        base_freq = self.sample_rate / 512.0
        self.freqs = np.fft.rfftfreq(512, d=1.0/self.sample_rate)
        self.line_curve, = self.ax_curve.plot(self.freqs[1:], np.zeros(self.num_bins - 1), color='#00e676', linewidth=2.0)
        self.ax_curve.axhline(0, color='#888888', linestyle='--', alpha=0.4, linewidth=1.0)
        
        # Configure scales and ticks for all logarithmic frequency plots
        for ax in [self.ax_spec_pre, self.ax_spec_post, self.ax_curve]:
            ax.set_xscale('log') if ax == self.ax_curve else ax.set_yscale('log')
            
            if ax == self.ax_curve:
                ax.set_xlim(base_freq, 20000)
                ax.set_ylim(-80, 12)
            else:
                ax.set_ylim(base_freq, 20000)
                
            ax.grid(True, which='both', color=self.grid_color, linestyle=':', alpha=0.4)
            ax.tick_params(colors=self.text_color, labelsize=8)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color(self.grid_color)
            ax.spines['bottom'].set_color(self.grid_color)
            
        # State trackers
        self.mesh_pre = None
        self.mesh_post = None
        self.time_axis = None
        self.mag_peak = 1.0
        
    def on_track_loaded(self):
        engine = AudioEngine.instance
        if not engine or not engine.audio_loaded:
            return
            
        self.sample_rate = engine.sample_rate
        self.num_bins = engine.num_bins
        self.freqs = np.fft.rfftfreq(engine.windowLength, d=1.0/self.sample_rate)
        
        base_freq = self.sample_rate / engine.windowLength
        for ax in [self.ax_spec_pre, self.ax_spec_post, self.ax_curve]:
            if ax == self.ax_curve:
                ax.set_xlim(base_freq, 20000)
            else:
                ax.set_ylim(base_freq, 20000)
                
        self.line_curve.set_xdata(self.freqs[1:])
        
        self.load_pre_eq_spectrogram()
        
    def load_pre_eq_spectrogram(self):
        engine = AudioEngine.instance
        if not engine or not engine.audio_loaded:
            return
            
        # Average pre-EQ magnitude spectrum of left and right channels
        mag_pre = (np.abs(engine.ZxxL) + np.abs(engine.ZxxR)) / 2.0
        
        # Find peak magnitude in input data and normalize
        self.mag_peak = np.max(mag_pre)
        if self.mag_peak < 1e-9:
            self.mag_peak = 1.0
        mag_pre_normalized = mag_pre / self.mag_peak
        
        # Downsample horizontally to 1000 frames to ensure fast plotting
        num_frames = mag_pre_normalized.shape[1]
        step = max(1, num_frames // 1000)
        mag_pre_ds = mag_pre_normalized[:, ::step][:, :1000]
        
        # Interpolate from linear FFT bins to log-spaced frequency grid (200 bins)
        base_freq = self.sample_rate / engine.windowLength
        lin_freqs = np.fft.rfftfreq(engine.windowLength, d=1.0/engine.sample_rate)
        log_freqs = np.logspace(np.log10(base_freq), np.log10(20000), 200)
        
        f_interp = interp1d(lin_freqs, mag_pre_ds, axis=0, bounds_error=False, fill_value=1e-6)
        mag_pre_log = f_interp(log_freqs)
        
        # Convert to dB
        db_pre_log = 20 * np.log10(mag_pre_log + 1e-6)
        db_pre_log_clipped = np.clip(db_pre_log, -80, 5)
        
        # Clear previous mesh if it exists
        if self.mesh_pre:
            self.mesh_pre.remove()
            self.mesh_pre = None
            
        total_duration = len(engine.sig) / engine.sample_rate
        self.time_axis = np.linspace(0, total_duration, mag_pre_ds.shape[1])
        
        T, F = np.meshgrid(self.time_axis, log_freqs)
        self.mesh_pre = self.ax_spec_pre.pcolormesh(
            T, F, db_pre_log_clipped, 
            cmap='magma', 
            vmin=-60, vmax=0, 
            shading='auto',
            zorder=3
        )
        
        self.ax_spec_pre.set_xlim(0, total_duration)
        self.ax_spec_post.set_xlim(0, total_duration)
        
        # Synchronize and pre-populate the post-EQ spectrogram with the same dimensions
        self.recompute_graphics()
        
    def recompute_graphics(self):
        engine = AudioEngine.instance
        if not engine or not engine.audio_loaded:
            return
            
        # Apply the current EQ gains column-wise
        gains_col = engine.gains[:, np.newaxis]
        post_ZxxL = engine.ZxxL * gains_col
        post_ZxxR = engine.ZxxR * gains_col
        
        mag_post = (np.abs(post_ZxxL) + np.abs(post_ZxxR)) / 2.0
        mag_post_normalized = mag_post / self.mag_peak
        
        # Downsample horizontally to 1000 frames
        num_frames = mag_post_normalized.shape[1]
        step = max(1, num_frames // 1000)
        mag_post_ds = mag_post_normalized[:, ::step][:, :1000]
        
        # Interpolate to log-frequency scale
        base_freq = self.sample_rate / engine.windowLength
        lin_freqs = np.fft.rfftfreq(engine.windowLength, d=1.0/engine.sample_rate)
        log_freqs = np.logspace(np.log10(base_freq), np.log10(20000), 200)
        
        f_interp = interp1d(lin_freqs, mag_post_ds, axis=0, bounds_error=False, fill_value=1e-6)
        mag_post_log = f_interp(log_freqs)
        
        db_post_log = 20 * np.log10(mag_post_log + 1e-6)
        db_post_log_clipped = np.clip(db_post_log, -80, 5)
        
        # Clear previous mesh if it exists
        if self.mesh_post:
            self.mesh_post.remove()
            self.mesh_post = None
            
        T, F = np.meshgrid(self.time_axis, log_freqs)
        self.mesh_post = self.ax_spec_post.pcolormesh(
            T, F, db_post_log_clipped, 
            cmap='magma', 
            vmin=-60, vmax=0, 
            shading='auto',
            zorder=3
        )
        
        # Update EQ response curve line
        if engine.gains is not None and len(engine.gains) == self.num_bins:
            db_curve = 20 * np.log10(engine.gains + 1e-6)
            db_curve_clipped = np.clip(db_curve, -80, 12)
            self.line_curve.set_ydata(db_curve_clipped[1:])
            
        # Update static playhead positions to align with the current position slider value
        total_duration = len(engine.sig) / engine.sample_rate
        if hasattr(engine, 'positionSlider') and engine.positionSlider:
            ratio = engine.positionSlider.value() / 100.0
        else:
            ratio = engine.frame / max(1, engine.ZxxL.shape[1])
        current_time = ratio * total_duration
        self.line_playhead_pre.set_xdata([current_time, current_time])
        self.line_playhead_post.set_xdata([current_time, current_time])
        
        self.canvas.draw()
