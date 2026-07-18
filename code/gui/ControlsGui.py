import os
from signal import signal
import sys

import gui.EqWindow
from AudioEngine import AudioEngine
from gui.ControlPoint import ControlPoint
from PySide6.QtCore import QPointF, QSize, Qt, QLineF, QTimer
from PySide6.QtWidgets import QApplication, QLabel, QHBoxLayout, QVBoxLayout, QMainWindow, QPushButton, QSlider, QStyle, QWidget
from PySide6.QtGui import QIcon, QPainter, QPainterPath, QPen, QColor
import numpy as np
import soundfile as sf  # für das Laden des Signals
from scipy import signal
import sounddevice as sd


class ControlsGui(QWidget):
    def __init__(self, eqWindow: EqWindow.EqWindow, visualizer=None, comparison_table=None):
        super().__init__()
        self.visualizer = visualizer
        self.comparison_table = comparison_table
        layoutV = QVBoxLayout(self)
        layoutV.setContentsMargins(0, 0, 0, 0)
        layoutV.setSpacing(10)

        # Row 1 layout
        widget1 = QWidget()
        layout1 = QHBoxLayout(widget1)
        layout1.setContentsMargins(0, 0, 0, 0)

        searchButton = QPushButton("Select Audio File", self)
        searchButton.clicked.connect(self.search_file)
        
        self.pause_play_button = QPushButton()
        self.pause_play_button.setIcon(
            self.pause_play_button.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )        
        self.pause_play_button.clicked.connect(self.play_pause_audio)
  
        positionSlider = QSlider(Qt.Orientation.Horizontal, self)
        positionSlider.sliderReleased.connect(lambda: self.set_button(AudioEngine.instance.play_audio(positionSlider.value() / 100.0)))

        AudioEngine.instance.positionSlider = positionSlider

        layout1.addWidget(searchButton, 1)
        layout1.addWidget(self.pause_play_button, 0)
        layout1.addWidget(positionSlider, 3)

        # Row 2 layout
        widget2 = QWidget()
        layout2 = QHBoxLayout(widget2)
        layout2.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("No file selected")
        
        self.recompute_button = QPushButton("Apply EQ")
        self.recompute_button.clicked.connect(self.on_recompute_clicked)

        from PySide6.QtWidgets import QComboBox
        self.eqWindow = eqWindow
        self.preset_combo = QComboBox(self)
        self.preset_combo.addItems(["Custom", "Flat", "LPF (Low Pass)", "HPF (High Pass)", "BP 500Hz", "BP 1kHz"])
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        self._applying_preset = False
        eqWindow.pointSelectionChanged.connect(self.on_points_changed)

        self.export_button = QPushButton("Export Audio...")
        self.export_button.clicked.connect(self.export_audio)

        layout2.addWidget(self.label, 2)
        layout2.addWidget(self.export_button, 1)
        layout2.addWidget(self.recompute_button, 1)

        # Row 3 layout
        widget3 = QWidget()
        layout3 = QHBoxLayout(widget3)
        layout3.setContentsMargins(0, 0, 0, 0)
        
        preset_label = QLabel("Preset:")
        layout3.addWidget(preset_label, 0)
        layout3.addWidget(self.preset_combo, 1)
        layout3.addStretch(3)

        layoutV.addWidget(widget1)
        layoutV.addWidget(widget2)
        layoutV.addWidget(widget3)

        # GUI timer to update progress slider and playback state safely in the GUI thread
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(100)  # every 100ms


    def on_recompute_clicked(self):
        # 1. Apply the current EQ curve to AudioEngine
        AudioEngine.instance.update_gains()
        # 2. Recompute the visualizer graphics
        if self.visualizer:
            self.visualizer.recompute_graphics()
        # 3. Update the energy label
        self.update_energy_status()


    def update_progress(self):
        engine = AudioEngine.instance
        if engine and engine.audio_loaded:
            if engine.playing:
                if engine.stream and not engine.stream.active:
                    engine.stop()
                    self.set_button(False)
                
                if hasattr(engine, 'positionSlider') and engine.positionSlider and not engine.positionSlider.isSliderDown():
                    total_frames = engine.ZxxL.shape[1]
                    if total_frames > 0:
                        val = min(100, int(engine.frame / total_frames * 100))
                        engine.positionSlider.blockSignals(True)
                        engine.positionSlider.setValue(val)
                        engine.positionSlider.blockSignals(False)


    def update_energy_status(self):
        if self.comparison_table:
            self.comparison_table.update_metrics()


    def set_button(self, paused: bool):
        self.pause_play_button.setIcon(
            self.pause_play_button.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause if paused else QStyle.StandardPixmap.SP_MediaPlay)
        )
        

    def play_pause_audio(self):
        if AudioEngine.instance.playing:
            self.set_button(False)
            AudioEngine.instance.stop()
        else:
            if AudioEngine.instance.audio_loaded:
                success = AudioEngine.instance.play_audio()
                self.set_button(success)



    def search_file(self):
        #open a file dialog to search for a file
        from PySide6.QtWidgets import QFileDialog
        file_dialog = QFileDialog(self)
        file_dialog.setDirectory(os.path.dirname(__file__) + "/..")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Audio Files (*.mp3 *.wav *.flac)")
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            print(f"Selected file: {file_path}")
            # Signaldaten einlesen
            try:
                sig, sr = sf.read(file_path)
            except Exception as e:
                print(f"Fehler beim Lesen von {file_path}: {e}")
                return
            
            self.label.setText(os.path.basename(file_path))
            AudioEngine.instance.load_audio(sig, sr)
            AudioEngine.instance.frame = 0
            self.update_energy_status()
            if self.visualizer:
                self.visualizer.on_track_loaded()

    def on_points_changed(self):
        if not self._applying_preset:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(0)  # Reset to "Custom"
            self.preset_combo.blockSignals(False)

    def on_preset_changed(self, index):
        if index == 0:
            return  # Custom
        
        self._applying_preset = True
        
        from PySide6.QtCore import QPointF
        
        # Presets mapping: Flat, LPF, HPF, BP 500Hz, BP 1kHz
        presets = {
            1: [QPointF(0.0, 12.0/92.0), QPointF(1.0, 12.0/92.0)], # Flat
            2: [QPointF(0.0, 12.0/92.0), QPointF(0.5, 12.0/92.0), QPointF(0.7, 1.0), QPointF(1.0, 1.0)], # LPF
            3: [QPointF(0.0, 1.0), QPointF(0.4, 1.0), QPointF(0.6, 12.0/92.0), QPointF(1.0, 12.0/92.0)], # HPF
            4: [QPointF(0.0, 1.0), QPointF(0.3, 1.0), QPointF(0.466, 12.0/92.0), QPointF(0.63, 1.0), QPointF(1.0, 1.0)], # BP 500Hz
            5: [QPointF(0.0, 1.0), QPointF(0.4, 1.0), QPointF(0.566, 12.0/92.0), QPointF(0.73, 1.0), QPointF(1.0, 1.0)] # BP 1kHz
        }
        
        points = presets.get(index)
        if points:
            self.eqWindow.points = [QPointF(p.x(), p.y()) for p in points]
            self.eqWindow.selected = -1
            self.eqWindow.update()
            self.eqWindow.pointSelectionChanged.emit()
            
            # Recompute gains and update visualization/RMS labels
            AudioEngine.instance.update_gains()
            if self.visualizer:
                self.visualizer.recompute_graphics()
            self.update_energy_status()
            
        self._applying_preset = False

    def export_audio(self):
        engine = AudioEngine.instance
        if not engine or not engine.audio_loaded:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export Error", "Please select and load an audio file first before exporting.")
            return

        from PySide6.QtWidgets import QFileDialog, QMessageBox
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Equalized Audio",
            os.path.join(os.path.dirname(__file__), "..", "equalized_output.wav"),
            "WAV Files (*.wav);;FLAC Files (*.flac);;All Files (*)"
        )
        
        if file_path:
            success = engine.export_audio(file_path)
            if success:
                QMessageBox.information(
                    self,
                    "Export Success",
                    f"Successfully applied equalizer filter and exported audio to:\n{os.path.basename(file_path)}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Export Failure",
                    "An error occurred while exporting the audio file. Please check the logs."
                )

