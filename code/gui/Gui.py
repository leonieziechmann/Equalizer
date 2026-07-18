import sys
from . import EqWindow
from gui.ControlsGui import ControlsGui
from PySide6.QtCore import QSize, Qt
from AudioEngine import AudioEngine
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QPushButton, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QLabel
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Equalizer Dashboard")
        self.setMinimumSize(QSize(1100, 650))
        
        # Apply visual styles to PySide6 components
        self.setStyleSheet("""
            QMainWindow {
                background-color: #282828;
            }
            QWidget {
                background-color: #282828;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #383838;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #00e676;
            }
            QPushButton:pressed {
                background-color: #242424;
            }
            QSlider::groove:vertical {
                background: #1e1e1e;
                width: 6px;
                border-radius: 3px;
            }
            QSlider::handle:vertical {
                background: #00e676;
                border: 1px solid #1e1e1e;
                height: 16px;
                width: 16px;
                margin: 0 -5px;
                border-radius: 8px;
            }
            QSlider::handle:vertical:hover {
                background: #00ff87;
            }
            QSlider::groove:horizontal {
                background: #1e1e1e;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #7c4dff;
                border: 1px solid #1e1e1e;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #b388ff;
            }
            QComboBox {
                background-color: #383838;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 4px 10px;
                color: #ffffff;
                font-weight: bold;
                min-width: 100px;
            }
            QComboBox:hover {
                background-color: #4a4a4a;
                border: 1px solid #7c4dff;
            }
            QComboBox QAbstractItemView {
                background-color: #282828;
                color: #e0e0e0;
                selection-background-color: #7c4dff;
                selection-color: #ffffff;
                border: 1px solid #555555;
            }
            QLabel {
                font-size: 12px;
                font-weight: 500;
            }
            QTableWidget {
                background-color: #1e1e1e;
                alternate-background-color: #252525;
                gridline-color: #383838;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-radius: 4px;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #383838;
                color: #00e676;
                padding: 6px;
                font-weight: bold;
                border: 1px solid #555555;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        eq_Widget = QWidget()
        eq_layout = QVBoxLayout(eq_Widget)
        eq_layout.setContentsMargins(10, 10, 10, 10)
        eq_layout.setSpacing(10)

        eq_window = EqWindow.EqWindow()
        self.eq_window = eq_window
        AudioEngine.instance = AudioEngine(eq_window) 

        # Create control panel for selected points
        controls_panel = QWidget()
        controls_layout = QHBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(15)
        
        # 50% Left side: Sliders (Gain and Frequency)
        sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 0, 0, 0)
        sliders_layout.setSpacing(8)
        
        gain_row = QWidget()
        gain_row_layout = QHBoxLayout(gain_row)
        gain_row_layout.setContentsMargins(0, 0, 0, 0)
        self.gain_label = QLabel("Gain: -")
        self.gain_label.setMinimumWidth(120)
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(-800, 120)
        self.gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gain_slider.setTickInterval(100)  # Ticks every 10 dB
        self.gain_slider.setEnabled(False)
        gain_row_layout.addWidget(self.gain_label, 1)
        gain_row_layout.addWidget(self.gain_slider, 3)
        
        freq_row = QWidget()
        freq_row_layout = QHBoxLayout(freq_row)
        freq_row_layout.setContentsMargins(0, 0, 0, 0)
        self.freq_label = QLabel("Frequency: -")
        self.freq_label.setMinimumWidth(120)
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(0, 1000)
        self.freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.freq_slider.setTickInterval(100)  # 10 ticks across log range
        self.freq_slider.setEnabled(False)
        freq_row_layout.addWidget(self.freq_label, 1)
        freq_row_layout.addWidget(self.freq_slider, 3)
        
        sliders_layout.addWidget(gain_row)
        sliders_layout.addWidget(freq_row)
        
        # 50% Right side: Buttons (Add / Remove)
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(15)
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.add_button = QPushButton("Add Point")
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.setEnabled(False)
        
        self.add_button.setStyleSheet("""
            QPushButton {
                background-color: #383838;
                border: 1px solid #7c4dff;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #b388ff;
            }
        """)
        self.remove_button.setStyleSheet("""
            QPushButton {
                background-color: #383838;
                border: 1px solid #ff5252;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #ff8a80;
            }
            QPushButton:disabled {
                border: 1px solid #444444;
                color: #888888;
            }
        """)
        
        buttons_layout.addWidget(self.add_button)
        buttons_layout.addWidget(self.remove_button)
        
        controls_layout.addWidget(sliders_widget, 1)
        controls_layout.addWidget(buttons_widget, 1)
        
        # Connections
        eq_window.pointSelectionChanged.connect(self.update_point_controls)
        self.gain_slider.valueChanged.connect(self.on_gain_slider_changed)
        self.freq_slider.valueChanged.connect(self.on_freq_slider_changed)
        self.add_button.clicked.connect(self.on_add_point_clicked)
        self.remove_button.clicked.connect(self.on_remove_point_clicked)

        # Instantiate VisualizerWidget early to pass callback
        from gui.VisualizerWidget import VisualizerWidget
        self.visualizer = VisualizerWidget()

        # Instantiate ComparisonTable
        from gui.ComparisonTable import ComparisonTable
        self.comparison_table = ComparisonTable()

        # ControlsGui on left side above EQ Window
        from gui.ControlsGui import ControlsGui
        self.controls_gui = ControlsGui(eq_window, self.visualizer, self.comparison_table)

        # Assemble Left Column
        eq_layout.addWidget(self.controls_gui, 0)
        eq_layout.addWidget(eq_window, 1)
        eq_layout.addWidget(controls_panel, 0)

        # Arrange right side: Visualizer and Comparison Table
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(15)
        right_layout.addWidget(self.visualizer, 4)
        right_layout.addWidget(self.comparison_table, 2)

        # Main Layout (Horizontal)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        main_layout.addWidget(eq_Widget, 4)
        main_layout.addWidget(right_widget, 5)

        central_widget.setLayout(main_layout)
        eq_Widget.setLayout(eq_layout)
        controls_panel.setLayout(controls_layout)

    def update_point_controls(self):
        selected_idx = self.eq_window.selected
        if selected_idx >= 0 and selected_idx < len(self.eq_window.points):
            point = self.eq_window.points[selected_idx]
            
            self.gain_slider.setEnabled(True)
            self.freq_slider.setEnabled(True)
            self.remove_button.setEnabled(True)
            
            # Map point.y() to gain_db
            gain_db = (1.0 - point.y()) * 92.0 - 80.0
            # Map point.x() to freq_hz
            freq_hz = 20.0 * (1000.0 ** point.x())
            
            self.gain_slider.blockSignals(True)
            self.freq_slider.blockSignals(True)
            self.gain_slider.setValue(int(gain_db * 10))
            self.freq_slider.setValue(int(point.x() * 1000))
            self.gain_slider.blockSignals(False)
            self.freq_slider.blockSignals(False)
            
            self.gain_label.setText(f"Gain: {gain_db:.1f} dB")
            if freq_hz >= 1000.0:
                self.freq_label.setText(f"Frequency: {freq_hz/1000.0:.2f} kHz")
            else:
                self.freq_label.setText(f"Frequency: {int(freq_hz)} Hz")
        else:
            self.gain_slider.setEnabled(False)
            self.freq_slider.setEnabled(False)
            self.remove_button.setEnabled(False)
            self.gain_label.setText("Gain: -")
            self.freq_label.setText("Frequency: -")

    def on_gain_slider_changed(self, value):
        selected_idx = self.eq_window.selected
        if selected_idx >= 0 and selected_idx < len(self.eq_window.points):
            point = self.eq_window.points[selected_idx]
            gain_db = value / 10.0
            point.setY(1.0 - (gain_db + 80.0) / 92.0)
            self.eq_window.update()
            self.gain_label.setText(f"Gain: {gain_db:.1f} dB")

    def on_freq_slider_changed(self, value):
        selected_idx = self.eq_window.selected
        if selected_idx >= 0 and selected_idx < len(self.eq_window.points):
            point = self.eq_window.points[selected_idx]
            point.setX(value / 1000.0)
            self.eq_window.points.sort(key=lambda p: p.x())
            self.eq_window.selected = self.eq_window.points.index(point)
            self.eq_window.update()
            
            freq_hz = 20.0 * (1000.0 ** point.x())
            if freq_hz >= 1000.0:
                self.freq_label.setText(f"Frequency: {freq_hz/1000.0:.2f} kHz")
            else:
                self.freq_label.setText(f"Frequency: {int(freq_hz)} Hz")

    def on_add_point_clicked(self):
        self.eq_window.add_new_point()

    def on_remove_point_clicked(self):
        self.eq_window.remove_selected_point()

def main():
    app = QApplication([])
    window = MainWindow()
    window.showMaximized()

    app.exec()
    app.quit()


