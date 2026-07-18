from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PySide6.QtCore import Qt
from AudioEngine import AudioEngine
import numpy as np

class ComparisonTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setRowCount(7)
        self.setHorizontalHeaderLabels(["Metric", "Original", "EQ Signal"])
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAlternatingRowColors(True)
        
        # Row labels and metric definitions (Name, is_dual)
        self.metrics = [
            ("RMS (Root Mean Square)", True),
            ("Peak Amplitude", True),
            ("Crest Factor", True),
            ("Correlation between Signals", False),
            ("Mean Squared Error (MSE)", False),
            ("Mean Absolute Error (MAE)", False),
            ("Signal-to-Distortion Ratio (SDR)", False),
        ]
        
        # Set up static structure
        for i, (name, is_dual) in enumerate(self.metrics):
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.setItem(i, 0, item)
            
        # Set spans for comparison metrics
        self.setSpan(3, 1, 1, 2)
        self.setSpan(4, 1, 1, 2)
        self.setSpan(5, 1, 1, 2)
        self.setSpan(6, 1, 1, 2)
        
        self.update_metrics()
        
    def update_metrics(self):
        engine = AudioEngine.instance
        if not engine or not engine.audio_loaded:
            # Display placeholders
            for i, (name, is_dual) in enumerate(self.metrics):
                if is_dual:
                    self.set_val(i, 1, "-")
                    self.set_val(i, 2, "-")
                else:
                    self.set_val(i, 1, "-")
            return
            
        # Get metrics dictionary from audio engine
        metrics_dict = engine.get_comparison_metrics()
        if not metrics_dict:
            return
            
        # 1. RMS
        self.set_val(0, 1, f"{metrics_dict['rms_orig']:.5f}")
        self.set_val(0, 2, f"{metrics_dict['rms_recon']:.5f}")

        # 2. Peak Amplitude
        self.set_val(1, 1, f"{metrics_dict['peak_orig']:.5f}")
        self.set_val(1, 2, f"{metrics_dict['peak_recon']:.5f}")

        # 3. Crest Factor
        self.set_val(2, 1, f"{metrics_dict['crest_orig']:.2f}")
        self.set_val(2, 2, f"{metrics_dict['crest_recon']:.2f}")

        # 4. Correlation
        self.set_val(3, 1, f"{metrics_dict['correlation']:.5f}")

        # 5. MSE
        self.set_val(4, 1, f"{metrics_dict['mse']:.5f}")

        # 6. MAE
        self.set_val(5, 1, f"{metrics_dict['mae']:.5f}")

        # 7. SDR
        sdr = metrics_dict['sdr']
        if np.isinf(sdr) or sdr > 150: # Very high SDR or infinite
            sdr_str = "∞ dB (Identical)"
        else:
            sdr_str = f"{sdr:.2f} dB"
        self.set_val(6, 1, sdr_str)

    def set_val(self, r, c, val_str):
        item = QTableWidgetItem(val_str)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(r, c, item)
