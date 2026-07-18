from gui.ControlPoint import ControlPoint
from PySide6.QtCore import QPointF, QSize, Qt, QLineF, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget
from PySide6.QtGui import QPainter, QPainterPath, QPen, QColor
import numpy as np


class EqWindow(QWidget):

    pointSelectionChanged = Signal()

    points = [
		    QPointF(.3, .5),
		    QPointF(.3, .2)
    ]

    selected = -1
    dragging = False
    radius = 6


    def __init__(self):
        super().__init__()
        self.freqs = np.zeros(5)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.sliders = []


        
    def update_frequencies(self, frequencies):
        self.freqs = frequencies
        self.update()


    def lerp(self, color1: QColor, color2: QColor, t: float) -> QColor:
        r = int(color1.red() + (color2.red() - color1.red()) * t)
        g = int(color1.green() + (color2.green() - color1.green()) * t)
        b = int(color1.blue() + (color2.blue() - color1.blue()) * t)
        return QColor(r, g, b)
    

    def paintEvent(self, event):
        with QPainter(self) as painter:
            painter.fillRect(self.rect(), QColor(40, 40, 40))

            # paint Frequencies
            for i in range(len(self.freqs)):
                value = abs(self.freqs[i]) / 256
                x = (i + 0.5) / len(self.freqs)
                y = 1 - np.clip(value, 0, 1)
                color = self.lerp(QColor(100, 255, 100), QColor(255, 100, 100), np.clip(value, 0, 1))
                painter.setPen(QPen(color, 2))
                painter.drawLine(
                    self.toScreenPos(QPointF(x, 1)),
                    self.toScreenPos(QPointF(x, y))
                )


            # Paint Grid Lines and Labels (dB and Frequency ticks)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Setup font
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            # 1. Draw dB Grid Lines and Labels
            db_ticks = [12, 0, -20, -40, -60, -80]
            for db in db_ticks:
                y = 1.0 - (db + 80.0) / 92.0
                p_left = self.toScreenPos(QPointF(0, y))
                p_right = self.toScreenPos(QPointF(1, y))
                
                # Draw subtle grid line
                painter.setPen(QPen(QColor(70, 70, 70, 150), 1, Qt.PenStyle.DashLine))
                painter.drawLine(p_left, p_right)
                
                # Draw dB label next to left border
                painter.setPen(QPen(QColor(180, 180, 180)))
                label = f"{db:+d} dB" if db != 0 else "0 dB"
                text_rect = painter.fontMetrics().boundingRect(label)
                painter.drawText(
                    int(p_left.x() - text_rect.width() - 8),
                    int(p_left.y() + text_rect.height() / 2 - 2),
                    label
                )

            # 2. Draw Frequency Grid Lines and Labels
            freq_ticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            for f_val in freq_ticks:
                x = np.log10(f_val / 20.0) / 3.0
                p_top = self.toScreenPos(QPointF(x, 0))
                p_bottom = self.toScreenPos(QPointF(x, 1))
                
                # Draw subtle grid line
                painter.setPen(QPen(QColor(70, 70, 70, 150), 1, Qt.PenStyle.DashLine))
                painter.drawLine(p_top, p_bottom)
                
                # Draw frequency label below the bottom border
                painter.setPen(QPen(QColor(180, 180, 180)))
                label = f"{f_val/1000:.0f}kHz" if f_val >= 1000 else f"{f_val}Hz"
                text_rect = painter.fontMetrics().boundingRect(label)
                painter.drawText(
                    int(p_bottom.x() - text_rect.width() / 2),
                    int(p_bottom.y() + text_rect.height() + 4),
                    label
                )

            # Polynomobjekt


            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            #Borders
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            left = int(self.width() * 0.05)
            top = int(self.height() * 0.05)
            painter.drawRect(self.rect().adjusted(left, top, -left, -top))

         
            



            # Draw points
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)


            # Kontrollpunkt
            for i, p in enumerate(self.points):
                point_pos = self.toScreenPos(p)

                painter.setBrush(Qt.GlobalColor.white if i != self.selected else Qt.GlobalColor.red)
                painter.drawEllipse(point_pos, 6, 6)

            #Linie
            if len(self.points) == 0:
                return

            painter.setPen(QPen(QColor(100,100,200), 2))

            if self.points[0].x() >0:
                painter.drawLine(QLineF(self.toScreenPos(QPointF(0, self.points[0].y())), self.toScreenPos(self.points[0])))

            if self.points[-1].x() <1:
                painter.drawLine(QLineF(self.toScreenPos(QPointF(1, self.points[-1].y())), self.toScreenPos(self.points[-1])))

            for i in range(len(self.points)-1):
                painter.drawLine(QLineF(self.toScreenPos(self.points[i]), self.toScreenPos(self.points[i+1])))


    def interpolate(self, f: float, sample_rate: float = 44100) -> float:
        from audio.eq_curve import EqCurve
        pts = [(p.x(), p.y()) for p in self.points]
        curve = EqCurve(pts)
        return curve.interpolate(f, sample_rate) 

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        self.selected = -1

        for i, point in enumerate(self.points):
            if QLineF(event.position(), self.toScreenPos(point)).length() <= self.radius + 2:
                self.selected = i
                self.dragging = True
                break

        self.update()
        self.pointSelectionChanged.emit()
    

    def mouseDoubleClickEvent(self, event):
        newPoint = self.screenPosToModel(event.position())

        self.points.append(newPoint)
        self.points.sort(key=lambda p: p.x())
        self.selected = self.points.index(newPoint)
        self.update()
        self.pointSelectionChanged.emit()
        

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected >= 0:
            self.points[self.selected] = self.screenPosToModel(event.position())
            dragged_point = self.points[self.selected]
            self.points.sort(key=lambda p: p.x())
            self.selected = self.points.index(dragged_point)

            self.update()
            self.pointSelectionChanged.emit()
    
    
    def toScreenPos(self, pos: QPointF) -> QPointF:
        borderL = self.width() * 0.05
        borderR = self.width() * 0.95
        borderU = self.height() * 0.05
        borderD = self.height() * 0.95

        return QPointF(pos.x() * (borderR-borderL) + borderL, pos.y() * (borderD-borderU) + borderU)


    def screenPosToModel(self, pos: QPointF) -> QPointF:
        borderL = self.width() * 0.05
        borderR = self.width() * 0.95
        borderU = self.height() * 0.05
        borderD = self.height() * 0.95

        return QPointF(np.clip((pos.x()-borderL) / (borderR-borderL), 0, 1), np.clip((pos.y() -borderU) / (borderD-borderU), 0, 1))


    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.pointSelectionChanged.emit()



    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.selected >= 0:
            self.points.pop(self.selected)
            self.selected = -1
            self.update()
            self.pointSelectionChanged.emit()
            
    def add_new_point(self):
        newPoint = QPointF(0.5, 0.5)
        self.points.append(newPoint)
        self.points.sort(key=lambda p: p.x())
        self.selected = self.points.index(newPoint)
        self.update()
        self.pointSelectionChanged.emit()

    def remove_selected_point(self):
        if self.selected >= 0 and self.selected < len(self.points):
            self.points.pop(self.selected)
            self.selected = -1
            self.update()
            self.pointSelectionChanged.emit()



# #main test
# app = QApplication(sys.argv)

# window = EqWindow()
# window.show()

# app.exec()