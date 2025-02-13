import sys
import cv2
import numpy as np

from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
# QWebEngineView 필요 (PySide6‑QtWebEngine 설치 필요)
from PySide6.QtWebEngineWidgets import QWebEngineView

# Matplotlib canvas for 2D graph
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Plotly for 3D graph
import plotly.graph_objects as go

# --- Video Widget (실시간 영상) ---
class VideoWidget(QLabel):
    def __init__(self, parent=None, cam_index=2, width=640, height=480):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)
        self.timer.start(30)  # 약 30fps

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qimg))

    def release(self):
        if self.cap is not None:
            self.cap.release()

# --- 2D Graph Widget (Matplotlib) ---
class Graph2DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.plot_example()
        self.canvas = FigureCanvas(self.fig)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def plot_example(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.ax.plot(x, y)
        self.ax.set_title("2D Example Graph")

# --- 3D Graph Widget (Plotly in QWebEngineView) ---
class Plotly3DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QWebEngineView()
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)
        self.create_plotly_graph()

    def create_plotly_graph(self):
        # 예시: 스트라이크존 박스를 3D로 표시 (모서리 선으로 연결)
        # 스트라이크존 박스 최소/최대 좌표 (필요에 따라 수정)
        box_min = np.array([-0.08, 0.09, -0.10])
        box_max = np.array([ 0.08, 0.31,  0.10])
        # 8개 꼭짓점 계산
        corners = np.array([
            [box_min[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_max[2]],
            [box_min[0], box_max[1], box_max[2]]
        ])
        # 박스 엣지 인덱스
        edges = [
            (0,1), (1,2), (2,3), (3,0),   # 아래면
            (4,5), (5,6), (6,7), (7,4),   # 윗면
            (0,4), (1,5), (2,6), (3,7)    # 수직 엣지
        ]

        data = []
        for edge in edges:
            x_vals = [corners[edge[0]][0], corners[edge[1]][0]]
            y_vals = [corners[edge[0]][1], corners[edge[1]][1]]
            z_vals = [corners[edge[0]][2], corners[edge[1]][2]]
            data.append(go.Scatter3d(x=x_vals, y=y_vals, z=z_vals,
                                     mode='lines',
                                     line=dict(color='yellow', width=5)))
        layout = go.Layout(title="3D Strike Zone",
                           scene = dict(
                                xaxis_title="X",
                                yaxis_title="Y",
                                zaxis_title="Z"))
        fig = go.Figure(data=data, layout=layout)
        # Plotly HTML 결과물 생성 (Plotly JavaScript 는 CDN으로 로드)
        html = fig.to_html(include_plotlyjs='cdn')
        self.view.setHtml(html)

# --- Main Window ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARUCO Tracker with Strike Zone (PySide6)")
        # 전체 레이아웃: 좌측은 영상, 우측은 두 그래프 영역 (수직 배치)
        main_layout = QHBoxLayout(self)
        
        # 좌측 영역: Video widget
        self.video_widget = VideoWidget(self, cam_index=2, width=640, height=480)
        main_layout.addWidget(self.video_widget, 1)
        
        # 우측 영역: 그래프들 (수직 레이아웃)
        right_layout = QVBoxLayout()
        # 2D 그래프 위젯
        self.graph2d = Graph2DWidget(self)
        right_layout.addWidget(self.graph2d, 1)
        # 구분선 (Optional)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        right_layout.addWidget(line)
        # 3D Plotly 그래프 위젯
        self.plotly3d = Plotly3DWidget(self)
        right_layout.addWidget(self.plotly3d, 1)
        
        main_layout.addLayout(right_layout, 1)
        
    def closeEvent(self, event):
        self.video_widget.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)  # 초기 창 크기 설정
    window.show()
    sys.exit(app.exec())