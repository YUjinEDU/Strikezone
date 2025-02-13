아래는 PyQtGraph를 처음 시작하는 분들을 위한 기초 튜토리얼입니다. 이 튜토리얼에서는 PyQtGraph의 기본 개념, 설치, 간단한 예제 코드, 그리고 실시간 업데이트나 3D 그래프 등 확장 기능에 대해 단계별로 설명합니다.

---

## 1. PyQtGraph란?

- **PyQtGraph**는 PyQt(Pyside) 기반의 빠르고 인터랙티브한 데이터 시각화 라이브러리입니다.  
- 2D 그래프(라인, 스캐터, 이미지 등)와 3D 그래프(OpenGL 기반)를 쉽게 생성할 수 있으며, 실시간 데이터 업데이트와 대량 데이터 처리에 최적화되어 있습니다.  
- Qt의 QGraphicsView와 OpenGL(QOpenGLWidget)을 내부적으로 사용하므로, GUI 애플리케이션에 자연스럽게 통합됩니다.

---

## 2. 설치하기

PyQtGraph는 pip로 쉽게 설치할 수 있습니다. (PyQt나 PySide도 필요합니다.)

```bash
pip install pyqtgraph
```

예를 들어, PySide6를 사용하고 싶다면:

```bash
pip install PySide6 pyqtgraph
```

---

## 3. 기본 사용법

### 3.1. 간단한 2D 그래프 그리기

아래는 가장 기본적인 예제입니다. 데이터를 그리고 업데이트하는 간단한 예제 코드입니다.

```python
import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph 기본 예제")
        self.plotWidget = pg.PlotWidget()
        self.setCentralWidget(self.plotWidget)
        self.plot_data()

    def plot_data(self):
        # 간단한 선 그래프 데이터 생성 (예: 사인 함수)
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)
        self.plotWidget.plot(x, y, pen=pg.mkPen('r', width=2))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

**설명:**  
- `pg.PlotWidget()`를 생성하여 기본 선 그래프를 그립니다.  
- `plot()` 메서드를 사용하여 x, y 데이터를 그리며, pen 인수를 이용해 선의 색상이나 두께를 지정할 수 있습니다.

---

### 3.2. 실시간 업데이트 (Timer 사용)

실시간 데이터를 업데이트하려면 QTimer를 사용하여 데이터를 주기적으로 갱신할 수 있습니다.

```python
import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 업데이트 예제")
        self.plotWidget = pg.PlotWidget()
        self.setCentralWidget(self.plotWidget)
        
        # 빈 plot 생성 (미리 scatter 데이터 추가)
        self.curve = self.plotWidget.plot(pen=pg.mkPen('b', width=2))
        self.x = np.linspace(0, 10, 1000)
        self.ptr = 0
        
        # 타이머 설정: 50ms마다 update 호출
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)

    def update_plot(self):
        # 예제: 사인파를 시간에 따라 이동시키는 예
        y = np.sin(self.x + self.ptr/100)
        self.curve.setData(self.x, y)
        self.ptr += 1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

**설명:**  
- `QTimer`를 이용하여 일정 간격마다 `update_plot` 함수를 호출합니다.  
- `curve.setData()` 메서드를 사용하면 기존의 그래프 데이터를 새 값으로 바꾸어 실시간 업데이트가 가능합니다.

---

### 3.3. 2D 기록지 스타일로 커스터마이징

투구 기록지와 같이 사각형 영역에 점을 찍고 싶다면, 배경에 사각형을 그리고 scatter plot 데이터를 업데이트하는 방식으로 구성할 수 있습니다.

```python
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout

class RecordSheetWidget(pg.PlotWidget):
    def __init__(self, width=400, height=600, parent=None):
        super().__init__(parent)
        self.setBackground('w')  # 흰 배경
        self.setFixedSize(width, height)
        self.setMouseEnabled(x=False, y=False)
        self.showGrid(x=False, y=False)
        self.setXRange(0, width)
        self.setYRange(0, height)
        # 기록지 테두리 추가: QtGraphicsRectItem을 사용 (PyQtGraph 내부 아이템)
        rect = pg.QtGui.QGraphicsRectItem(0, 0, width, height)
        rect.setPen(pg.mkPen(color='b', width=2))
        self.addItem(rect)
        # y축 반전: 기록지에서는 보통 위쪽이 투구측이므로
        self.invertY(True)
        # Scatter plot: 투구점 그리기
        self.scatter = self.plot([], [], pen=None, symbol='o', symbolBrush='g', symbolSize=8)

    def updateData(self, xdata, ydata):
        self.scatter.setData(x=xdata, y=ydata)
```

**설명:**  
- `RecordSheetWidget`은 `pg.PlotWidget`을 상속받아 사각형 영역과 scatter plot을 추가합니다.  
- `invertY(True)`를 사용하면 y축 방향을 반전하여 기록지 스타일로 만들 수 있습니다.  
- `updateData()` 메서드로 데이터를 업데이트하면 실시간으로 점들이 표시됩니다.

---

### 3.4. 3D 그래프 (GLViewWidget 사용)

PyQtGraph는 3D 그래프를 위해 OpenGL 기반의 `GLViewWidget`을 제공합니다.

```python
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QWidget, QVBoxLayout

class ThreeDGraphWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=5)
        self.opts['center'] = pg.Vector(0, 0, 0)
        # 바닥 그리드 추가
        grid = gl.GLGridItem()
        grid.setSize(x=10, y=10)
        grid.setSpacing(x=1, y=1)
        self.addItem(grid)
        # 3D 산점도 (빈 상태에서 시작)
        self.scatter = gl.GLScatterPlotItem()
        self.addItem(self.scatter)
    
    def updateData(self, xdata, ydata, zdata):
        if len(xdata) > 0:
            pos = np.vstack([xdata, ydata, zdata]).T
            self.scatter.setData(pos=pos, size=5, color=(1,0.5,0,1))
        else:
            self.scatter.setData(pos=np.array([]))
```

**설명:**  
- `GLViewWidget`을 상속받아 3D 공간에 산점도를 그리고, 카메라 위치와 그리드를 설정합니다.  
- `updateData()` 메서드를 통해 3D 점 데이터를 업데이트할 수 있으며, 마우스 조작(회전, 확대, 이동 등)도 기본적으로 지원됩니다.

---

## 4. 전체 파이프라인 구성 예

앞서 소개한 위젯들을 하나의 메인 창에 통합하면 다음과 같이 구성할 수 있습니다.  
예를 들어, 좌측에는 OpenCV로 캡처한 프레임, 우측 상단에는 2D 기록지, 우측 하단에는 3D 그래프를 배치하는 방식입니다.

```python
import sys, cv2, numpy as np, pyqtgraph as pg, pyqtgraph.opengl as gl
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap

# 앞서 정의한 RecordSheetWidget과 ThreeDGraphWidget를 사용한다고 가정

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph 튜토리얼")
        self.setGeometry(100, 100, 1200, 800)
        
        self.record_sheet_x = []  # 2D 기록지 데이터
        self.record_sheet_y = []
        self.pitch_points_3d_x = []  # 3D 데이터
        self.pitch_points_3d_y = []
        self.pitch_points_3d_z = []
        
        self.init_ui()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("카메라 열기 실패")
            sys.exit(1)
            
        # 비디오 프레임 업데이트 타이머 (약 30ms)
        self.timer_video = QTimer(self)
        self.timer_video.timeout.connect(self.update_video)
        self.timer_video.start(30)
        
        # 그래프 업데이트 타이머 (1초마다)
        self.timer_graph = QTimer(self)
        self.timer_graph.timeout.connect(self.update_graphs)
        self.timer_graph.start(1000)
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # 좌측: 비디오 프레임 (60%)
        self.video_label = QLabel()
        self.video_label.setFixedWidth(720)
        self.video_label.setScaledContents(True)
        main_layout.addWidget(self.video_label)
        
        # 우측: 두 영역 (수직 배치, 40%)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        # 상단: 2D 기록지
        self.record_sheet = RecordSheetWidget(400, 600)
        self.record_sheet.setFixedHeight(320)
        right_layout.addWidget(self.record_sheet)
        # 하단: 3D 그래프
        self.threeD_graph = ThreeDGraphWidget()
        self.threeD_graph.setFixedHeight(480)
        right_layout.addWidget(self.threeD_graph)
        
        main_layout.addWidget(right_widget)
    
    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytesPerLine = ch * w
        qImg = QImage(frame_rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qImg))
        
        # 예제: 무작위로 새로운 투구점을 추가 (실제 AR 처리 결과로 대체)
        if np.random.rand() < 0.01:
            new_point = np.array([
                np.random.uniform(-0.08, 0.08),    # x
                np.random.uniform(0.09, 0.31),       # y
                -0.3                               # z (고정)
            ])
            # 2D 기록지 단순 선형 매핑: x ∈ [-0.08, 0.08] → [0, 400], y ∈ [0.09, 0.31] → [0, 600]
            rec_x = ((new_point[0] + 0.08) / 0.16) * 400
            rec_y = ((new_point[1] - 0.09) / 0.22) * 600
            self.record_sheet_x.append(rec_x)
            self.record_sheet_y.append(rec_y)
            self.pitch_points_3d_x.append(new_point[0])
            self.pitch_points_3d_y.append(new_point[1])
            self.pitch_points_3d_z.append(new_point[2])
    
    def update_graphs(self):
        self.record_sheet.updateData(self.record_sheet_x, self.record_sheet_y)
        self.threeD_graph.updateData(self.pitch_points_3d_x, self.pitch_points_3d_y, self.pitch_points_3d_z)
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

---

## 결론

1. **PyQtGraph의 강점**:  
   - 간단한 코드로 2D/3D 그래프를 구현할 수 있으며, 실시간 데이터 업데이트에 유리합니다.
   - Qt와 자연스럽게 통합되어 PySide/PyQt 애플리케이션 내에서 고성능의 시각화를 제공합니다.
   - 커스터마이징이 자유로워서 색상, 마커, 그리드 등 원하는 스타일로 쉽게 조정할 수 있습니다.

2. **파이프라인 요약**:  
   - **데이터 취득**: OpenCV로 카메라 프레임 읽기 → AR/비전 처리 → 필요한 좌표 산출  
   - **데이터 전처리**: 2D 기록지 좌표 변환 및 3D 좌표 저장  
   - **실시간 업데이트**: QTimer로 프레임과 그래프 업데이트  
   - **시각화**: PyQtGraph의 PlotWidget과 GLViewWidget (또는 QOpenGLWidget) 이용

이 튜토리얼을 통해 PyQtGraph의 기본 개념과 사용법을 익힌 후, 여러분의 실제 프로젝트(예: 투구 기록 시스템)에 맞게 코드를 확장해 나갈 수 있습니다. 추가적으로 공식 문서(https://pyqtgraph.readthedocs.io/)와 예제들을 참고하면 더 많은 기능과 세부사항을 익힐 수 있습니다.