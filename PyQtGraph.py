import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGraphicsRectItem
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout

class RecordSheetWidget(pg.PlotWidget):
    def __init__(self, width=400, height=600, parent=None):
        super().__init__(parent)
        # 짙은 파랑 배경
        self.setBackground('lightblue')
        # 크기 설정
        self.setFixedSize(width, height)
        self.setMouseEnabled(x=False, y=False)
        self.showGrid(x=False, y=False)
        self.setXRange(0, width)
        self.setYRange(0, height)
        # 기록지 테두리 추가: QtGraphicsRectItem을 사용 (PyQtGraph 내부 아이템)
        rect = QGraphicsRectItem(0, 0, width, height)
        rect.setPen(pg.mkPen(color='b', width=2))
        self.addItem(rect)
        # y축 반전: 기록지에서는 보통 위쪽이 투구측이므로
        self.invertY(True)
        # Scatter plot: 투구점 그리기
        self.scatter = self.plot([], [], pen=None, symbol='o', symbolBrush='g', symbolSize=8)

               # x, y 축 범례 숨기기
        self.getPlotItem().hideAxis('bottom')
        self.getPlotItem().hideAxis('left')
        
        self.timer = QTimer()
        xdata = np.random.rand(10)
        ydata = np.random.rand(10)
        self.timer.timeout.connect(self.updateData(xdata, ydata))
        self.timer.start(50)

    def updateData(self, xdata, ydata):
        self.scatter.setData(x=xdata, y=ydata)


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
    window = RecordSheetWidget()
    window.show()
    sys.exit(app.exec())
