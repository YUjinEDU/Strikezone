import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # 3D 지원

# VideoWidget: OpenCV 영상 표시용 Tkinter 레이블
class VideoWidget(tk.Label):
    def __init__(self, parent, cam_index=2, width=640, height=480):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # PIL 이미지로 변환 후 PhotoImage 생성
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.imgtk = imgtk  # 참조 유지 (가비지 컬렉션 방지)
            self.config(image=imgtk)
        # 30ms마다 영상 갱신 (약 30fps)
        self.after(60, self.update_video)

    def release(self):
        if self.cap is not None:
            self.cap.release()

# GraphWidget: 2D 그래프 (matplotlib)
class GraphWidget(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.plot_example()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_example(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.ax.plot(x, y)
        self.ax.set_title("2D Example Graph")

# Interactive3DGraphWidget2: 3D 그래프 (matplotlib)
class Interactive3DGraphWidget2(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.plot_strike_zone()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_strike_zone(self):
        # 스트라이크존 박스의 최소, 최대 좌표 (예시: 필요에 따라 수정)
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
            (0,1), (1,2), (2,3), (3,0),  # 아래면
            (4,5), (5,6), (6,7), (7,4),  # 윗면
            (0,4), (1,5), (2,6), (3,7)   # 수직 엣지
        ]
        for edge in edges:
            xs = [corners[edge[0]][0], corners[edge[1]][0]]
            ys = [corners[edge[0]][1], corners[edge[1]][1]]
            zs = [corners[edge[0]][2], corners[edge[1]][2]]
            self.ax.plot(xs, ys, zs, color='yellow')
        self.ax.set_title("3D Strike Zone")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("ARUCO Tracker with Strike Zone (Tkinter)")
    # 창 크기 설정
    root.geometry("1200x800")

    # 메인 프레임 구성: 좌측에 비디오, 우측에 그래프 영역
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 좌측 비디오 영역
    video_frame = tk.Frame(main_frame)
    video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    video_widget = VideoWidget(video_frame, cam_index=2, width=640, height=480)
    video_widget.pack(fill=tk.BOTH, expand=True)

    # 좌측과 우측 사이에 분할선
    separator_vert = ttk.Separator(main_frame, orient=tk.VERTICAL)
    separator_vert.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

    # 우측 그래프 영역
    graph_frame = tk.Frame(main_frame)
    graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 상단 2D 그래프
    graph_widget = GraphWidget(graph_frame)
    graph_widget.pack(fill=tk.BOTH, expand=True)

    # 상단과 하단 사이의 분할선
    separator_horiz = ttk.Separator(graph_frame, orient=tk.HORIZONTAL)
    separator_horiz.pack(fill=tk.X, padx=5, pady=5)

    # 하단 3D 그래프
    interactive3d_widget = Interactive3DGraphWidget2(graph_frame)
    interactive3d_widget.pack(fill=tk.BOTH, expand=True)

    root.mainloop()
    video_widget.release()