import time
import cv2
import plotly.graph_objs as go
import numpy as np
from config import BALL_COLOR, STRIKE_COLOR

class TextEffect:
    def __init__(self):
        self.effects = []
    
    def add_effect(self, text, duration=2.0):
        """텍스트 효과 추가"""
        self.effects.append({
            'start_time': time.time(),
            'duration': duration,
            'text': text
        })
    
    def add_strike_effect(self):
        """STRIKE! 효과 추가"""
        self.add_effect("STRIKE!")
    
    def add_ball_effect(self):
        """BALL! 효과 추가"""
        self.add_effect("BALL!")
    
    def draw(self, frame, result=""):
        """텍스트 효과 그리기"""
        now = time.time()
        alive = []
        for eff in self.effects:
            age = now - eff['start_time']
            if age < eff['duration']:
                # 텍스트 그리기
                h, w = frame.shape[:2]
                
                if result == "strike":
                    cv2.putText(frame, eff['text'], (w-200, 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 200, 200), 3, cv2.LINE_AA)
                    alive.append(eff)
                else:
                    cv2.putText(frame, eff['text'], (w-150, 50),
                                cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                    alive.append(eff)
                
        self.effects = alive

class PlotlyVisualizer:
    @staticmethod
    def create_3d_polygon_trace(points, color, name):
        """3D 다각형 trace 생성"""
        pts_closed = np.vstack([points, points[0]])
        trace = go.Scatter3d(
            x=pts_closed[:,0],
            y=pts_closed[:,1],
            z=pts_closed[:,2],
            mode='lines',
            line=dict(color=color, width=4),
            name=name
        )
        return trace
    
    @staticmethod
    def create_3d_trajectory_trace(trajectory_points, color='white', width=4, name='Trajectory'):
        """3D 궤적 trace 생성"""
        if len(trajectory_points) < 2:
            return None

        x_coords = [float(point[0]) for point in trajectory_points]
        y_coords = [float(point[1]) for point in trajectory_points]
        z_coords = [float(point[2]) for point in trajectory_points]

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            line=dict(color=color, width=width),
            name=name
        )
    
    @staticmethod
    def create_box_trace(corners_3d, box_edges, color='blue'):
        """박스의 선 trace 생성"""
        lines = []
        for e in box_edges:
            p1 = corners_3d[e[0]]
            p2 = corners_3d[e[1]]
            x_ = [float(p1[0]), float(p2[0])]
            y_ = [float(p1[1]), float(p2[1])]
            z_ = [float(p1[2]), float(p2[2])]
            line = go.Scatter3d(
                x=x_, y=y_, z=z_,
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False
            )
            lines.append(line)
        return lines
    
    @staticmethod
    def create_2d_polygon_trace(points, color, name):
        """2D 다각형 trace 생성"""
        pts_closed = np.vstack([points, points[0]])
        return go.Scatter(
            x=pts_closed[:,0].tolist(),
            y=pts_closed[:,2].tolist(),  # z 좌표를 y로 사용
            mode='lines',
            line=dict(color=color, width=2),
            name=name
        )