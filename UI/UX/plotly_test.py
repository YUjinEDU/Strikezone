import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import webbrowser

# (A) 박스 모서리 계산 함수
def get_box_corners(box_min, box_max):
    """
    box_min, box_max: (3,) = [xmin, ymin, zmin], [xmax,ymax,zmax]
    return -> shape (8,3) 8개 꼭짓점
    """
    x0, y0, z0 = box_min
    x1, y1, z1 = box_max
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=float)
    return corners

# (B) 박스 엣지 연결 (각 꼭짓점 index 쌍)
box_edges = [
    (0,1), (1,2), (2,3), (3,0),  # 아래면
    (4,5), (5,6), (6,7), (7,4),  # 윗면
    (0,4), (1,5), (2,6), (3,7)   # 수직 엣지
]

def create_box_trace(corners_3d, color='blue'):
    """
    corners_3d: (8,3) array
    -> returns list of plotly Scatter3d for each line
    """
    lines = []
    for e in box_edges:
        p1 = corners_3d[e[0]]
        p2 = corners_3d[e[1]]
        # x,y,z 좌표를 따로
        x_ = [p1[0], p2[0]]
        y_ = [p1[1], p2[1]]
        z_ = [p1[2], p2[2]]
        line = go.Scatter3d(
            x=x_, y=y_, z=z_,
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False
        )
        lines.append(line)
    return lines

# (C) 샘플 데이터
box_min = np.array([-0.215, 0.48, 0.0])
box_max = np.array([ 0.215, 0.98, 0.48])
corners_3d = get_box_corners(box_min, box_max)

# 예: 공 찍힌 위치들 (10개 샘플)
ball_points = [
    (0.0,   0.6,  0.1),
    (0.1,   0.65, 0.3),
    (-0.15, 0.75, 0.2),
    (0.08,  0.8,  0.4),
    (0.2,   0.7,  0.45),
    (-0.1,  0.95, 0.3),
    (0.07,  0.9,  0.35),
    (0.09,  0.85, 0.3),
    (-0.02, 0.7,  0.2),
    (0.3,   0.52, 0.52),
    (0.3,   0.63, 0.15),
    (0.3,   0.63, 0.15)
]

# (D) 박스(스트라이크 존) Trace
box_traces = create_box_trace(corners_3d, color='blue')

# (E) 공 점들 Scatter3D
ball_x = [p[0] for p in ball_points]
ball_y = [p[1] for p in ball_points]
ball_z = [p[2] for p in ball_points]

ball_trace = go.Scatter3d(
    x=ball_x, y=ball_y, z=ball_z,
    mode='markers',
    marker=dict(size=5, color='red'),
    name='Ball Points'
)

# (F) Layout 및 시각화
fig = go.Figure(data= box_traces + [ball_trace])
fig.update_layout(
    scene = dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'
    ),
    title="3D Strike Zone & Ball Points"
)
# fig.write_html('3d_plot.html')  # 그래프를 HTML 파일로 저장
# webbrowser.open('3d_plot.html')  # HTML 파일을 새 브라우저 창으로 열기
fig.show()  # 렌더러 인자 제거하여 기본 설정 사용
