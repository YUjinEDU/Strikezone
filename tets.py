import plotly.graph_objects as go
import numpy as np
import time
from IPython.display import display

# ============================================================
# 1. 2D 투구 기록지 (Record Sheet) 설정 (Plotly FigureWidget)
# ============================================================

# 기록지(사각형) 크기: (가로 x 세로)
record_sheet_width = 400
record_sheet_height = 600

# 2D 기록지 FigureWidget 생성: 사각형 테두리와 scatter trace를 추가
record_sheet_fig = go.FigureWidget()
record_sheet_fig.add_scatter(
    x=[], y=[], mode='markers',
    marker=dict(color='green', size=10),
    name='Pitch Points'
)
record_sheet_fig.update_layout(
    title="Pitch Record Sheet (2D)",
    xaxis=dict(range=[0, record_sheet_width], showgrid=False, zeroline=False),
    yaxis=dict(range=[0, record_sheet_height], showgrid=False, zeroline=False),
    width=500,
    height=700,
    shapes=[
        dict(
            type="rect",
            x0=0, y0=0,
            x1=record_sheet_width, y1=record_sheet_height,
            line=dict(color="RoyalBlue", width=3)
        )
    ]
)
# 투구 기록지는 보통 위쪽이 투구측이도록 y축을 반전시킵니다.
record_sheet_fig.update_yaxes(autorange="reversed")

# 화면에 표시 (Jupyter Notebook 환경에서)
display(record_sheet_fig)


# ============================================================
# 2. 3D 인터랙티브 그래프 설정 (Plotly FigureWidget)
# ============================================================

# --- 도형 데이터 (예제용 더미 데이터) ---
# (실제 코드에서는 기존 카메라 캘리브레이션/마커 추정 결과로 구한 좌표를 사용하면 됩니다.)

# (a) 스트라이크존 박스: 여기서는 간단하게 박스의 아래 면(4점)을 예제로 사용합니다.
strike_box = np.array([
    [-0.1, -0.1, -0.1],
    [ 0.1, -0.1, -0.1],
    [ 0.1,  0.1, -0.1],
    [-0.1,  0.1, -0.1]
])
# (b) ball_zone_corners: 3D 평면상의 사각형 (4점)
ball_zone_corners_3d = np.array([
    [-0.08, 0.09, -0.10],
    [ 0.08, 0.09, -0.10],
    [ 0.08, 0.31, -0.10],
    [-0.08, 0.31, -0.10]
])
# (c) ball_zone_corners2: ball_zone_corners에서 z축 방향으로 이동 (예: -0.2 추가)
ball_zone_corners2_3d = ball_zone_corners_3d.copy()
ball_zone_corners2_3d[:, 2] -= 0.2

def create_3d_polygon_trace(points, color, name):
    """ 주어진 점(2차원 배열)을 이어 닫힌 선분 형태의 3D trace 생성 """
    points_closed = np.vstack([points, points[0]])  # 시작점을 끝에 추가하여 닫음
    trace = go.Scatter3d(
        x=points_closed[:, 0],
        y=points_closed[:, 1],
        z=points_closed[:, 2],
        mode='lines',
        line=dict(color=color, width=5),
        name=name
    )
    return trace

# 각 도형의 trace 생성
strike_box_trace = create_3d_polygon_trace(strike_box, color='red', name='Strike Zone')
ball_zone_trace = create_3d_polygon_trace(ball_zone_corners_3d, color='blue', name='Ball Zone')
ball_zone2_trace = create_3d_polygon_trace(ball_zone_corners2_3d, color='green', name='Ball Zone 2')

# 3D 산점도: 투구 점들을 실시간 업데이트할 trace (초기에는 빈 리스트)
pitch_points_3d_trace = go.Scatter3d(
    x=[], y=[], z=[],
    mode='markers',
    marker=dict(color='orange', size=5),
    name='Pitch Points'
)

# 3D FigureWidget 생성
three_d_fig = go.FigureWidget(data=[strike_box_trace, ball_zone_trace, ball_zone2_trace, pitch_points_3d_trace])
three_d_fig.update_layout(
    title="3D Interactive Pitching Zone",
    scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False)
    
    ),
    width=700,
    height=700
)
display(three_d_fig)


# ============================================================
# 3. 실시간 업데이트 시뮬레이션
# ============================================================

# 아래 함수는 ball_zone_corners2 평면 내에서 랜덤한 점을 생성합니다.
# (실제 코드에서는 카메라/마커 처리 결과로 얻은 점의 3D 좌표가 들어옵니다.)
def random_point_in_polygon(polygon):
    """
    polygon: (4,3) 배열. 평면상 4개 점.
    간단한 방법: polygon의 한 꼭짓점을 기준으로 다른 두 꼭짓점 방향의 선형 결합으로 생성.
    """
    a = np.random.random()
    b = np.random.random()
    if a + b > 1:
        a, b = 1 - a, 1 - b
    point = polygon[0] + a * (polygon[1] - polygon[0]) + b * (polygon[3] - polygon[0])
    return point

# 2D 기록지에 점을 찍기 위해, 3D 점을 2D 좌표로 변환하는 간단한 매핑 함수
# (여기서는 ball_zone_corners2_3d의 (x,y) 영역을 기록지 전체로 선형 매핑합니다)
def project_3d_to_record_sheet(point_3d, polygon_3d, rec_width, rec_height):
    # 평면(3D)의 점들을 2D로 간주 (z 무시)
    poly_2d = polygon_3d[:, :2]
    min_xy = poly_2d.min(axis=0)
    max_xy = poly_2d.max(axis=0)
    pt = point_3d[:2]
    norm_x = (pt[0] - min_xy[0]) / (max_xy[0] - min_xy[0] + 1e-8)
    norm_y = (pt[1] - min_xy[1]) / (max_xy[1] - min_xy[1] + 1e-8)
    # 기록지에서는 y축을 반전 (위쪽이 투구측)
    rec_x = norm_x * rec_width
    rec_y = rec_height - norm_y * rec_height
    return rec_x, rec_y

# 데이터 저장용 리스트 (2D와 3D 각각)
record_sheet_x = []
record_sheet_y = []
pitch_points_3d_x = []
pitch_points_3d_y = []
pitch_points_3d_z = []

# 시뮬레이션: 20회 반복하여 점 추가 (실제 환경에서는 카메라 처리 루프 내에서 업데이트)
for i in range(20):
    # (실제 검출된 점) ball_zone_corners2 평면 내의 랜덤 3D 점 생성
    new_point_3d = random_point_in_polygon(ball_zone_corners2_3d)
    pitch_points_3d_x.append(new_point_3d[0])
    pitch_points_3d_y.append(new_point_3d[1])
    pitch_points_3d_z.append(new_point_3d[2])
    
    # 2D 기록지 좌표로 매핑
    rec_x, rec_y = project_3d_to_record_sheet(new_point_3d, ball_zone_corners2_3d, record_sheet_width, record_sheet_height)
    record_sheet_x.append(rec_x)
    record_sheet_y.append(rec_y)
    
    # 2D 기록지 업데이트
    with record_sheet_fig.batch_update():
        record_sheet_fig.data[0].x = record_sheet_x
        record_sheet_fig.data[0].y = record_sheet_y
    
    # 3D 그래프 업데이트 (산점도 trace의 데이터 업데이트)
    with three_d_fig.batch_update():
        three_d_fig.data[-1].x = pitch_points_3d_x
        three_d_fig.data[-1].y = pitch_points_3d_y
        three_d_fig.data[-1].z = pitch_points_3d_z
    
    time.sleep(1)  # 실시간 업데이트를 모방 (실제 환경에서는 프레임 단위 업데이트)
