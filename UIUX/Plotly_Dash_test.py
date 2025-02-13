import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
import imutils
import time
import threading

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import random  # (테스트용 임의값 생성 시 사용)

########################################
# 전역 데이터 (Dash 업데이트용)
########################################
record_sheet_x = []
record_sheet_y = []
pitch_points_3d_x = []
pitch_points_3d_y = []
pitch_points_3d_z = []

# 2D 기록지 크기 (픽셀 단위)
record_sheet_width = 400
record_sheet_height = 500

########################################
# 카메라 캘리브레이션 데이터 로드
########################################
calib_data = np.load("camera_calib.npz")
camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]

########################################
# 투영 함수: 3D 점을 2D 기록지 좌표로 변환
# (원래 코드의 project_3d_to_record_sheet 함수를 참고)
########################################
def project_3d_to_record_sheet(point_3d, polygon_3d, rec_width, rec_height):
    # polygon_3d: 기준 영역 (예, ball_zone_corners2; shape: (4,3))
    # 여기서는 단순 예로, polygon의 x,y 최소/최대값을 이용해 정규화합니다.
    poly_2d = polygon_3d[:, :2]
    min_xy = poly_2d.min(axis=0)
    max_xy = poly_2d.max(axis=0)
    pt = point_3d[:2]
    norm_x = (pt[0] - min_xy[0]) / (max_xy[0] - min_xy[0] + 1e-8)
    norm_y = (pt[1] - min_xy[1]) / (max_xy[1] - min_xy[1] + 1e-8)
    rec_x = norm_x * rec_width
    rec_y = rec_height - norm_y * rec_height  # y축 반전
    return rec_x, rec_y

# ball_zone_corners2 (원래 코드에서는 ball_zone_corners를 복사하고 Z값을 조정)
ball_zone_corners = np.array([
    [-0.08, 0.09, 0],
    [ 0.08, 0.09, 0],
    [ 0.08, 0.31, 0],
    [-0.08, 0.31, 0]
], dtype=np.float32)
ball_zone_corners2 = ball_zone_corners.copy()
# 예: Z축 조정 (원래 코드에서는 ball_zone_corners2[:,2] -= 0.3)
ball_zone_corners2[:,2] -= 0.3

########################################
# 카메라 처리 함수 (별도 스레드에서 실행)
# - ARUCO 마커, mediapipe 손, 색상 기반 공 검출 등을 수행하고,
#   검출된 공의 3D 좌표(마커 좌표계)를 전역 리스트에 추가합니다.
########################################
def process_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # Mediapipe 손 검출 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    # ARUCO 사전 및 파라미터 설정
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()

    global record_sheet_x, record_sheet_y, pitch_points_3d_x, pitch_points_3d_y, pitch_points_3d_z

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # (옵션) 손 검출 등 추가 가능 – 여기서는 단순히 ARUCO와 공 검출에 집중합니다.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # 마커 포즈 추정 (첫 번째 마커 사용)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.16, camera_matrix, dist_coeffs)
            rvec = rvecs[0]
            tvec = tvecs[0]

            # 공 검출 (색상 기반; 녹색 공)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            greenLower = (29, 86, 6)
            greenUpper = (64, 255, 255)
            mask_green = cv2.inRange(hsv, greenLower, greenUpper)
            mask = cv2.erode(mask_green, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 1:
                    center = (int(x), int(y))
                    # 공의 실제 반지름 0.036(m)을 사용하여 깊이 추정
                    f_x = camera_matrix[0, 0]
                    estimated_Z = (f_x * 0.036) / radius
                    # 카메라 좌표계에서 공의 3D 좌표 계산
                    ball_3d_cam = np.array([
                        (center[0] - camera_matrix[0, 2]) * estimated_Z / camera_matrix[0, 0],
                        (center[1] - camera_matrix[1, 2]) * estimated_Z / camera_matrix[1, 1],
                        estimated_Z
                    ])
                    # 마커 좌표계로 변환: R_marker.T * (ball_3d_cam - tvec)
                    R_marker, _ = cv2.Rodrigues(rvec)
                    point_in_marker_coord = (np.dot(R_marker.T, (ball_3d_cam.reshape(3,1) - tvec.reshape(3,1)))).flatten()
                    
                    # 전역 리스트에 검출된 점 추가 (이미 여러 프레임에 걸쳐 누적)
                    rec_x, rec_y = project_3d_to_record_sheet(point_in_marker_coord, ball_zone_corners2, record_sheet_width, record_sheet_height)
                    record_sheet_x.append(rec_x)
                    record_sheet_y.append(rec_y)
                    
                    pitch_points_3d_x.append(point_in_marker_coord[0])
                    pitch_points_3d_y.append(point_in_marker_coord[1])
                    pitch_points_3d_z.append(point_in_marker_coord[2])
        # CPU 사용률을 낮추기 위한 짧은 지연
        time.sleep(0.05)

# 별도 스레드로 카메라 처리 시작
camera_thread = threading.Thread(target=process_camera, daemon=True)
camera_thread.start()

########################################
# Dash Figure 생성 함수들
########################################
def create_record_sheet_fig():
    fig = go.Figure(
        data=[go.Scatter(
            x=record_sheet_x,
            y=record_sheet_y,
            mode='markers',
            marker=dict(color='green', size=10),
            name='Pitch Points'
        )],
        layout=go.Layout(
            title="Pitch Record Sheet (2D)",
            xaxis=dict(range=[0, record_sheet_width], showgrid=False, zeroline=False),
            yaxis=dict(range=[0, record_sheet_height], showgrid=False, zeroline=False)
        )
    )
    return fig

# 고정된 3D 영역 (스트라이크존 박스, 볼 존, 볼 존2)
# 아래 값들은 원래 코드에서 계산된 값 대신 간단 예시 값으로 설정 (필요에 따라 수정)
strike_zone_trace = go.Scatter3d(
    x=[-0.08, 0.08, 0.08, -0.08, -0.08],
    y=[-0.15, -0.15, 0, 0, -0.15],
    z=[0.10, 0.10, 0.30, 0.30, 0.10],
    mode='lines',
    line=dict(color='red', width=4),
    name='Strike Zone Box'
)
ball_zone_trace = go.Scatter3d(
    x=[-0.08, 0.08, 0.08, -0.08, -0.08],
    y=[0.09, 0.09, 0.31, 0.31, 0.09],
    z=[0, 0, 0, 0, 0],
    mode='lines',
    line=dict(color='blue', width=4),
    name='Ball Zone'
)
ball_zone2_trace = go.Scatter3d(
    x=[-0.08, 0.08, 0.08, -0.08, -0.08],
    y=[0.09, 0.09, 0.31, 0.31, 0.09],
    z=[-0.3, -0.3, -0.3, -0.3, -0.3],
    mode='lines',
    line=dict(color='green', width=4),
    name='Ball Zone 2'
)

def create_three_d_fig():
    fig = go.Figure(
        data=[
            strike_zone_trace,
            ball_zone_trace,
            ball_zone2_trace,
            go.Scatter3d(
                x=pitch_points_3d_x,
                y=pitch_points_3d_y,
                z=pitch_points_3d_z,
                mode='markers',
                marker=dict(color='orange', size=5),
                name='Pitch Points'
            )
        ],
        layout=go.Layout(
            title="3D Interactive Pitching Zone",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )
    )
    return fig

########################################
# Dash 앱 생성 및 레이아웃 설정
########################################
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Strike Zone Dashboard"),
    html.Div([
        dcc.Graph(id='record-sheet', figure=create_record_sheet_fig(), style={'flex': '1'}),
        dcc.Graph(id='three-d-plot', figure=create_three_d_fig(), style={'flex': '1'})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # 1초마다 업데이트
        n_intervals=0
    )
])

# Dash 콜백: Interval마다 전역 리스트를 바탕으로 Figure를 재생성하여 업데이트
@app.callback(
    [Output('record-sheet', 'figure'),
     Output('three-d-plot', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dash_figures(n_intervals):
    updated_record_sheet_fig = create_record_sheet_fig()
    updated_three_d_fig = create_three_d_fig()
    return updated_record_sheet_fig, updated_three_d_fig

########################################
# 메인: Dash 앱 실행
########################################
if __name__ == '__main__':
    app.run_server(debug=True)
