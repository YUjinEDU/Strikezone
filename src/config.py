import os
import numpy as np
from datetime import datetime

# 캘리브레이션 파일 경로
CALIBRATION_PATH = "camera_calib.npz"

# ArUco 마커 설정
ARUCO_MARKER_SIZE = 0.16
SKIP_FRAMES = 1

# 공 크기
BALL_RADIUS_REAL = 0.036

# 로그 설정
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
LOG_FILENAME = os.path.join(log_dir, f"strike_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 색상 기반 객체 추적 범위
GREEN_LOWER = (29, 86, 6)
GREEN_UPPER = (64, 255, 255)
# RED_LOWER1 = (0, 70, 50)
# RED_UPPER1 = (10, 255, 255)
# RED_LOWER2 = (170, 70, 50)
# RED_UPPER2 = (180, 255, 255)



"""
스트라이크존 Small
"""
# 영역 설정
# STRIKE_ZONE_CORNERS = np.array([
#     [-0.08, 0.15, 0],  # Bottom-left
#     [ 0.08, 0.15, 0],  # Bottom-right
#     [ 0.08, 0.25, 0],  # Top-right
#     [-0.08, 0.25, 0],  # Top-left
# ], dtype=np.float32)

# BALL_ZONE_CORNERS = np.array([
#     [-0.08, 0.13, 0],  # Bottom-left
#     [ 0.08, 0.13, 0],  # Bottom-right
#     [ 0.08, 0.27, 0],  # Top-right
#     [-0.08, 0.27, 0],  # Top-left
# ], dtype=np.float32)


"""
스트라이크존 Big
"""
#영역 설정
STRIKE_ZONE_CORNERS = np.array([
    [-0.15, 0.25, 0],  # Bottom-left
    [ 0.15, 0.25, 0],  # Bottom-right
    [ 0.15, 0.65, 0],  # Top-right
    [-0.15, 0.65, 0],  # Top-left
], dtype=np.float32)


BALL_ZONE_CORNERS = np.array([
    [-0.15, 0.24, 0],  # Bottom-left
    [ 0.15, 0.24, 0],  # Bottom-right
    [ 0.15, 0.66, 0],  # Top-right
    [-0.15, 0.66, 0],  # Top-left
], dtype=np.float32)


ZONE_Z_DIFF = 0.20  # 스트라이크존과 볼존의 Z축 차이 (20cm)
# 오프셋 설정
BOX_X_OFFSET = 0.00   # 좌우 오프셋 없음
BOX_Y_MIN = -ZONE_Z_DIFF     # 뒤쪽으로 10cm
BOX_Y_MAX = 0.0       # 마커 위치까지
BOX_Z_MIN = STRIKE_ZONE_CORNERS[0, 1]      # 바닥에서 25cm
BOX_Z_MAX = STRIKE_ZONE_CORNERS[2, 1]      # 최대 높이 65cm

# 박스 크기를 스트라이크 존에서 자동 계산
def calculate_box_from_strike_zone(strike_zone, x_offset=0.0):
    x_min = min(strike_zone[:, 0]) - x_offset
    x_max = max(strike_zone[:, 0]) + x_offset
    
    return (
        np.array([x_min, BOX_Y_MIN, BOX_Z_MIN]), 
        np.array([x_max, BOX_Y_MAX, BOX_Z_MAX])
    )

# 박스 설정
BOX_MIN, BOX_MAX = calculate_box_from_strike_zone(STRIKE_ZONE_CORNERS, BOX_X_OFFSET)



BOX_EDGES = [
    (0,1), (1,2), (2,3), (3,0),  # 아래면
    (4,5), (5,6), (6,7), (7,4),  # 윗면
    (0,4), (1,5), (2,6), (3,7),  # 수직 엣지
    (2,8), (8,3),               # 삼각형 아래면
    (7,9), (9,6),               # 삼각형 윗면
    (9,8)                       # 삼각형 수직
]

# 회전 행렬
ROTATION_MATRIX = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float32)

# 시각화 설정
RECORD_SHEET_WIDTH = 0.6
RECORD_SHEET_HEIGHT = 0.5

# 색상 설정
BALL_COLOR = (204, 204, 0)
STRIKE_COLOR = (0, 200, 200)


# main.py의 판정 및 표시에 사용될 수 있는 상수
PITCH_EVENT_COOLDOWN = 2.0  # 스트라이크/볼 판정 후 다음 판정까지 최소 시간 (초)
TRAJECTORY_DISPLAY_DURATION = 2.0  # 판정 후 궤적 표시 시간 (초)
FPS_UPDATE_INTERVAL = 0.5  # 화면 FPS 표시 업데이트 간격 (초)