import os
import numpy as np
from datetime import datetime

# 캘리브레이션 파일 (repo에 존재하는 파일명으로 설정)
# 실행 위치에 따라 상대경로를 맞춰 주세요. (예: 프로젝트 루트에서 실행 시 "src/camera_calib.npz")
CALIBRATION_PATH = "camera_calib.npz"

# ArUco 마커 설정
ARUCO_MARKER_SIZE = 0.16
SKIP_FRAMES = 1

# 공 실제 반지름 (m)
BALL_RADIUS_REAL = 0.036

# 로그 설정
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
LOG_FILENAME = os.path.join(log_dir, f"strike_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 색상 기반 객체 추적 범위 (HSV)
GREEN_LOWER = (29, 86, 6)
GREEN_UPPER = (64, 255, 255)

"""
좌표계 정의(마커 좌표계, 단위 m)
- X: 좌우(+는 오른쪽)
- Y: 깊이/앞뒤(+는 포수/타자 방향 등 진행 방향)
- Z: 높이(+는 위)
"""

# 스트라이크존(앞 판정면, y=0 평면) 코너들: [x, y, z]
STRIKE_ZONE_CORNERS = np.array([
    [-0.15, 0.00, 0.25],  # Bottom-left
    [ 0.15, 0.00, 0.25],  # Bottom-right
    [ 0.15, 0.00, 0.65],  # Top-right
    [-0.15, 0.00, 0.65],  # Top-left
], dtype=np.float32)

# 판정용/시각화용 폴리곤(앞면) - 위와 동일
BALL_ZONE_CORNERS = STRIKE_ZONE_CORNERS.copy()

# 두 판정면 간 깊이 간격(= Y축 오프셋, m)
ZONE_DEPTH = 0.20

# 박스(시각화용) 설정
BOX_X_OFFSET = 0.00                  # 좌우 여유
BOX_Y_MIN = 0.00                     # 앞 판정면 y
BOX_Y_MAX = ZONE_DEPTH               # 뒤 판정면 y
BOX_Z_MIN = float(np.min(STRIKE_ZONE_CORNERS[:, 2]))  # 최소 높이
BOX_Z_MAX = float(np.max(STRIKE_ZONE_CORNERS[:, 2]))  # 최대 높이

def calculate_box_from_strike_zone(strike_zone, x_offset=0.0):
    x_min = float(np.min(strike_zone[:, 0])) - x_offset
    x_max = float(np.max(strike_zone[:, 0])) + x_offset
    return (
        np.array([x_min, BOX_Y_MIN, BOX_Z_MIN], dtype=np.float32),
        np.array([x_max, BOX_Y_MAX, BOX_Z_MAX], dtype=np.float32),
    )

BOX_MIN, BOX_MAX = calculate_box_from_strike_zone(STRIKE_ZONE_CORNERS, BOX_X_OFFSET)

# 박스 엣지(0~7 정육면체 코너만 사용)
BOX_EDGES = [
    (0,1), (1,2), (2,3), (3,0),  # 앞면(y=BOX_Y_MIN)
    (4,5), (5,6), (6,7), (7,4),  # 뒷면(y=BOX_Y_MAX)
    (0,4), (1,5), (2,6), (3,7),  # 세로 엣지
]

# 시각화 설정
RECORD_SHEET_WIDTH = 0.6
RECORD_SHEET_HEIGHT = 0.5

# 색상
BALL_COLOR = (204, 204, 0)
STRIKE_COLOR = (0, 200, 200)

# 판정/표시 타이밍
PITCH_EVENT_COOLDOWN = 2.0      # (초)
TRAJECTORY_DISPLAY_DURATION = 2.0
FPS_UPDATE_INTERVAL = 0.5