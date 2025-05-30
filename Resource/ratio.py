import numpy as np

def scale_zone(corners, scale_factor):
    """
    주어진 꼭짓점 배열(corners)을 영역의 중심을 기준으로 scale_factor만큼 확대한 새로운 좌표를 반환합니다.
    """
    center = np.mean(corners, axis=0)
    return (corners - center) * scale_factor + center

# 확대한 비율 (예: 1.5배)
scale_factor = 2

# 기존 Strike Zone (타석 존)
strike_zone = np.array([
    [-0.08, 0.15, 0],  # Bottom-left
    [ 0.08, 0.15, 0],  # Bottom-right
    [ 0.08, 0.25, 0],  # Top-right
    [-0.08, 0.25, 0],  # Top-left
], dtype=np.float32)

# 기존 Ball Zone (볼 존)
ball_zone = np.array([
    [-0.08, 0.09, 0],  # Bottom-left
    [ 0.08, 0.09, 0],  # Bottom-right
    [ 0.08, 0.31, 0],  # Top-right
    [-0.08, 0.31, 0],  # Top-left
], dtype=np.float32)

# 영역 확대
scaled_strike_zone = scale_zone(strike_zone, scale_factor)
scaled_ball_zone = scale_zone(ball_zone, scale_factor)

print("Scaled Strike Zone Corners:")
print(scaled_strike_zone)

print("Scaled Ball Zone Corners:")
print(scaled_ball_zone)

# BOX 영역 (3차원 박스) 확대 처리
BOX_MIN = np.array([-0.08, -0.15, 0.10])
BOX_MAX = np.array([ 0.08,  0.0, 0.30])

# 박스의 중심 및 반크기 계산
box_center = (BOX_MIN + BOX_MAX) / 2
box_half_size = (BOX_MAX - BOX_MIN) / 2

# 확대한 박스 반크기 계산
scaled_box_half_size = box_half_size * scale_factor

# 확대한 BOX_MIN과 BOX_MAX 계산
scaled_box_min = box_center - scaled_box_half_size
scaled_box_max = box_center + scaled_box_half_size

print("Scaled BOX_MIN:")
print(scaled_box_min)

print("Scaled BOX_MAX:")
print(scaled_box_max)
