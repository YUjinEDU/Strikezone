import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

# 그림 설정
plt.figure(figsize=(12, 8))
ax = plt.gca()

# 컴포넌트 정의
components = {
    'camera': (0.1, 0.7, 0.15, 0.15, 'Camera\nSystem'),
    'aruco': (0.35, 0.7, 0.15, 0.15, 'ArUco\nMarker\nDetection'),
    'ball_tracking': (0.6, 0.7, 0.15, 0.15, 'Ball\nTracking'),
    'kalman': (0.85, 0.7, 0.15, 0.15, 'Kalman\nFilter'),
    
    'coord_transform': (0.35, 0.4, 0.15, 0.15, '3D Coordinate\nTransformation'),
    'strike_judge': (0.6, 0.4, 0.15, 0.15, 'Strike/Ball\nJudgement'),
    'velocity': (0.85, 0.4, 0.15, 0.15, 'Velocity\nCalculation'),
    
    'dashboard': (0.6, 0.1, 0.3, 0.15, 'Dashboard Visualization')
}

# 컴포넌트 그리기
for name, (x, y, w, h, label) in components.items():
    rect = Rectangle((x, y), w, h, facecolor='skyblue', edgecolor='black', alpha=0.7)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)

# 화살표 그리기
arrows = [
    # 상단 흐름
    ('camera', 'aruco'),
    ('camera', 'ball_tracking'),
    ('aruco', 'coord_transform'),
    ('ball_tracking', 'kalman'),
    ('kalman', 'coord_transform'),
    
    # 중간 흐름
    ('coord_transform', 'strike_judge'),
    ('coord_transform', 'velocity'),
    
    # 하단 흐름
    ('strike_judge', 'dashboard'),
    ('velocity', 'dashboard')
]

for start, end in arrows:
    start_comp = components[start]
    end_comp = components[end]
    
    # 시작점과 끝점 계산
    if start_comp[0] < end_comp[0]:  # 오른쪽으로 이동
        start_x = start_comp[0] + start_comp[2]
        start_y = start_comp[1] + start_comp[3]/2
    else:  # 왼쪽으로 이동
        start_x = start_comp[0]
        start_y = start_comp[1] + start_comp[3]/2
    
    if end_comp[1] < start_comp[1]:  # 아래로 이동
        end_x = end_comp[0] + end_comp[2]/2
        end_y = end_comp[1] + end_comp[3]
    else:  # 위로 이동 또는 수평 이동
        end_x = end_comp[0]
        end_y = end_comp[1] + end_comp[3]/2
    
    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), 
                           arrowstyle='->', color='black', linewidth=1.5,
                           connectionstyle='arc3,rad=0.1')
    ax.add_patch(arrow)

# 입력/출력 표시
input_circle = Circle((0.05, 0.75), 0.03, facecolor='lightgreen', edgecolor='black', alpha=0.7)
ax.add_patch(input_circle)
ax.text(0.05, 0.75, 'Video\nInput', ha='center', va='center', fontsize=8)

output_circle = Circle((0.95, 0.15), 0.03, facecolor='lightcoral', edgecolor='black', alpha=0.7)
ax.add_patch(output_circle)
ax.text(0.95, 0.15, 'User\nInterface', ha='center', va='center', fontsize=8)

# 제목 및 축 설정
plt.title('AR Strike Zone System Architecture', fontsize=16)
plt.axis('off')
plt.tight_layout()

# 저장
plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("시스템 구성도가 생성되었습니다: system_architecture.png") 