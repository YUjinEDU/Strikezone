import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
from collections import deque
import imutils
import time
import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
import threading
import plotly.graph_objs as go # 그래프 생성 라이브러리


last_time = 0.0
last_draw_time = 0.0
selected_camera_index = None  # 선택된 카메라 인덱스를 저장하는 변수
preview_stop_event = threading.Event() # 미리보기 중지 이벤트

start_time = None  # 속도 측정을 시작하는 시간
end_time = None    # 속도 측정을 종료하는 시간
distance_to_plate = 1  # 투구 거리 (미터)


# Mediapipe 손 추적 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 카메라 캘리브레이션 데이터 로드
calib_data = np.load("camera_calib.npz")
print("Keys in the calibration file:", calib_data.files)



def init_kalman_3d():
    """
    상태: (x, y, z, vx, vy, vz) -> 6차원
    측정: (x, y, z) -> 3차원
    """
    kf = cv2.KalmanFilter(6, 3)  # stateDim=6, measDim=3

    # Transition matrix (A)
    # 예: 등속도 모델
    # x' = x + vx
    # y' = y + vy
    # z' = z + vz
    # vx' = vx
    # vy' = vy
    # vz' = vz
    kf.transitionMatrix = np.array([
        [1,0,0,1,0,0],
        [0,1,0,0,1,0],
        [0,0,1,0,0,1],
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
    ], dtype=np.float32)

    # 측정행렬 (H)
    # 관측: x, y, z만 측정
    # H = [1 0 0 0 0 0
    #      0 1 0 0 0 0
    #      0 0 1 0 0 0]
    kf.measurementMatrix = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0]
    ], dtype=np.float32)

    # 공정 잡음, 측정 잡음 (예시값)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
    # 초기 오차 공분산
    kf.errorCovPost = np.eye(6, dtype=np.float32)

    # 초기 상태 추정치 (x,y,z,vx,vy,vz)
    kf.statePost = np.zeros((6,1), dtype=np.float32)

    return kf

# 전역 칼만 필터
kalman_3d = init_kalman_3d()

# 공 3D 궤적 저장 (보정 후 좌표)
kalman_trajectory_strike = []
kalman_trajectory_ball   = []



#asdasd
camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]

# 공의 실제 반지름 (미터 단위)
ball_radius_real = 0.036

# 공중에 띄울 스트라이크 존의 3D 좌표 (마커 기준으로 상대적인 위치)
# strike_zone_corners = np.array([
#     [-0.215, 0.2837, 0],  # Bottom-left
#     [ 0.215, 0.2837, 0],  # Bottom-right
#     [ 0.215, 0.7851, 0],  # Top-right
#     [-0.215, 0.7851, 0]   # Top-left
# ], dtype=np.float32)
# box_min = np.array([ -0.215,  -0.25, 0.00 ])  # x=-21.5cm, y=48cm,  z=0cm
# box_max = np.array([  0.215,  0.25, 0.48 ])  # x=21.5cm,  y=98cm,  z=48cm

strike_zone_corners = np.array([
    [-0.08, 0.15, 0],  # Bottom-left
    [ 0.08, 0.15, 0],  # Bottom-right
    [ 0.08, 0.25, 0],  # Top-right
    [-0.08, 0.25, 0],  # Top-left

], dtype=np.float32)

ball_zone_corners = np.array([
    [-0.08, 0.09, -0.10],  # Bottom-left
    [ 0.08, 0.09, -0.10],  # Bottom-right
    [ 0.08, 0.31, -0.10],  # Top-right
    [-0.08, 0.31, -0.10],  # Top-left

], dtype=np.float32)

ball_zone_corners2 = ball_zone_corners.copy()
ball_zone_corners2[:, 2] -= 0.2

box_edges = [
    (0,1), (1,2), (2,3), (3,0),  # 아래면
    (4,5), (5,6), (6,7), (7,4),  # 윗면
    (0,4), (1,5), (2,6), (3,7),  # 수직 엣지

    (4,8), (8,5), # 삼각형 아래면
    (7,9), (9,6), # 삼각형 윗면
    (9,8)    # 삼각형 수직
]



box_min = np.array([ -0.08,  -0.1, 0.10 ])  
box_max = np.array([  0.08,  0.1, 0.30 ])  

def get_box_corners_3d(box_min, box_max):
    """
    box_min: [xmin, ymin, zmin]
    box_max: [xmax, ymax, zmax]
    return -> shape (11,3) -> 8 corners of the box + 3 corners of a triangle
    """
    x0, y0, z0 = box_min
    x1, y1, z1 = box_max

    # 박스 8개 꼭짓점
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float32)

    #삼각형 꼭짓점 (Z, Y축의 위치가 바뀐 상태!!)
    triangle_points = np.array([
        [(x0+x1)/2, (z0+z1)/2+z0, y1],  # 8번
        [(x0+x1)/2, (z0+z1)/2+z0, y1+0.2], # 9번
    ], dtype=np.float32)

    # triangle_points = np.array([
    #     [(x0+x1)/2,y0 ,(z0+z1)/2+z0],  # 8번
    #     [(x0+x1)/2, y1,(z0+z1)/2+z0]     # 9번
    # ], dtype=np.float32)

    # 삼각형 꼭짓점 합치기 (총 11개)
    corners = np.concatenate((corners, triangle_points), axis=0)
    return corners

def project_box_corners_2d(corners_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """
    corners_3d: shape (8,3) box corners in marker coords
    return -> shape (8,2) 2D pixel coords
    """
    # (8,1,3) 형태로 reshape
    corners_3d_reshaped = corners_3d.reshape(-1,1,3)
    projected, _ = cv2.projectPoints(
        corners_3d_reshaped,
        rvec, tvec,
        camera_matrix, dist_coeffs
    )
    # projected.shape -> (8,1,2)
    return projected.reshape(-1,2)  # (8,2)

def draw_3d_box(frame, pts2d, color=(0,0,0), thickness=2):
    """
    pts2d: (8,2) -> index:
        0: (x0,y0,z0), 1: (x1,y0,z0), 2: (x1,y1,z0), 3: (x0,y1,z0)
        4: (x0,y0,z1), 5: (x1,y0,z1), 6: (x1,y1,z1), 7: (x0,y1,z1)
    """
    pts = pts2d.astype(int)
    # 아래면 (0->1->2->3->0)
    cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, thickness)
    cv2.line(frame, tuple(pts[1]), tuple(pts[2]), color, thickness)
    cv2.line(frame, tuple(pts[2]), tuple(pts[3]), color, thickness)
    cv2.line(frame, tuple(pts[3]), tuple(pts[0]), color, thickness)

    # 윗면 (4->5->6->7->4)
    cv2.line(frame, tuple(pts[4]), tuple(pts[5]), color, thickness)
    cv2.line(frame, tuple(pts[5]), tuple(pts[6]), color, thickness)
    cv2.line(frame, tuple(pts[6]), tuple(pts[7]), color, thickness)
    cv2.line(frame, tuple(pts[7]), tuple(pts[4]), color, thickness)

    # 수직 연결 (0->4, 1->5, 2->6, 3->7)
    cv2.line(frame, tuple(pts[0]), tuple(pts[4]), color, thickness)
    cv2.line(frame, tuple(pts[1]), tuple(pts[5]), color, thickness)
    cv2.line(frame, tuple(pts[2]), tuple(pts[6]), color, thickness)
    cv2.line(frame, tuple(pts[3]), tuple(pts[7]), color, thickness)

    # 삼각형 (6->9->7)
    cv2.line(frame, tuple(pts[6]), tuple(pts[9]), color, thickness)
    cv2.line(frame, tuple(pts[9]), tuple(pts[7]), color, thickness)

    # 삼각형 (2->8->3)
    cv2.line(frame, tuple(pts[2]), tuple(pts[8]), color, thickness)
    cv2.line(frame, tuple(pts[8]), tuple(pts[3]), color, thickness)
    
    # 삼각형 (9->8)
    cv2.line(frame, tuple(pts[9]), tuple(pts[8]), color, thickness)
    

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



# 회전 행렬 (예: 스트라이크 존을 90도 회전시키는 용도)
rotation_matrix = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float32)


ball_zone_corners = np.dot(ball_zone_corners, rotation_matrix.T)
ball_zone_corners2 = np.dot(ball_zone_corners2, rotation_matrix.T)

# ARUCO 사전 및 파라미터 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

frame_count = 0
skip_frames = 1  # 매 프레임마다 처리

out_count = 0
strike_count = 0
ball_count = 0

ball_color = (204, 204, 0)
strike_color = (0, 200, 200)  

# 감지된 점들의 리스트 (마커 좌표계 3D 좌표 기록)
detected_strike_points = []
detected_ball_points = []
ar_started = False  # AR 시작 여부

# 색상 기반 객체 추적 (공) 범위
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
redLower1 = (0, 70, 50)
redUpper1 = (10, 255, 255)
redLower2 = (170, 70, 50)
redUpper2 = (180, 255, 255)
pts = deque(maxlen=64)




###########################################################
# 텍스트 이펙트: STRIKE!
###########################################################
# 이펙트 정보: {'start_time': float, 'duration': float, 'text': str}

# 'STRIKE!' 효과 표시용
effect_end_time = 0.0
effects = []
result = ""

def add_strike_text_effect():
    # 1초 동안 STRIKE! 텍스트 표시
    effects.append({
        'start_time': time.time(),
        'duration': 1.0,
        'text': "STRIKE!"
    })
    
def add_ball_text_effect():
    # 1초 동안 STRIKE! 텍스트 표시
    effects.append({
        'start_time': time.time(),
        'duration': 1.0,
        'text': "BALL!"
    })

def draw_effects(frame, result):
    now = time.time()
    alive = []
    for eff in effects:
        age = now - eff['start_time']
        if age < eff['duration']:
            # 텍스트 그리기
            # 화면 중앙 위치
            h, w = frame.shape[:2]
            cx = w // 2
            cy = h // 2
            # 예: scale=3.0, 빨간색

            if result == "strike":
                cv2.putText(frame, eff['text'], (w-200, 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 200, 200), 3, cv2.LINE_AA)
                alive.append(eff)
            
            else:
                cv2.putText(frame, eff['text'], (w-150, 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                alive.append(eff)
            
    effects[:] = alive



def draw_grid(frame, points, num_divisions):
    """ 사각형 내부에 격자를 그리는 함수 """
    for i in range(1, num_divisions):
        # 수평선
        pt1 = tuple((points[0] + (points[3] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[1] + (points[2] - points[1]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)
        #print("pt1, pt2", pt1, pt2)
        # 수직선
        pt1 = tuple((points[0] + (points[1] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[3] + (points[2] - points[3]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)

def is_point_in_polygon(point, polygon):
    """ 점(2D)이 polygon 내부에 있는지 확인 """
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def reset():
    global detected_strike_points,detected_ball_points, ar_started, strike_count, out_count, ball_count
    ar_started = False
    strike_count = 0
    out_count = 0

def detect_hand_open(results):
    """ 손바닥이 펴진 상태인지 확인 """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손가락 끝 위치
            tips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
            # 손가락이 펴졌는지(손가락 끝 y가 관절 y보다 위인지) 간단 체크
            if all(tip.y < hand_landmarks.landmark[i - 2].y for tip, i in zip(tips, [8, 12, 16, 20])):
                return True
    return False

def detect_index_finger_only(results):
    """ 검지 손가락(인덱스 핑거)만 펴져 있는지 확인하는 함수 """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]   # 검지 손가락 끝
            index_mcp = hand_landmarks.landmark[5]   # 검지 손가락 관절

            # 검지만 올라갔는지 확인 (다른 손가락은 접혀 있어야 함)
            other_fingers_folded = all(
                hand_landmarks.landmark[i].y > hand_landmarks.landmark[i - 2].y
                for i in [12, 16, 20]  # 중지, 약지, 새끼 손가락
            )

            if index_tip.y < index_mcp.y and other_fingers_folded:
                return True  # 검지 손가락만 들려 있음
    return False


def estimate_ball_depth(radius, known_radius=0.036):
    """ 공의 깊이(Z)를 추정하는 함수 """
    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]
    f = (f_x + f_y) / 2
    Z = (f * known_radius) / radius
    return Z

def get_camera_list():
    """ 사용 가능한 카메라 목록 검색 """
    camera_list = []
    for index in range(10):
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            camera_list.append(f"Camera {index}")
            cap.release()
    return camera_list

class CameraPreview:
    def __init__(self, parent, camera_index):
        self.frame = ttk.Frame(parent)
        self.camera_index = camera_index
        self.label = ttk.Label(self.frame, text=f"Camera {camera_index}")
        self.preview = ttk.Label(self.frame)
        self.select_button = ttk.Button(
            self.frame,
            text="Select",
            command=lambda: self.on_select()
        )
        self.label.pack(pady=5)
        self.preview.pack(pady=5)
        self.select_button.pack(pady=5)

        self.stop_event = threading.Event()
        self.start_preview()
    
    def start_preview(self):
        def preview_thread():
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
            if not cap.isOpened():
                self.show_error()
                return
            
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.show_error()
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = imutils.resize(frame, width=320)
                image = PhotoImage(data=cv2.imencode('.png', frame)[1].tobytes())
                self.preview.configure(image=image)
                self.preview.image = image
            cap.release()
        threading.Thread(target=preview_thread, daemon=True).start()
    
    def show_error(self):
        black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        image = PhotoImage(data=cv2.imencode('.png', black_frame_rgb)[1].tobytes())
        self.preview.configure(image=image, text="연결 안됨", foreground="red")
        self.preview.image = image
    
    def stop(self):
        self.stop_event.set()
    
    def on_select(self):
        global selected_camera_index
        selected_camera_index = self.camera_index
        window.quit()

def create_camera_selection_gui():
    """ 카메라 선택 GUI 생성 """
    global window
    window = tk.Tk()
    window.title("Select Camera")
    window.geometry("1280x720")

    cameras = get_camera_list()
    if not cameras:
        print("No cameras found")
        return

    n_cameras = len(cameras)
    cols = min(3, n_cameras)
    rows = (n_cameras + cols - 1) // cols

    previews = []
    for i, camera in enumerate(cameras):
        idx = int(camera.split()[1])
        preview = CameraPreview(window, idx)
        preview.frame.grid(row=i//cols, column=i%cols, padx=10, pady=10)
        previews.append(preview)

    def on_closing():
        for preview in previews:
            preview.stop()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()

    for preview in previews:
        preview.stop()

def project_point_to_3dzone(point_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """
    마커 좌표계의 point_3d를
    Z=0 평면에 강제 투영(스트존 판정용) → 2D 좌표 반환
    """
    projected_3d = np.array([point_3d[0], point_3d[1], 0], dtype=np.float32)
    projected_2d, _ = cv2.projectPoints(
        np.array([projected_3d]), rvec, tvec, camera_matrix, dist_coeffs
    )
    return projected_2d[0][0], projected_3d

def project_real_3d(point_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """
    마커 좌표계의 point_3d(x, y, z)를
    그대로 영상에 투영(실제 높이 반영) → 2D 좌표 반환
    """
    object_points = np.array([point_3d], dtype=np.float32).reshape(-1,1,3)
    projected_2d, _ = cv2.projectPoints(
        object_points,
        rvec, tvec,
        camera_matrix, dist_coeffs
    )
    return projected_2d[0][0]  # (x, y)


if __name__ == "__main__":
    create_camera_selection_gui()
    if selected_camera_index is None:
        print("No camera selected. Exiting.")
        exit()

    cap = cv2.VideoCapture(selected_camera_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"Error: Cannot open camera index {selected_camera_index}")
        exit()

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        fps_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # Mediapipe 손 감지
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 손 펴면 AR 시작
        if not ar_started:
            if detect_hand_open(results):
                ar_started = True
                print("AR Started!")
            else:
                cv2.putText(frame, "Show your hand!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
        
        if ar_started:
            frame_count += 1
            if frame_count % skip_frames != 0:
                cv2.imshow('ARUCO Tracker with Strike Zone', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord('r'):
                    reset()
                continue
            
            if detect_index_finger_only(results):  # ⬅️ 검지만 보이면 시작
                if start_time is None:  # 처음 감지될 때만 시간 기록
                    start_time = time.time()  # ⬅️ 타이머 시작
                    print("Index Finger Detected! Timing begins.")


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(gray)

            if ids is not None:
                # 마커 포즈 추정
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, 0.16, camera_matrix, dist_coeffs
                )

                
                for rvec, tvec in zip(rvecs, tvecs):
                    overlay = frame.copy()

                    # 1) 첫 번째 영역
                    projected_points, _ = cv2.projectPoints(
                        ball_zone_corners, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    projected_points = projected_points.reshape(-1, 2).astype(int)
                    cv2.polylines(overlay, [projected_points], True, (200, 200, 0), 4)
                    cv2.fillPoly(overlay, [projected_points], (200, 200, 0))

                    # 2) 두 번째 영역
                    projected_points2, _ = cv2.projectPoints(
                        ball_zone_corners2, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    projected_points2 = projected_points2.reshape(-1, 2).astype(int)
                    cv2.polylines(overlay, [projected_points2], True, (0, 100, 255), 4)
                    cv2.fillPoly(overlay, [projected_points2], (0, 100, 255))

                    # 3) 반투명 합성 (한 번만)
                    alpha = 0.3  # 투명도 (0=완전투명, 1=불투명)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    ###########################################################################################
                    
                    # 8개 코너
                    box_corners_3d = get_box_corners_3d(box_min, box_max)
                    
                    pts2d = project_box_corners_2d(
                        box_corners_3d,
                        rvec, tvec,
                        camera_matrix, dist_coeffs
                    )
                    #print("pts2d", pts2d)
                    draw_3d_box(frame, pts2d, color=(0,0,0), thickness=4)
                    
                    # 앞면 4개 코너만 사용
                    grid_pts2d = pts2d[[0,1,5,4]]  
                    draw_grid(frame, grid_pts2d, 3)

                    ###########################################################################################

                    
            
                    # 공(녹) 검출
                    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                    mask_green = cv2.inRange(hsv, greenLower, greenUpper)
                    # mask_red1 = cv2.inRange(hsv, redLower1, redUpper1)
                    # mask_red2 = cv2.inRange(hsv, redLower2, redUpper2)
                    # mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                    # mask = cv2.bitwise_or(mask_green, mask_red)
                    mask = cv2.erode(mask_green, None, iterations=2)
                    mask = cv2.dilate(mask, None, iterations=2)

                    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    center = None

                    if len(cnts) > 0:
                        c = max(cnts, key=cv2.contourArea)
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        else:
                            center = (int(x), int(y))

                        if radius > 1:
                            # 공 화면 표시 (노란 원 + 빨간 점)
                            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                            
                            # 공 깊이 추정, 카메라 좌표계
                            estimated_Z = estimate_ball_depth(radius)
                            ball_3d_cam = np.array([[
                                (center[0] - camera_matrix[0,2]) * estimated_Z / camera_matrix[0,0],
                                (center[1] - camera_matrix[1,2]) * estimated_Z / camera_matrix[1,1],
                                estimated_Z
                            ]])

                            # 마커와 공 깊이 비교
                            marker_z = tvec[0][2]
                            depth_threshold = 0.05

                            # 카메라 → 마커 좌표계
                            R_marker, _ = cv2.Rodrigues(rvec)
                            point_in_marker_coord = np.dot(
                                R_marker.T,
                                (ball_3d_cam.reshape(3,1) - tvec.reshape(3,1))
                            ).T[0]

                            px, py, pz = point_in_marker_coord
                            #print(f"Ball 3D (Marker): ({px:.2f}, {py:.2f}, {pz:.2f})")

                            # 실제 3D 높이를 이용해 검정 점 찍기
                            # (직접 그릴 때)
                            pt_2d_real = project_real_3d(
                                point_in_marker_coord,
                                rvec, tvec,
                                camera_matrix, dist_coeffs
                            )
                            # (1) predict
                            prediction = kalman_3d.predict()  # shape=(6,1)
                            px_pred, py_pred, pz_pred, vx_pred, vy_pred, vz_pred = prediction.ravel()

                            # (2) 만약 공이 이번 프레임에 감지됐다면 -> correct
                            # 공 측정값: point_in_marker_coord = (px, py, pz)
                            measurement = np.array([
                                [px],  # x
                                [py],  # y
                                [pz],  # z
                            ], dtype=np.float32)

                            estimated = kalman_3d.correct(measurement)
                            ex, ey, ez, evx, evy, evz = estimated.ravel()

                            detected_strike_points.append({
                                        '3d_coord': (ex, ey, ez),
                                        'rvec': rvec.copy(),
                                        'tvec': tvec.copy()
                                    })
                            pt_2d = project_real_3d((ex, ey, ez), rvec, tvec, camera_matrix, dist_coeffs)
                            pt_2d = np.array(pt_2d).ravel()
                            cv2.circle(frame, (int(pt_2d[0]), int(pt_2d[1])), 8, (0,0,0), 1)
                            
                            
                            ### 스트라이크 판정
                            if (box_min[0] <= px <= box_max[0] and 
                                box_min[1] <= py <= box_max[1] and
                                box_min[2] <= pz <= box_max[2]):
                                
                                
                                
                                #cv2.circle(frame, (int(pt_2d_real[0]), int(pt_2d_real[1])), 8, (0,0,0), -1)

                                # (추가) 기록 후 매 프레임 다시 재투영
                                current_time = time.time()
                                if current_time - last_time > 1.0:
                                    strike_count += 1
                                    last_time = current_time
                                    print(f"Strike Count Increased: {strike_count}")

                                    detected_strike_points.append({
                                        '3d_coord': point_in_marker_coord,
                                        'rvec': rvec.copy(),
                                        'tvec': tvec.copy()
                                    })
                                    add_strike_text_effect()
                                    result = "strike"
                                    
                                    # ⬇️ 속도 측정 종료 및 계산
                                    if start_time is not None:
                                        end_time = time.time()
                                        print("time end")
                                        elapsed_time = end_time - start_time
                                        print(f"Elapsed Time: {elapsed_time:.2f} sec")
                                        
                                        if elapsed_time > 0.1:
                                            speed = (distance_to_plate / elapsed_time) * 3.6  # km/h 변환
                                            print(f"Ball Speed: {speed:.2f} km/h")
                                            cv2.putText(frame, f"{speed:.1f} km/h", (300, 310),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                        else:
                                            print("Elapsed time too short, skipping speed calculation.")
                                        start_time = None  
                            
                            ### 볼  판정
                            if (abs(estimated_Z - marker_z) < depth_threshold and
                                 is_point_in_polygon(center, projected_points)):
                                
                                # (추가) 기록 후 매 프레임 다시 재투영
                                current_time = time.time()
                                if current_time - last_time > 1.0:
                                    ball_count += 1
                                    last_time = current_time
                                    print(f"Ball Count Increased: {strike_count}")

                                    detected_ball_points.append({
                                        '3d_coord': point_in_marker_coord,
                                        'rvec': rvec.copy(),
                                        'tvec': tvec.copy()
                                    })
                                    add_ball_text_effect()
                                    result = "ball"

                                    # ⬇️ 속도 측정 종료 및 계산
                                    if start_time is not None:
                                        end_time = time.time()
                                        print("time end")
                                        elapsed_time = end_time - start_time
                                        print(f"Elapsed Time: {elapsed_time:.2f} sec")
                                        
                                        if elapsed_time > 0.1:
                                            speed = (distance_to_plate / elapsed_time) * 3.6  # km/h 변환
                                            print(f"Ball Speed: {speed:.2f} km/h")
                                            cv2.putText(frame, f"{speed:.1f} km/h", (300, 310),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                        else:
                                            print("Elapsed time too short, skipping speed calculation.")
                                        start_time = None 

                                
                            
                            # (선택) 마커/공 깊이 텍스트 표시
                            marker_depth_text = f"Marker Z: {marker_z:.2f} m"
                            ball_depth_text = f"Ball Z: {estimated_Z:.2f} m"
                            marker_position = tuple(map(int, pts2d[0]))  # 첫 번째 코너

                            cv2.putText(frame, marker_depth_text,
                                        (marker_position[0], marker_position[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                            cv2.putText(frame, ball_depth_text,
                                        (center[0]+20, center[1]+30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            draw_effects(frame, result)
            # 이전에 저장된 점들도 다시 재투영 → 계속 화면에 표시
            last_ball_point = None
            
            for point_data in detected_strike_points:
                

                if ids is not None:
                    # 마커가 인식되었다면
                    pt_2d_real = project_real_3d(
                        point_data['3d_coord'],
                        rvec, tvec,
                        camera_matrix, dist_coeffs
                    )
                    #print("pt_2d_real", pt_2d_real)
                    pt_2d_real = np.array(pt_2d_real).ravel()
                    cv2.circle(frame, (int(pt_2d_real[0]), int(pt_2d_real[1])), 8, (0, 200, 200), -1)


            
            for point_data in detected_ball_points:

                if ids is not None:
                    # 마커가 인식되었다면
                    pt_2d_real = project_real_3d(
                        point_data['3d_coord'],
                        rvec, tvec,
                        camera_matrix, dist_coeffs
                    )                    
                    pt_2d_real = np.array(pt_2d_real).ravel()
                    cv2.circle(frame, (int(pt_2d_real[0]), int(pt_2d_real[1])), 8, (255, 255, 0), -1)
            
            strike_ball_point = [dp['3d_coord'] for dp in detected_strike_points]
            strike_ball_x = [p[0] for p in strike_ball_point]
            strike_ball_y = [p[1] for p in strike_ball_point]
            strike_ball_z = [p[2] for p in strike_ball_point]

            strike_ball_text = [f'Strike {i+1}' for i in range(len(strike_ball_point))]

            strike_ball_trace = go.Scatter3d(
                x=strike_ball_x, y=strike_ball_y, z=strike_ball_z,
                mode='markers+text',
                marker=dict(size=5, color='yellow'),
                text=strike_ball_text,
                name='Strike Points'
            )

            ball_ball_point = [dp['3d_coord'] for dp in detected_ball_points]
            ball_ball_x = [p[0] for p in ball_ball_point]
            ball_ball_y = [p[1] for p in ball_ball_point]
            ball_ball_z = [p[2] for p in ball_ball_point]

            ball_ball_text = [f'Ball {i+1}' for i in range(len(ball_ball_point))]

            ball_ball_trace = go.Scatter3d(
                x=ball_ball_x, y=ball_ball_y, z=ball_ball_z,
                mode='markers+text',
                marker=dict(size=5, color='green'),
                text=ball_ball_text,
                name='Ball Points'
            )
            box_corners_3d = get_box_corners_3d(box_min, box_max)
            plotly_coner_3d = create_box_trace(box_corners_3d, color='blue')
            # (F) Layout 및 시각화
            fig = go.Figure(data= plotly_coner_3d + [strike_ball_trace] + [ball_ball_trace])
            fig.update_layout(
                scene = dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='cube'
                ),
                title="3D Strike Zone & Ball Points"
            )


            # 스트라이크/아웃 표시
            if strike_count >= 3:
                out_count += 1
                strike_count = 0
            
            if ball_count >= 5:
                ball_count = 0

            cv2.putText(frame, f"S {strike_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
            cv2.putText(frame, f"B {ball_count}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"O {out_count}", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS 계산
        fps_end_time = time.time()
        fps = 1.0 / (fps_end_time - fps_start_time + 1e-8)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('ARUCO Tracker with Strike Zone', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('r'):
            reset()
        elif key & 0xFF == ord('s'):
            cv2.imwrite("strike_zone.png", frame)
            print("Strike zone image saved.")
        elif key & 0xFF == ord('b'):
            cv2.imwrite("ball_zone.png", frame)
            print("Ball zone image saved.")
        elif key & 0xFF == ord('c'):
            detected_strike_points = []
            detected_ball_points = []
            print("Data Cleared.")
        elif key & 0xFF == ord('p'):
            # Plotly로 3D 그래프 표시
            fig.show()
            #print("3D Graph Displayed.")
        elif key & 0xFF == ord('t'):
            # 텍스트 이펙트 추가
            add_strike_text_effect()
            add_ball_text_effect()
            print("Text Effect Added.")


    cap.release()
    cv2.destroyAllWindows()