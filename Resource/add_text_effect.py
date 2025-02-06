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

global frame_count
last_strike_time = 0.0
selected_camera_index = None  # 선택된 카메라 인덱스를 저장하는 변수
preview_stop_event = threading.Event() # 미리보기 중지 이벤트

# Mediapipe 손 추적 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 카메라 캘리브레이션 데이터 로드
calib_data = np.load("camera_calib.npz")
print("Keys in the calibration file:", calib_data.files)

camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]

# 공의 실제 반지름 (미터 단위)
ball_radius_real = 0.036

# 공중에 띄울 스트라이크 존의 3D 좌표 (마커 기준)
strike_zone_corners = np.array([
    [-0.08, 0.1, 0],
    [ 0.08, 0.1, 0],
    [ 0.08, 0.3, 0],
    [-0.08, 0.3, 0]
], dtype=np.float32)

# 회전 행렬 (예: 스트라이크 존을 90도 회전)
rotation_matrix = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float32)
strike_zone_corners = np.dot(strike_zone_corners, rotation_matrix.T)

# ARUCO 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

frame_count = 0
skip_frames = 1  # 매 프레임마다 처리

out_count = 0
strike_count = 0

# 감지된 점들의 리스트 (마커 좌표계 3D 좌표)
detected_points = []
ar_started = False

# 색상 기반 객체 추적 (공) 범위
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
redLower1 = (0, 70, 50)
redUpper1 = (10, 255, 255)
redLower2 = (170, 70, 50)
redUpper2 = (180, 255, 255)
pts = deque(maxlen=64)

# 'STRIKE!' 효과 표시용
strike_effect_end_time = 0.0

###########################################################
# (A) 텍스트 이펙트: STRIKE!
###########################################################
# 이펙트 정보: {'start_time': float, 'duration': float, 'text': str}
effects = []

def draw_grid(frame, points, num_divisions):
    for i in range(1, num_divisions):
        pt1 = tuple((points[0] + (points[3] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[1] + (points[2] - points[1]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)

        pt1 = tuple((points[0] + (points[1] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[3] + (points[2] - points[3]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def reset():
    global detected_points, ar_started, strike_count, out_count, effects
    detected_points = []
    ar_started = False
    strike_count = 0
    out_count = 0
    effects.clear()

def detect_hand_open(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
            if all(tip.y < hand_landmarks.landmark[i - 2].y for tip, i in zip(tips, [8, 12, 16, 20])):
                return True
    return False

def estimate_ball_depth(radius, known_radius=0.036):
    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]
    f = (f_x + f_y) / 2
    Z = (f * known_radius) / radius
    return Z

def get_camera_list():
    camera_list = []
    for index in range(10):
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            camera_list.append(f"Camera {index}")
            cap.release()
    return camera_list

###########################################################
# (B) STRIKE! 텍스트 이펙트
###########################################################
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
        'text': "ball!"
    })

def draw_effects(frame):
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
            cv2.putText(frame, eff['text'], (cx-100, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
            alive.append(eff)
    effects[:] = alive

########################################
# project_point_to_3dzone() & project_3d_point()
########################################
def project_point_to_3dzone(point_3d, rvec, tvec, camera_matrix, dist_coeffs):
    # Z=0 평면 강제
    projected_3d = np.array([point_3d[0], point_3d[1], 0], dtype=np.float32)
    projected_2d, _ = cv2.projectPoints(
        np.array([projected_3d]), rvec, tvec, camera_matrix, dist_coeffs
    )
    return projected_2d[0][0], projected_3d

def project_3d_point(point_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """ 실제 (x,y,z) 그대로 2D로 투영 """
    object_points = np.array([point_3d], dtype=np.float32).reshape(-1,1,3)
    projected_2d, _ = cv2.projectPoints(
        object_points,
        rvec, tvec,
        camera_matrix, dist_coeffs
    )
    return projected_2d[0][0]

########################################
# 카메라 선택 GUI (생략 가능)
########################################
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

##############################
# 메인 실행부
##############################
if __name__ == "__main__":
    create_camera_selection_gui()
    if selected_camera_index is None:
        print("No camera selected. Exiting.")
        exit()

    cap = cv2.VideoCapture(selected_camera_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"Error: Cannot open camera index {selected_camera_index}")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # Mediapipe로 손 감지
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
                # 이펙트 그리기
                draw_effects(frame)
                cv2.imshow('ARUCO Tracker with Strike Zone', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord('r'):
                    reset()
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(gray)

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, 0.15, camera_matrix, dist_coeffs
                )

                for rvec, tvec in zip(rvecs, tvecs):
                    # 스트라이크 존
                    projected_points, _ = cv2.projectPoints(
                        strike_zone_corners, rvec, tvec, camera_matrix, dist_coeffs
                    )
                    projected_points = projected_points.reshape(-1, 2).astype(int)

                    cv2.polylines(frame, [projected_points], True, (0, 0, 0), 4)
                    draw_grid(frame, projected_points, 3)

                    # 공(녹/빨) 찾기
                    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                    mask_green = cv2.inRange(hsv, greenLower, greenUpper)
                    mask_red1 = cv2.inRange(hsv, redLower1, redUpper1)
                    mask_red2 = cv2.inRange(hsv, redLower2, redUpper2)
                    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
                    mask = cv2.bitwise_or(mask_green, mask_red)
                    mask = cv2.erode(mask, None, iterations=2)
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
                            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                            cv2.circle(frame, center, 5, (0, 0, 255), -1)

                            # 깊이
                            estimated_Z = estimate_ball_depth(radius)
                            ball_3d_cam = np.array([[
                                (center[0] - camera_matrix[0,2]) * estimated_Z / camera_matrix[0,0],
                                (center[1] - camera_matrix[1,2]) * estimated_Z / camera_matrix[1,1],
                                estimated_Z
                            ]])

                            marker_z = tvec[0][2]
                            depth_threshold = 0.05

                            # 스트라이크 판정
                            if (abs(estimated_Z - marker_z) < depth_threshold and
                                is_point_in_polygon(center, projected_points)):

                                R_marker, _ = cv2.Rodrigues(rvec)
                                point_in_marker_coord = np.dot(
                                    R_marker.T,
                                    (ball_3d_cam.reshape(3,1) - tvec.reshape(3,1))
                                ).T[0]

                                # 스트라이크!
                                current_time = time.time()
                                if current_time - last_strike_time > 1.0:
                                    strike_count += 1
                                    last_strike_time = current_time
                                    print(f"Strike Count: {strike_count}")

                                    # (C) STRIKE! 텍스트 이펙트
                                    add_strike_text_effect()
                                
                                 # (선택) 마커/공 깊이 텍스트 표시
                                marker_depth_text = f"Marker Z: {marker_z:.2f} m"
                                ball_depth_text = f"Ball Z: {estimated_Z:.2f} m"
                                marker_position = tuple(projected_points[0])  # 첫 번째 코너
                                cv2.putText(frame, marker_depth_text,
                                            (marker_position[0], marker_position[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                cv2.putText(frame, ball_depth_text,
                                            (center[0]+20, center[1]+30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 이펙트 그리기
            draw_effects(frame)

            if strike_count >= 3:
                out_count += 1
                strike_count = 0

            cv2.putText(frame, f"S {strike_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
            cv2.putText(frame, f"O {out_count}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time + 1e-8)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ARUCO Tracker with Strike Zone', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('r'):
            reset()

    cap.release()
    cv2.destroyAllWindows()
