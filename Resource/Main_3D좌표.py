import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
from collections import deque
import imutils
import time

# Mediapipe 손 추적 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 카메라 캘리브레이션 데이터 로드
calib_data = np.load("camera_calib.npz")
camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]

# 스트라이크 존의 3D 좌표 (마커 기준으로 상대적인 위치)
strike_zone_corners = np.array([
    [-0.08, 0.1, 0],  # Bottom-left
    [0.08, 0.1, 0],   # Bottom-right
    [0.08, 0.3, 0],   # Top-right
    [-0.08, 0.3, 0]   # Top-left
], dtype=np.float32)

# 회전 행렬 생성 (90도 회전)
rotation_matrix = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float32)

# 스트라이크 존의 각 점에 회전 행렬 적용
strike_zone_corners = np.dot(strike_zone_corners, rotation_matrix.T)

# 카메라 설정
cap = cv2.VideoCapture(2)  # 비브캠 사용
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

# ARUCO 사전 및 파라미터 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

frame_count = 0
skip_frames = 1  # 매 프레임마다 처리

out_count = 0
strike_count = 0
detected_points = []
ar_started = False  # AR 시작 여부

# 색상 기반 객체 추적 설정
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
redLower = (170, 210, 150)
redUpper = (173, 220, 175)
pts = deque(maxlen=64)

# 공의 실제 지름 (미터 단위로 측정)
actual_ball_diameter = 0.06  # 예: 공의 실제 지름이 7.4 cm

def draw_grid(frame, points, num_divisions):
    """ 사각형 내부에 격자를 그리는 함수 """
    for i in range(1, num_divisions):
        pt1 = tuple((points[0] + (points[3] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[1] + (points[2] - points[1]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)
        
        pt1 = tuple((points[0] + (points[1] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[3] + (points[2] - points[3]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)

def reset():
    global detected_points, ar_started, strike_count, out_count
    detected_points = []
    ar_started = False
    strike_count = 0
    out_count = 0

def detect_hand_open(results):
    """ 손바닥이 펴진 상태를 감지하는 함수 """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
            if all(tip.y < hand_landmarks.landmark[tip_index].y for tip, tip_index in zip(tips, [6, 10, 14, 18])):  
                return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if not ar_started:
        if detect_hand_open(results):
            ar_started = True
            print("AR Started!")
        else:
            cv2.putText(frame, "Show your hand!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2, cv2.LINE_AA)

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

            for rvec, tvec in zip(rvecs, tvecs):
                projected_points, _ = cv2.projectPoints(strike_zone_corners, rvec, tvec, camera_matrix, dist_coeffs)
                projected_points = projected_points.reshape(-1, 2).astype(int)

                for i in range(4):
                    pt1 = tuple(projected_points[i])
                    pt2 = tuple(projected_points[(i+1) % 4])
                    cv2.line(frame, pt1, pt2, (0, 0, 0), 4)

                draw_grid(frame, projected_points, 3)

                blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                mask_green = cv2.inRange(hsv, greenLower, greenUpper)
                mask_red = cv2.inRange(hsv, redLower, redUpper)
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
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    if radius > 3:
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        pts.appendleft(center)

                        ball_position_2d = np.array([[x, y]], dtype=np.float32)
                        apparent_ball_diameter = 2 * radius
                        focal_length = camera_matrix[0, 0]
                        distance_to_ball = (actual_ball_diameter * focal_length) / apparent_ball_diameter

                        ball_position_3d_camera = np.array([x, y, distance_to_ball])
                        print(f"Ball 3D position (camera coordinates): {ball_position_3d_camera}")

                        ret, rvec, tvec = cv2.solvePnP(strike_zone_corners, corners[0], camera_matrix, dist_coeffs)
                        R, _ = cv2.Rodrigues(rvec)
                        strike_zone_corners_world = np.dot(R, strike_zone_corners.T).T + tvec.T
                        print(f"Strike zone 3D world coordinates: {strike_zone_corners_world}")

                        ball_position_3d_world = np.dot(R, ball_position_3d_camera.T).T + tvec.T
                        print(f"Ball 3D position (world coordinates): {ball_position_3d_world}")

                        if (np.min(strike_zone_corners_world[:, 0]) <= ball_position_3d_world[0, 0] <= np.max(strike_zone_corners_world[:, 0]) and 
                            np.min(strike_zone_corners_world[:, 1]) <= ball_position_3d_world[0, 1] <= np.max(strike_zone_corners_world[:, 1]) and
                            np.min(strike_zone_corners_world[:, 2]) <= ball_position_3d_world[0, 2] <= np.max(strike_zone_corners_world[:, 2])):
                            detected_points.append(center)
                            strike_count += 1
                            cv2.putText(frame, "Strike", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)
                            time.sleep(1)

        for point in detected_points:
            cv2.circle(frame, point, 8, (0, 0, 0), -1)

        if strike_count % 3 == 0 and strike_count != 0:
            out_count += 1
            strike_count = 0

        cv2.putText(frame, f"S {strike_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"O {out_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('ARUCO Tracker with Strike Zone', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        reset()

cap.release()
cv2.destroyAllWindows()
