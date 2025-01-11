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
print("Keys in the calibration file:", calib_data.files)

# 올바른 키 이름을 사용하여 데이터를 로드합니다.
camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]

# 공중에 띄울 스트라이크 존의 3D 좌표 (마커 기준으로 상대적인 위치)
strike_zone_corners = np.array([
    [-0.06, 0.08, 0],  # Bottom-left
    [0.06, 0.08, 0],   # Bottom-right
    [0.06, 0.20, 0],   # Top-right
    [-0.06, 0.20, 0]   # Top-left
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
cap = cv2.VideoCapture(2)  # 드로이드캠 사용
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # 프레임 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 프레임 높이 설정

# ARUCO 사전 및 파라미터 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

frame_count = 0
skip_frames = 1  # 매 프레임마다 처리

out_count = 0
strike_count = 0
# 감지된 점들의 리스트
detected_points = []
ar_started = False  # AR 시작 여부

prev_position = None
prev_time = None

# 색상 기반 객체 추적 설정
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
redLower = (170, 210, 150)
redUpper = (173, 220, 175)
pts = deque(maxlen=64)

def draw_grid(frame, points, num_divisions):
    """ 사각형 내부에 격자를 그리는 함수 """
    for i in range(1, num_divisions):
        # 수평선 그리기
        pt1 = tuple((points[0] + (points[3] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[1] + (points[2] - points[1]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)
        
        # 수직선 그리기
        pt1 = tuple((points[0] + (points[1] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[3] + (points[2] - points[3]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 1)

def is_point_in_polygon(point, polygon):
    """ 점이 다각형 내부에 있는지 확인하는 함수 """
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

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
            # 각 손가락 끝의 랜드마크 위치를 확인
            tips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
            if all(tip.y < hand_landmarks.landmark[tip_index].y for tip, tip_index in zip(tips, [6, 10, 14, 18])):  # 손가락이 펴진 상태인지 확인
                return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe를 사용하여 손 감지
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if not ar_started:
        if detect_hand_open(results):
            ar_started = True
            print("AR Started!")
            #cv2.putText(frame, "Start", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Show your hand!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2, cv2.LINE_AA)

    if ar_started:
        frame_count += 1
        if frame_count % skip_frames != 0:
            # 몇 프레임 건너뛰기
            cv2.imshow('ARUCO Tracker with Strike Zone', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('r'):
                reset()
            continue

        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ARUCO 마커 탐지
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        # 마커가 감지되면
        if ids is not None:
            # 마커를 그리기
            #frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 마커 좌표 추정
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

            for rvec, tvec in zip(rvecs, tvecs):
                #frame = aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                
                # 스트라이크 존 그리기
                projected_points, _ = cv2.projectPoints(strike_zone_corners, rvec, tvec, camera_matrix, dist_coeffs)
                projected_points = projected_points.reshape(-1, 2)
                projected_points = projected_points.astype(int)  # 좌표를 정수형으로 변환

                # 사각형 그리기
                for i in range(4):
                    pt1 = tuple(projected_points[i])
                    pt2 = tuple(projected_points[(i+1) % 4])
                    cv2.line(frame, pt1, pt2, (0, 0, 0), 4)

                # 사각형 내부에 격자 그리기
                draw_grid(frame, projected_points, 3)  # 3등분하여 격자 그리기

                # 녹색 및 형광 빨간색 공 추적
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

                        # # 현재 위치와 시간 업데이트
                        # current_position = np.array([x, y])
                        # current_time = time.time()

                        # if prev_position is not None and prev_time is not None:
                        #     time_diff = current_time - prev_time
                        #     position_diff = np.linalg.norm(current_position - prev_position)
                        #     speed_mps = position_diff / time_diff  # 속도 계산 (단위: m/s)
                        #     speed_kph = speed_mps * 3.6  # 속도 변환 (단위: km/h)
                        #     print(f"Speed: {speed_kph:.2f} km/h")

                        # 이전 위치와 시간을 현재 위치와 시간으로 업데이트
                        # prev_position = current_position
                        # prev_time = current_time

                       # 공의 2D 좌표 (화면 좌표계)
                        ball_position_2d = np.array([[x, y]], dtype=np.float32)

                        # 공의 2D 좌표를 3D 좌표로 변환 (카메라 좌표계로 변환)
                        ball_position_3d = cv2.perspectiveTransform(ball_position_2d.reshape(-1, 1, 2), np.linalg.inv(camera_matrix))

                        # 마커의 실제 크기 (미터 단위)
                        marker_size = 0.05  # 5cm 크기의 마커

                        # 마커의 네 모서리 3D 좌표 (카메라 좌표계)
                        marker_corners_3d = np.array([
                            [-marker_size / 2, marker_size / 2, 0],  # 왼쪽 위
                            [marker_size / 2, marker_size / 2, 0],   # 오른쪽 위
                            [marker_size / 2, -marker_size / 2, 0],  # 오른쪽 아래
                            [-marker_size / 2, -marker_size / 2, 0]  # 왼쪽 아래
                        ], dtype=np.float32)


                        # 마커의 3D 위치 및 크기 설정
                        ret, rvec, tvec = cv2.solvePnP(marker_corners_3d, corners[0], camera_matrix, dist_coeffs)

                        # 회전 벡터를 회전 행렬로 변환
                        R, _ = cv2.Rodrigues(rvec)

                        # 마커의 3D 좌표를 월드 좌표계로 변환
                        marker_corners_3d_world = np.dot(R, marker_corners_3d.T).T + tvec.T

                        # 공의 3D 위치가 마커의 3D 경계 내에 있는지 확인
                        if (np.min(marker_corners_3d_world[:, 0]) <= ball_position_3d[0, 0, 0] <= np.max(marker_corners_3d_world[:, 0]) and
                            np.min(marker_corners_3d_world[:, 1]) <= ball_position_3d[0, 0, 1] <= np.max(marker_corners_3d_world[:, 1])):
                            detected_points.append(center)
                            strike_count += 1
                            cv2.putText(frame, "Strike", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)



        for point in detected_points:
            cv2.circle(frame, point, 8, (0, 0, 0), 2)
            #time.sleep(1)
        
        if strike_count % 3 == 0 and strike_count != 0:
            out_count += 1
            strike_count = 0  # 스트라이크 카운트 초기화

        cv2.putText(frame, f"S {strike_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"O {out_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 결과 출력
    cv2.imshow('ARUCO Tracker with Strike Zone', frame)

    # 키 입력 처리
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        reset()

cap.release()
cv2.destroyAllWindows()
