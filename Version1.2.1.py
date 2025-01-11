import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
from collections import deque
import imutils

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

# ARUCO 사전 및 파라미터 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

frame_count = 0
skip_frames = 1  # 매 프레임마다 처리

# 비디오 캡처 초기화
cap = cv2.VideoCapture(1)

out_count = 0
strike_count = 0

ar_started = False # AR 시작 여부

# 색상 기반 객체 추적 설정
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
redLower = (170, 210, 150)
redUpper = (173, 220, 175)
pts = deque(maxlen=64)

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

        if ids is not None:
            # 마커 좌표 추정
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

             # 각 마커에 대해 반복 처리
            for i in range(len(ids)):
                rvec = rvecs[i]  # 회전 벡터
                tvec = tvecs[i]  # 평행이동 벡터
                
                # 회전 벡터를 회전 행렬로 변환
                R, _ = cv2.Rodrigues(rvec)
                
                # tvec를 (3,1) 형태로 변환
                tvec_reshaped = tvec.reshape(3, 1)
                
                # -R.T @ tvec 계산
                translation = -R.T @ tvec_reshaped
                
                # 카메라에서 마커로의 변환 행렬 생성
                T_camera_to_marker = np.hstack((R.T, translation))  # 수평 결합
                T_camera_to_marker = np.vstack((T_camera_to_marker, [0, 0, 0, 1])) 
                
                # Display marker's Z-axis depth
                marker_z = tvec_reshaped[2][0]  # Z-axis value
                cv2.putText(frame, f"Marker Z: {marker_z:.2f} m", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


            # Ball detection and 3D position estimation
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
                if radius > 10:
                    # Estimate depth Z
                    known_width = 0.06  # Actual diameter of the ball in meters
                    f_x = camera_matrix[0, 0]
                    Z = (known_width * f_x) / (2 * radius)

                    # Convert 2D to 3D camera coordinates
                    c_x = camera_matrix[0, 2]
                    c_y = camera_matrix[1, 2]
                    X = (x - c_x) * Z / f_x
                    Y = (y - c_y) * Z / camera_matrix[1, 1]

                    # Display ball's Z-axis depth
                    cv2.putText(frame, f"Ball Z: {Z:.2f} m", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                    # Homogeneous coordinates
                    p_camera = np.array([X, Y, Z, 1])

                    # Transform to marker coordinates
                    p_marker = T_camera_to_marker @ p_camera
                    p_marker = p_marker / p_marker[3]

                    # Define strike zone boundaries in marker coordinates
                    strike_zone_x_min = -0.05  # Example values
                    strike_zone_x_max = 0.05
                    strike_zone_y_min = 0.1
                    strike_zone_y_max = 0.3
                    strike_zone_z_min = -0.1
                    strike_zone_z_max = 0.1

                    # Check if ball is within strike zone
                    if (strike_zone_x_min < p_marker[0] < strike_zone_x_max and
                        strike_zone_y_min < p_marker[1] < strike_zone_y_max and
                        strike_zone_z_min < p_marker[2] < strike_zone_z_max):
                        strike_count += 1
                        cv2.putText(frame, "Strike", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)
                    else:
                        out_count += 1
                        cv2.putText(frame, "Out", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"S {strike_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)
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