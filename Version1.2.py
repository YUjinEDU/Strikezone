#####################################
# Version 1.2
# Date: 2025.04.28
# Author: 박유진
# Description: 3D 증강현실 코드 (2D 판정 제거)
#  - 손 인식으로 AR 시작
#  - 공의 깊이를 계산하여 마커 기준 3D 스트라이크 존 판정
########################################

import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
from collections import deque
import imutils

# Mediapipe 손 추적 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 카메라 캘리브레이션 데이터 로드
calib_data = np.load("camera_calib.npz")
print("Keys in the calibration file:", calib_data.files)

# 올바른 키 이름을 사용하여 데이터를 로드합니다.
camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]

# 공의 실제 지름 (단위: 미터) - 예: 야구공 지름 약 7.3cm -> 0.073
known_width = 0.073

# 공중에 띄울 스트라이크 존의 3D 좌표 (마커 기준으로 상대적 위치)
strike_zone_corners = np.array([
    [-0.08, 0.1, 0],   # Bottom-left
    [ 0.08, 0.1, 0],   # Bottom-right
    [ 0.08, 0.3, 0],   # Top-right
    [-0.08, 0.3, 0]    # Top-left
], dtype=np.float32)

# 스트라이크 존 높이 (Z축 방향으로의 높이)
strike_zone_height = 0.2  # 20cm 정도 예시

# 회전 행렬 생성 (90도 회전)
rotation_matrix = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float32)

# 스트라이크 존 각 점에 회전 행렬 적용
strike_zone_corners = np.dot(strike_zone_corners, rotation_matrix.T)

# 카메라 설정
cap = cv2.VideoCapture(1)  # 카메라 인덱스(환경에 맞게 수정)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ARUCO 사전 및 파라미터 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

frame_count = 0
skip_frames = 1  # 성능 조정(매 프레임 처리)

out_count = 0
strike_count = 0
ar_started = False  # AR 시작 여부

# 색상 기반 객체 추적(공 검출) 설정
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
redLower   = (170, 210, 150)
redUpper   = (173, 220, 175)
pts = deque(maxlen=64)

def detect_hand_open(results):
    """손바닥이 펴진 상태를 감지하는 함수"""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 각 손가락 끝의 랜드마크 인덱스
            finger_tips_idx = [8, 12, 16, 20]  
            # 해당 손가락 관절 인덱스
            finger_joints_idx = [6, 10, 14, 18]
            
            tips = [hand_landmarks.landmark[i] for i in finger_tips_idx]
            # 손가락이 펴졌는지 판단(y 값 비교, Mediapipe는 y가 위->아래 증가)
            if all(tip.y < hand_landmarks.landmark[joint_idx].y 
                   for tip, joint_idx in zip(tips, finger_joints_idx)):
                return True
    return False

def define_strike_zone_3d(corners, height):
    """스트라이크 존을 3D 볼륨(8개 꼭지점)으로 정의"""
    base = corners  # 바닥 사각형
    top = corners + np.array([0, 0, height])  # 높이 추가
    strike_zone_3d = np.vstack((base, top))
    return strike_zone_3d

strike_zone_3d = define_strike_zone_3d(strike_zone_corners, strike_zone_height)

def 스트라이크_존에_포함(p_marker):
    """3D 좌표 p_marker가 스트라이크 존 볼륨 내에 있는지 확인"""
    x, y, z = p_marker[:3]
    
    # 스트라이크 존의 가로/세로 절반 크기
    half_length = 0.08  # 16cm 가정
    half_width  = 0.10  # 20cm 가정
    height      = strike_zone_height  # 20cm 가정
    
    # 마커 좌표계에서 x, y, z 범위를 확인
    # (원하는 좌표 범위에 맞게 조정 가능)
    return (
        -half_length < x < half_length and
        -half_width  < y < half_width  and
         0          < z < height
    )

def reset():
    global ar_started, strike_count, out_count
    ar_started = False
    strike_count = 0
    out_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe로 손 감지
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if not ar_started:
        # 손을 펴면 AR 시작
        if detect_hand_open(results):
            ar_started = True
            print("AR Started!")
        else:
            cv2.putText(frame, "Show your hand!", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, 
                        (0, 255, 255), 2, cv2.LINE_AA)

    if ar_started:
        frame_count += 1
        if frame_count % skip_frames != 0:
            # 프레임 건너뛰기
            cv2.imshow('ARUCO Tracker (3D only)', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('r'):
                reset()
            continue

        # GRAY 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ARUCO 마커 탐지
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # 마커 자세(rvec, tvec) 추정
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, 0.05, camera_matrix, dist_coeffs
            )

            for rvec, tvec in zip(rvecs, tvecs):
                # 스트라이크 존을 2D로 표시(시각화 용도)
                projected_points, _ = cv2.projectPoints(
                    strike_zone_corners, rvec, tvec, 
                    camera_matrix, dist_coeffs
                )
                projected_points = projected_points.reshape(-1, 2)
                projected_points = projected_points.astype(int)

                # 사각형 그리기
                for i in range(4):
                    pt1 = tuple(projected_points[i])
                    pt2 = tuple(projected_points[(i+1) % 4])
                    cv2.line(frame, pt1, pt2, (0, 0, 0), 3)

                # (옵션) 축 그리기
                # cv2.drawAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

                # 회전/변환 행렬 계산
                R, _ = cv2.Rodrigues(rvec)
                T_marker_to_camera = np.hstack((R, tvec.reshape(3, 1)))
                T_marker_to_camera = np.vstack((T_marker_to_camera, [0, 0, 0, 1]))
                T_camera_to_marker = np.linalg.inv(T_marker_to_camera)

                # 공(녹색 or 빨간색) 추적
                blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                mask_green = cv2.inRange(hsv, greenLower, greenUpper)
                mask_red   = cv2.inRange(hsv, redLower,   redUpper)
                mask = cv2.bitwise_or(mask_green, mask_red)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                if len(cnts) > 0:
                    c = max(cnts, key=cv2.contourArea)
                    ((cx, cy), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        center = (int(M["m10"] / M["m00"]), 
                                  int(M["m01"] / M["m00"]))

                        if radius > 3:
                            # 공 표시
                            cv2.circle(frame, (int(cx), int(cy)), 
                                       int(radius), (0, 255, 255), 2)
                            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                            pts.appendleft(center)

                            # 공 깊이(Z) 추정
                            f_x = camera_matrix[0, 0]
                            depth = (known_width * f_x) / (2 * radius)

                            # 2D 픽셀 -> 3D 카메라 좌표
                            c_x = camera_matrix[0, 2]
                            c_y = camera_matrix[1, 2]
                            X = (center[0] - c_x) * depth / f_x
                            Y = (center[1] - c_y) * depth / camera_matrix[1, 1]
                            Z = depth
                            p_camera = np.array([X, Y, Z, 1])

                            # 마커 좌표계로 변환
                            p_marker = T_camera_to_marker @ p_camera
                            p_marker = p_marker / p_marker[3]

                            # 스트라이크 존 판정 (3D만)
                            if 스트라이크_존에_포함(p_marker[:3]):
                                strike_count += 1
                                cv2.putText(frame, "3D Strike", 
                                            (center[0] + 10, center[1]), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.7, (0, 255, 0), 2)
                            else:
                                out_count += 1
                                cv2.putText(frame, "3D Out", 
                                            (center[0] + 10, center[1]), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.7, (0, 0, 255), 2)

        # 스트라이크가 3번 쌓일 때마다 아웃 처리
        if strike_count % 3 == 0 and strike_count != 0:
            out_count += 1
            strike_count = 0

        # 스트라이크, 아웃 카운트 표시
        cv2.putText(frame, f"S {strike_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"O {out_count}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # 결과 화면
    cv2.imshow('ARUCO Tracker (3D only)', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset()

cap.release()
cv2.destroyAllWindows()
