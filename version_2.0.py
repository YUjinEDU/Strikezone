import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
from collections import deque
import imutils
import time

last_strike_time = 0.0


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

# 공의 실제 반지름 (미터 단위, 예: 0.036 미터는 약 7cm)
ball_radius_real = 0.036

# 공중에 띄울 스트라이크 존의 3D 좌표 (마커 기준으로 상대적인 위치)
strike_zone_corners = np.array([
    [-0.08, 0.05, 0],  # Bottom-left
    [0.08, 0.05, 0],   # Bottom-right
    [0.08, 0.15, 0],   # Top-right
    [-0.08, 0.15, 0]   # Top-left
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
cap = cv2.VideoCapture(1)  # 비브캠 사용
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 604)  # 프레임 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

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

# 색상 기반 객체 추적 설정
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)  # HSV 범위를 확장
redLower1 = (0, 70, 50)
redUpper1 = (10, 255, 255)
redLower2 = (170, 70, 50)
redUpper2 = (180, 255, 255)
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
            # 손가락 끝의 랜드마크 위치 확인
            tips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
            if all(tip.y < hand_landmarks.landmark[i - 2].y for tip, i in zip(tips, [8, 12, 16, 20])):  # 손가락이 펴진 상태 확인
                return True
    return False

def estimate_ball_depth(radius, known_radius=0.036):
    """
    공의 깊이 (Z축)를 추정하는 함수
    radius: 공의 이미지 반지름 (픽셀 단위)
    known_radius: 공의 실제 반지름 (미터 단위)
    """
    # 공의 실제 크기와 이미지 크기를 이용한 깊이 계산
    # Z = (f * R) / r
    f_x = camera_matrix[0, 0]
    f_y = camera_matrix[1, 1]
    f = (f_x + f_y) / 2
    Z = (f * known_radius) / radius
    return Z

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # Mediapipe를 사용하여 손 감지
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    # ar_started = True  # 항상 AR 시작으로 설정됨을 수정
    if not ar_started:
        if detect_hand_open(results):
            ar_started = True
            print("AR Started!")
        else:
            cv2.putText(frame, "Show your hand!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

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
            # 마커 좌표 추정
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.15, camera_matrix, dist_coeffs)

            for idx, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                # 스트라이크 존 그리기
                projected_points, _ = cv2.projectPoints(strike_zone_corners, rvec, tvec, camera_matrix, dist_coeffs)
                projected_points = projected_points.reshape(-1, 2)
                projected_points = projected_points.astype(int)  # 좌표를 정수형으로 변환

                # 사각형 그리기
                cv2.polylines(frame, [projected_points], isClosed=True, color=(0, 0, 0), thickness=4)

                # 사각형 내부에 격자 그리기
                draw_grid(frame, projected_points, 3)  # 3등분하여 격자 그리기

                # 녹색 및 형광 빨간색 공 추적
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

                    if radius > 3:
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        pts.appendleft(center)

                        # 공의 깊이 추정
                        estimated_Z = estimate_ball_depth(radius)
                        print(f"Estimated Depth (Z): {estimated_Z:.2f} meters")

                        # 스트라이크 존 깊이와 비교하여 스트라이크 여부 판단
                        marker_z = tvec[0][2]  # 단위: 미터
                        depth_threshold = 0.05  # 깊이 차이 허용 범위 (미터)

                        if abs(estimated_Z - marker_z) < depth_threshold:
                            # 공의 깊이가 스트라이크 존과 유사
                            cv2.putText(frame, "Depth Strike!", (center[0] + 20, center[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                             # 스트라이크 존 내부에 있는지 확인
                            if is_point_in_polygon(center, projected_points):
                                detected_points.append(center)
                                cv2.putText(frame, "Strike Zone Hit!", (center[0] + 10, center[1] + 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

                                current_time = time.time()
                                #strike_count += 1

                                if current_time - last_strike_time > 1.0:  # 1초 경과 조건
                                    strike_count += 1  # 스트라이크 증가
                                    last_strike_time = current_time  # 마지막 증가 시간 업데이트
                                    print(f"Strike Count Increased: {strike_count}")

                        else:
                            cv2.putText(frame, "Not Strike Depth", (center[0] + 20, center[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        # 마커의 깊이와 공의 깊이를 화면에 출력
                        marker_depth_text = f"Marker Z: {marker_z:.2f} m"
                        ball_depth_text = f"Ball Z: {estimated_Z:.2f} m"
                        # 마커의 위치 근처에 마커 깊이 표시
                        marker_position = tuple(projected_points[0])  # 스트라이크 존의 첫 번째 점 근처
                        cv2.putText(frame, marker_depth_text, (marker_position[0], marker_position[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                        # 공의 위치 근처에 공 깊이 표시
                        cv2.putText(frame, ball_depth_text, (center[0] + 20, center[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                       
        for point in detected_points:
            cv2.circle(frame, point, 8, (0, 0, 0), -1)
        
        if strike_count >= 3:
            out_count += 1
            strike_count = 0  # 스트라이크 카운트 초기화

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
