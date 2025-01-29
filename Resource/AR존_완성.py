import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp

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

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

# 공중에 띄울 스트라이크 존의 3D 좌표 (마커 기준으로 상대적인 위치)
strike_zone_corners = np.array([
    [-0.05, 0.1, 0],  # Bottom-left
    [0.05, 0.1, 0],   # Bottom-right
    [0.05, 0.2, 0],   # Top-right
    [-0.05, 0.2, 0]   # Top-left
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
cap = cv2.VideoCapture(0)  # 비브캠 사용
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 프레임 너비 설정
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

def draw_grid(frame, points, num_divisions):
    """ 사각형 내부에 격자를 그리는 함수 """
    for i in range(1, num_divisions):
        # 수평선 그리기
        pt1 = tuple((points[0] + (points[3] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[1] + (points[2] - points[1]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 1)
        
        # 수직선 그리기
        pt1 = tuple((points[0] + (points[1] - points[0]) * i / num_divisions).astype(int))
        pt2 = tuple((points[3] + (points[2] - points[3]) * i / num_divisions).astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 1)

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
        else:
            cv2.putText(frame, "Show your hand!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 2, cv2.LINE_AA)

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
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 마커 좌표 추정
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

            for rvec, tvec in zip(rvecs, tvecs):
                frame = cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                
                # 스트라이크 존 그리기
                projected_points, _ = cv2.projectPoints(strike_zone_corners, rvec, tvec, camera_matrix, dist_coeffs)
                projected_points = projected_points.reshape(-1, 2)
                projected_points = projected_points.astype(int)  # 좌표를 정수형으로 변환

                # 사각형 그리기
                for i in range(4):
                    pt1 = tuple(projected_points[i])
                    pt2 = tuple(projected_points[(i+1) % 4])
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 4)

                # 사각형 내부에 격자 그리기
                draw_grid(frame, projected_points, 3)  # 3등분하여 격자 그리기

            

                # 객체 감지 (예: 간단한 색 기반 감지)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                greenLower = np.array([29, 86, 6])
                greenUpper = np.array([64, 255, 255])
                mask = cv2.inRange(hsv, greenLower, greenUpper)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    # 컨투어의 중심점 계산
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        if is_point_in_polygon((cX, cY), projected_points):
                            detected_points.append((cX, cY))  # 감지된 점 저장
                            
        # 저장된 점 그리기
        for point in detected_points:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)
            strike_count += 1
            # 화면 중앙에 출력
            cv2.putText(frame, "Strike", (100, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)

        
        if strike_count % 3 == 0 and strike_count != 0:
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
