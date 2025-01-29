import cv2
import cv2.aruco as aruco
import numpy as np

# 카메라 캘리브레이션 데이터 로드
calib_data = np.load("camera_calib.npz")
print("Keys in the calibration file:", calib_data.files)

# 올바른 키 이름을 사용하여 데이터를 로드합니다.
camera_matrix = calib_data["camera_matrix"]
dist_coeffs = calib_data["dist_coeffs"]

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정

# ARUCO 사전 및 파라미터 설정
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

frame_count = 0
skip_frames = 2  # 매 5프레임마다 한 번씩 처리

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        # 몇 프레임 건너뛰기
        cv2.imshow('ARUCO Tracker with Strike Zone', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
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
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            # 사각형 내부에 격자 그리기
            draw_grid(frame, projected_points, 3)  # 4등분하여 격자 그리기

    # 결과 출력
    cv2.imshow('ARUCO Tracker with Strike Zone', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
