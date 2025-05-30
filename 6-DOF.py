import cv2
import cv2.aruco as aruco
import numpy as np
import os

# --- 중요 설정 값들 (사용자 환경에 맞게 반드시 수정해야 합니다!) ---

# 1. 카메라 보정 값 (반드시 실제 카메라로 보정한 값을 사용하세요!)
# 이 값들은 예시이며, 정확한 포즈 추정을 위해서는 실제 보정된 값이 필요합니다.
# 카메라 매트릭스 (예시 값)
data = np.load('camera_calib_2.npz')
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# 2. ArUco 마커의 실제 한 변의 길이 (미터 단위)
# 예: 마커 한 변의 길이가 5cm라면 0.05로 설정
marker_length_meters = 0.185  # <<<--- 여기에 실제 마커 크기(미터)를 입력하세요!

# 3. 사용할 ArUco 딕셔너리 선택 (6x6 마커용)
# DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000 등이 있습니다.
# 사용하려는 마커에 맞는 딕셔너리를 선택하세요.
aruco_dict_type = aruco.DICT_5X5_250
# --- 설정 값 끝 ---

def main():
    # ArUco 딕셔너리 불러오기
    try:
        dictionary = aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters) # OpenCV 4.7.0 이상 권장
    except AttributeError:
        # OpenCV 구버전 (ArucoDetector가 없는 경우) fallback 또는 에러 처리
        print("오류: ArucoDetector를 찾을 수 없습니다. OpenCV 버전이 4.7.0 이상인지 확인하거나, 구버전 방식의 마커 검출 코드를 사용해야 합니다.")
        print("OpenCV 4.6.0 이하에서는 다음과 같이 사용합니다:")
        print("dictionary = aruco.Dictionary_get(aruco_dict_type)")
        print("parameters = aruco.DetectorParameters_create()")
        print("# corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)")
        return

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다.")
        return

    print("카메라 피드를 시작합니다. ESC 키를 누르면 종료됩니다.")
    print(f"사용 중인 ArUco 딕셔너리: {aruco_dict_type}")
    print(f"설정된 마커 한 변의 길이: {marker_length_meters * 100} cm")
    print("주의: 정확한 포즈 추정을 위해서는 카메라 매트릭스와 왜곡 계수를 실제 값으로 반드시 수정해야 합니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 그레이스케일로 변환 (마커 검출에 사용)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ArUco 마커 검출
        # OpenCV 4.7.0 이상: ArucoDetector 사용
        corners, ids, rejected_img_points = detector.detectMarkers(gray)
        
        # OpenCV 4.6.0 이하: detectMarkers 직접 사용
        # corners, ids, rejected_img_points = aruco.detectMarkers(gray, dictionary, parameters=parameters)


        if ids is not None and len(ids) > 0:
            # 검출된 각 마커에 대해 포즈 추정
            # rvecs: 회전 벡터 (각 마커에 대한 것)
            # tvecs: 이동 벡터 (각 마커에 대한 것)
            # estimatePoseSingleMarkers 함수는 각 마커의 코너와 실제 크기, 카메라 정보를 바탕으로 포즈를 추정합니다.
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_length_meters, camera_matrix, dist_coeffs
            )

            # 검출된 마커 및 포즈 시각화
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                rvec = rvecs[i]
                tvec = tvecs[i]
                
                # 마커 좌표계의 축 그리기 (OpenCV 4.7.0 이상 권장)
                # drawFrameAxes의 세 번째 인자는 축의 길이(미터 단위)입니다. 마커 길이의 절반으로 설정합니다.
                try:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length_meters / 2)
                except AttributeError:
                    # 구버전 OpenCV에서는 cv2.aruco.drawAxis 사용
                    aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length_meters / 2)


                # 마커 ID 및 거리 정보 텍스트로 표시 (선택 사항)
                marker_id = ids[i][0]
                distance = np.linalg.norm(tvec) # 카메라 원점으로부터 마커 원점까지의 거리
                cv2.putText(
                    frame,
                    f"ID: {marker_id} Dist: {distance:.2f}m",
                    (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 15), # 첫 번째 코너 위에 표시
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # 결과 프레임 보여주기
        cv2.imshow("ArUco 6-DoF Pose Estimation", frame)

        # 'ESC' 키를 누르면 루프 종료
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램을 종료합니다.")

if __name__ == "__main__":
   
    main()