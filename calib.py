import cv2
import numpy as np
import glob
import time
import random

# 체스보드 크기 설정 (내부 코너 수)
chessboard_size = (11, 8)
frame_size = (640, 480)  # 리사이징 후의 해상도

# 체스보드 3D 포인트 준비
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 저장할 배열 준비
objpoints = [] # 3D 포인트
imgpoints = [] # 2D 포인트

# 캘리브레이션 이미지 로드
images = glob.glob("./calibration_images/*.jpg")

# 이미지 샘플링
num_samples = 30  # 사용할 샘플 수
if len(images) > num_samples:
    images = random.sample(images, num_samples)

if len(images) == 0:
    print("이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.")
else:
    start_time = time.time()
    for image_file in images:
        img = cv2.imread(image_file)
        img = cv2.resize(img, frame_size)  # 이미지 리사이징
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # 코너 그리기
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    if len(objpoints) > 0 and len(imgpoints) > 0:
        # 카메라 캘리브레이션
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)
        np.savez("camera_calib_SPC.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("카메라 캘리브레이션 완료. 결과가 저장되었습니다.")
    else:
        print("체스보드 코너를 찾을 수 있는 이미지가 충분하지 않습니다.")
    
    end_time = time.time()
    print(f"캘리브레이션에 걸린 시간: {end_time - start_time} 초")
