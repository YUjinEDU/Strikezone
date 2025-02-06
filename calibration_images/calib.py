import cv2
import numpy as np
import glob

# 체스보드의 내부 코너 수 (가로, 세로) → 실제 체스보드와 일치하는지 확인!
chessboard_size = (8, 11)

# 종료 기준 설정
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D 점 생성 (체스보드의 실제 좌표)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 이미지 포인트와 객체 포인트 저장 리스트
objpoints = []  # 3D 점
imgpoints = []  # 2D 점

# 캘리브레이션 이미지 경로 확인
images = glob.glob('*.jpg')
if not images:
    print("경고: calibration_images 폴더에 이미지가 없습니다.")
    exit()

# 이미지 처리 루프
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    if img is None:
        print(f"경고: {fname} 이미지를 읽을 수 없습니다.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)  # 노이즈 제거
    gray = cv2.equalizeHist(gray)  # 명암 대비 개선
    
    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # 코너 위치 개선
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

        # 코너 시각화 (디버깅용)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow(f'Chessboard Corners - Image {idx}', img)
        cv2.waitKey(500)  # 0.5초 대기
    else:
        print(f"경고: {fname}에서 체스보드 코너를 찾지 못했습니다.")

cv2.destroyAllWindows()

# 캘리브레이션 수행 전 검증
if len(objpoints) == 0:
    print("오류: 캘리브레이션을 위한 유효한 이미지가 없습니다.")
    exit()

# 카메라 캘리브레이션 수행
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("카메라 매트릭스:\n", mtx)
    print("왜곡 계수:\n", dist)
else:
    print("캘리브레이션 실패")
