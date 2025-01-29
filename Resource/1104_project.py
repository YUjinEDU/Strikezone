import cv2
import numpy as np
np.set_printoptions(precision=2, suppress=True)

# 이미지 크기를 줄이는 함수
def resize_image(image, scale_percent=50):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

# 이미지를 불러오고 크기를 줄이는 함수
def load_and_resize_images(image_paths, scale_percent=50):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Error loading image {path}")
            continue
        img_resized = resize_image(img, scale_percent)
        images.append(img_resized)
    return images

# 체스보드 코너 찾기 함수
def FindCornerPoints(src_img, patternSize):
    found, corners = cv2.findChessboardCorners(src_img, patternSize)
    if not found:
        return found, corners
    
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)    
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term_crit)
    return found, corners

# 체스보드 패턴 크기
patternSize = (10, 7)

# 이미지 파일 경로
image_paths = [
    'OpenCV_Python_2ndEdition_Examples/data/photo_(1).jpg',
    'OpenCV_Python_2ndEdition_Examples/data/photo_(2).jpg',
    'OpenCV_Python_2ndEdition_Examples/data/photo_(3).jpg',
    'OpenCV_Python_2ndEdition_Examples/data/photo_(4).jpg',
    'OpenCV_Python_2ndEdition_Examples/data/photo_(5).jpg'
]

# 이미지 불러오기 및 크기 줄이기
images = load_and_resize_images(image_paths, scale_percent=50)

# 이미지 크기 정보 가져오기
if len(images) > 0:
    imageSize = (images[0].shape[1], images[0].shape[0])  # (width, height)
else:
    print("No images loaded")
    exit(1)

# 월드 좌표 설정 (Z = 0)
xN, yN = patternSize  # (6, 3)
mW = np.zeros((xN * yN, 3), np.float32)  # (18, 3)
mW[:, :2] = np.mgrid[0:xN, 0:yN].T.reshape(-1, 2)  # mW points on Z = 0
mW[:, :2] += 1

# 3D (obj_points) <-> 2D (img_points) 매칭
obj_points = []
img_points = []
for img in images:
    found, corners = FindCornerPoints(img, patternSize)
    if found:
        obj_points.append(mW)
        img_points.append(corners)

# 캘리브레이션 수행 전 obj_points와 img_points에 데이터가 있는지 확인
if len(obj_points) == 0 or len(img_points) == 0:
    print("No valid images for calibration")
    exit(1)

# 카메라 캘리브레이션
errors, K, dists, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, imageSize, None, None)
print("calibrateCamera: errors=", errors)
np.savez('OpenCV_Python_2ndEdition_Examples/data/calib_project.npz', K=K, dists=dists, rvecs=rvecs, tvecs=tvecs)

print("calibrated K=\n", K)
print("dists=", dists)

# 축 및 3D 물체 그리기
index = [0, 5, 17, 12]  # 4-corner index
axis3d = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)  # -Z : towards the camera

for i, img in enumerate(images):
    print("tvec[{}]={}".format(i, tvecs[i].T))
    print("rvec[{}]={}".format(i, rvecs[i].T))  # rvecs[i].shape = (3, 1)

    R, _ = cv2.Rodrigues(rvecs[i])  # R.shape = (3, 3)
    print("R[{}]=\n{}".format(i, R))

    # 축 그리기
    axis_2d, _ = cv2.projectPoints(axis3d, rvecs[i], tvecs[i], K, dists)
    axis_2d = np.int32(axis_2d).reshape(-1, 2)
    cv2.line(img, tuple(axis_2d[0]), tuple(axis_2d[1]), (255, 0, 0), 3)
    cv2.line(img, tuple(axis_2d[0]), tuple(axis_2d[2]), (0, 255, 0), 3)
    cv2.line(img, tuple(axis_2d[0]), tuple(axis_2d[3]), (0, 0, 255), 3)

    # Z = 0에서 pW 그리기
    pW = mW[index]  # 4-corners' coord (x, y, 0)
    p1, _ = cv2.projectPoints(pW, rvecs[i], tvecs[i], K, dists)
    p1 = np.int32(p1)
    cv2.drawContours(img, [p1], -1, (0, 255, 255), -1)
    cv2.polylines(img, [p1], True, (0, 255, 0), 2)

    # Z = -2에서 pW 그리기
    pW[:, 2] = -2  # 4-corners' coord (x, y, -2)
    p2, _ = cv2.projectPoints(pW, rvecs[i], tvecs[i], K, dists)
    p2 = np.int32(p2)
    cv2.polylines(img, [p2], True, (0, 0, 255), 2)

    # 두 사각형 사이의 엣지 그리기
    for j in range(4):
        x1, y1 = p1[j][0]  # Z = 0
        x2, y2 = p2[j][0]  # Z = -2
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 재투영 오류 계산
    pts, _ = cv2.projectPoints(mW, rvecs[i], tvecs[i], K, dists)
    errs = cv2.norm(img_points[i], np.float32(pts))
    print("errs[{}]={}".format(i, errs))

    # 이미지 보여주기
    cv2.imshow(f'img{i + 1}', img)

cv2.waitKey()
cv2.destroyAllWindows()
