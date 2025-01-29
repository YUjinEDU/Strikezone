import cv2
import cv2.aruco as aruco

# ARUCO 마커 사전 생성
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# 여러 개의 ARUCO 마커 생성
for i in range(5):  # 5개의 마커 생성 (필요한 만큼 생성)``
    marker_image = aruco.generateImageMarker(aruco_dict, i, 700)
    cv2.imwrite(f"aruco_marker2_{i}.png", marker_image)

# 예시로 첫 번째 마커 출력
cv2.imshow("ARUCO Marker", marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
print(cv2.__version__)
