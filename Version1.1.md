### Z축 정보를 추가한 스트라이크 존 감지 코드

안녕하세요! 기존의 2D 증강현실(AR) 스트라이크 존 감지 코드를 기반으로, Z축(깊이) 정보를 추가하여 3D 스트라이크 존을 구현해보겠습니다. 이를 통해 공의 깊이까지 고려하여 더 정확한 스트라이크 감지가 가능해집니다.

아래는 수정된 전체 코드와 각 변경 사항에 대한 설명입니다.

---

#### **변경 사항 요약:**
1. **깊이(Z) 추정 추가:** 공의 반지름을 이용해 깊이를 계산합니다.
2. **2D 픽셀 좌표를 3D 카메라 좌표로 변환:** X, Y, Z 좌표를 계산합니다.
3. **카메라 좌표계를 마커 좌표계로 변환:** 공의 위치를 마커 기준으로 변환합니다.
4. **3D 스트라이크 존 내 포함 여부 확인:** Z축을 포함하여 공이 스트라이크 존 내에 있는지 확인합니다.

---

### 수정된 전체 코드

```python
#####################################
# Version 1.1
# Date: 2025.04.27
# Author: 박유진
# Description: 3D 증강현실을 구현한 코드, 손을 인식하여 그 손의 위치에 따라 2D 및 3D 증강현실이 나타난다.
# 공의 깊이 정보(Z축)를 추가하여 3D 스트라이크 존 및 아웃 존을 구현하였다.
# 스테레오 비젼은 아직 구현되지 않았으나, 향후 추가 가능.
########################################

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

# 공의 실제 지름 (단위: 미터)
known_width = 0.073  # 예: 야구공 지름 약 7.3cm

# 공중에 띄울 스트라이크 존의 3D 좌표 (마커 기준으로 상대적인 위치)
strike_zone_corners = np.array([
    [-0.08, 0.1, 0],  # Bottom-left
    [0.08, 0.1, 0],   # Bottom-right
    [0.08, 0.3, 0],   # Top-right
    [-0.08, 0.3, 0]   # Top-left
], dtype=np.float32)

# 스트라이크 존의 높이 (Z축 방향)
strike_zone_height = 0.2  # 예: 20cm 높이

# 회전 행렬 생성 (90도 회전)
rotation_matrix = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float32)

# 스트라이크 존의 각 점에 회전 행렬 적용
strike_zone_corners = np.dot(strike_zone_corners, rotation_matrix.T)

# 스트라이크 존 3D 볼륨 정의 (마커 좌표계 기준)
def define_strike_zone_3d(corners, height):
    """ 스트라이크 존을 3D 볼륨으로 정의 """
    # 바닥 사각형
    base = corners
    # 상단 사각형 (높이 추가)
    top = corners + np.array([0, 0, height])
    # 8개의 꼭지점
    strike_zone_3d = np.vstack((base, top))
    return strike_zone_3d

strike_zone_3d = define_strike_zone_3d(strike_zone_corners, strike_zone_height)

# 카메라 설정
cap = cv2.VideoCapture(1)  # 비브캠 사용
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

# 색상 기반 객체 추적 설정
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
redLower = (170, 210, 150)
redUpper = (173, 220, 175)
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
            # 각 손가락 끝의 랜드마크 위치를 확인
            tips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
            if all(tip.y < hand_landmarks.landmark[tip_index].y for tip, tip_index in zip(tips, [6, 10, 14, 18])):  # 손가락이 펴진 상태인지 확인
                return True
    return False

def 스트라이크_존에_포함(p_marker):
    """ 공의 3D 좌표가 스트라이크 존 내에 있는지 확인 """
    x, y, z = p_marker[:3]
    half_length = 0.08  # 스트라이크 존 길이의 절반 (예: 16cm 길이)
    half_width = 0.10   # 스트라이크 존 너비의 절반 (예: 20cm 너비)
    height = strike_zone_height  # 스트라이크 존 높이
    
    return (-half_length < x < half_length) and (-half_width < y < half_width) and (0 < z < height)

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

        # 마커가 감지되면
        if ids is not None:
            # 마커를 그리기
            #frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 마커 좌표 추정
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

            for rvec, tvec in zip(rvecs, tvecs):
                #cv2.drawAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                
                # 스트라이크 존 그리기 (2D 투영)
                projected_points, _ = cv2.projectPoints(strike_zone_corners, rvec, tvec, camera_matrix, dist_coeffs)
                projected_points = projected_points.reshape(-1, 2)
                projected_points = projected_points.astype(int)  # 좌표를 정수형으로 변환

                # 사각형 그리기
                for i in range(4):
                    pt1 = tuple(projected_points[i])
                    pt2 = tuple(projected_points[(i+1) % 4])
                    cv2.line(frame, pt1, pt2, (0, 0, 0), 4)

                # 사각형 내부에 격자 그리기
                draw_grid(frame, projected_points, 3)  # 3등분하여 격자 그리기

                # 녹색 및 형광 빨간색 공 추적
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
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        if radius > 3:
                            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                            pts.appendleft(center)

                            if is_point_in_polygon(center, projected_points):
                                detected_points.append(center)
                                strike_count += 1
                                cv2.putText(frame, "Strike", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)

                '''
                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 0), thickness)
                '''

                # Z축 정보 추가
                # rvec과 tvec을 이용하여 회전 행렬 R과 변환 벡터 T를 얻습니다.
                R, _ = cv2.Rodrigues(rvec)
                T_marker_to_camera = np.hstack((R, tvec.reshape(3, 1)))
                T_marker_to_camera = np.vstack((T_marker_to_camera, [0, 0, 0, 1]))
                T_camera_to_marker = np.linalg.inv(T_marker_to_camera)

                # 공이 검출되었는지 확인
                if center is not None and radius > 3:
                    # 깊이(Z) 추정
                    f_x = camera_matrix[0, 0]
                    depth = (known_width * f_x) / (2 * radius)

                    # 3D 카메라 좌표로 변환
                    c_x = camera_matrix[0, 2]
                    c_y = camera_matrix[1, 2]
                    X = (center[0] - c_x) * depth / f_x
                    Y = (center[1] - c_y) * depth / camera_matrix[1, 1]
                    Z = depth
                    p_camera = np.array([X, Y, Z, 1])

                    # 마커 좌표계로 변환
                    p_marker = T_camera_to_marker @ p_camera
                    p_marker = p_marker / p_marker[3]  # 호모제네이즈 좌표

                    # 스트라이크 존 내에 있는지 확인
                    if 스트라이크_존에_포함(p_marker[:3]):
                        strike_count += 1
                        cv2.putText(frame, "3D Strike", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        out_count += 1
                        cv2.putText(frame, "3D Out", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for point in detected_points:
            cv2.circle(frame, point, 8, (0, 0, 0), -1)
        
        # 스트라이크 카운트가 3의 배수일 때 아웃 카운트 증가
        if strike_count % 3 == 0 and strike_count != 0:
            out_count += 1
            strike_count = 0  # 스트라이크 카운트 초기화

        # 스트라이크 및 아웃 카운트 표시
        cv2.putText(frame, f"S {strike_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"O {out_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # 결과 출력
    cv2.imshow('ARUCO Tracker with 3D Strike Zone', frame)

    # 키 입력 처리
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        reset()

cap.release()
cv2.destroyAllWindows()
```

---

### 주요 변경 사항 설명

1. **공의 깊이(Z) 추정:**
   - 공의 실제 지름(`known_width`)을 미터 단위로 설정합니다. 예를 들어, 야구공의 지름은 약 7.3cm이므로 `0.073` 미터로 설정했습니다.
   - 공의 이미지에서 반지름(`radius`)을 이용해 깊이(`depth`)를 계산합니다.
     ```python
     depth = (known_width * f_x) / (2 * radius)
     ```

2. **2D 픽셀 좌표를 3D 카메라 좌표로 변환:**
   - 공의 중심 좌표 `(u, v)`와 깊이 `Z`를 이용하여 3D 카메라 좌표 `(X, Y, Z)`를 계산합니다.
     ```python
     X = (center[0] - c_x) * depth / f_x
     Y = (center[1] - c_y) * depth / camera_matrix[1, 1]
     Z = depth
     p_camera = np.array([X, Y, Z, 1])
     ```

3. **카메라 좌표계를 마커 좌표계로 변환:**
   - ARUCO 마커로부터 얻은 회전 벡터(`rvec`)와 변환 벡터(`tvec`)를 이용해 변환 행렬(`T_camera_to_marker`)을 생성합니다.
     ```python
     R, _ = cv2.Rodrigues(rvec)
     T_marker_to_camera = np.hstack((R, tvec.reshape(3, 1)))
     T_marker_to_camera = np.vstack((T_marker_to_camera, [0, 0, 0, 1]))
     T_camera_to_marker = np.linalg.inv(T_marker_to_camera)
     ```
   - 3D 카메라 좌표를 마커 좌표계로 변환합니다.
     ```python
     p_marker = T_camera_to_marker @ p_camera
     p_marker = p_marker / p_marker[3]  # 호모제네이즈 좌표
     ```

4. **3D 스트라이크 존 내 포함 여부 확인:**
   - 변환된 마커 좌표계의 공 위치가 스트라이크 존 내에 있는지 확인합니다.
     ```python
     if 스트라이크_존에_포함(p_marker[:3]):
         strike_count += 1
         cv2.putText(frame, "3D Strike", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
     else:
         out_count += 1
         cv2.putText(frame, "3D Out", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
     ```

5. **스트라이크 존 3D 볼륨 정의:**
   - 스트라이크 존을 3D 볼륨으로 정의하여, 공의 위치가 이 볼륨 내에 있는지 확인합니다.
     ```python
     def define_strike_zone_3d(corners, height):
         """ 스트라이크 존을 3D 볼륨으로 정의 """
         # 바닥 사각형
         base = corners
         # 상단 사각형 (높이 추가)
         top = corners + np.array([0, 0, height])
         # 8개의 꼭지점
         strike_zone_3d = np.vstack((base, top))
         return strike_zone_3d
     ```

6. **함수 추가:**
   - `스트라이크_존에_포함(p_marker)` 함수는 공의 3D 좌표가 스트라이크 존 내에 있는지 확인합니다.
     ```python
     def 스트라이크_존에_포함(p_marker):
         """ 공의 3D 좌표가 스트라이크 존 내에 있는지 확인 """
         x, y, z = p_marker[:3]
         half_length = 0.08  # 스트라이크 존 길이의 절반 (예: 16cm 길이)
         half_width = 0.10   # 스트라이크 존 너비의 절반 (예: 20cm 너비)
         height = strike_zone_height  # 스트라이크 존 높이
         
         return (-half_length < x < half_length) and (-half_width < y < half_width) and (0 < z < height)
     ```

7. **스트라이크 및 아웃 카운트 표시:**
   - 공이 스트라이크 존 내에 있을 때와 아닐 때 각각 다른 텍스트와 색상으로 표시합니다.
     ```python
     if 스트라이크_존에_포함(p_marker[:3]):
         strike_count += 1
         cv2.putText(frame, "3D Strike", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
     else:
         out_count += 1
         cv2.putText(frame, "3D Out", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
     ```

---

### 추가 고려 사항

1. **카메라 캘리브레이션:**
   - 정확한 3D 변환을 위해서는 카메라 캘리브레이션이 정확히 이루어져야 합니다. `camera_calib.npz` 파일이 올바르게 생성되었는지 확인하세요.

2. **공의 실제 크기 설정:**
   - `known_width` 변수는 실제 공의 지름을 미터 단위로 설정해야 합니다. 사용하는 공의 실제 크기에 맞게 조정하세요.

3. **스트라이크 존 크기 조정:**
   - `half_length`, `half_width`, `strike_zone_height` 값은 실제 스트라이크 존의 크기에 맞게 조정해야 합니다.

4. **공 검출 정확성:**
   - 공 검출 부분이 색상 기반이므로, 환경에 따라 색상 범위를 조정하거나 더 정교한 공 검출 방법을 고려할 수 있습니다.

5. **실시간 성능 최적화:**
   - Z축 계산과 3D 변환이 실시간 성능에 영향을 줄 수 있으므로, 최적화가 필요할 수 있습니다.

---

### 결론

수정된 코드는 이제 공의 Z축 정보를 추가하여 3D 스트라이크 존 감지가 가능합니다. 이를 통해 공이 스트라이크 존 내에 있는지 보다 정확하게 판단할 수 있으며, 증강현실 경험을 더욱 향상시킬 수 있습니다. 필요에 따라 스트라이크 존의 크기와 공 검출 방법을 조정하여 최적의 성능을 얻으시기 바랍니다. 추가적인 질문이나 문제가 있으면 언제든지 문의해주세요!





---

아래는 **3D 방식만** 사용하는 코드 예시입니다. 기존 코드에서 **2D 판정(다각형 내부 감지 등)** 과 관련된 부분(예: `is_point_in_polygon`, `detected_points` 등)을 제거하고, **3D 스트라이크 존**만 사용하는 로직으로 구성했습니다.

---

## 주요 변경 사항

1. **2D 판정 로직 제거**  
   - `is_point_in_polygon` 함수와 해당 함수 호출을 제거했습니다.  
   - `detected_points` 리스트 및 관련 로직을 제거했습니다.

2. **3D 스트라이크 존 판정만 사용**  
   - `스트라이크_존에_포함(p_marker[:3])` 함수만 사용하여, 공이 마커 좌표계에서 정의된 3D 스트라이크 존에 포함되는지 확인합니다.  
   - 포함 시 `3D Strike`, 그렇지 않으면 `3D Out`으로 처리합니다.

3. **2D 결과 그리기는 유지**  
   - 마커 위에 스트라이크 존을 2D로 시각화(`cv2.projectPoints`)하는 부분은 유지했습니다.  
   - AR 화면에서 사각형을 표시해, 시각적 참조를 위해 남겨두었습니다.

4. **기타**  
   - `skip_frames`, 손 인식 로직, 카메라 캘리브레이션, 색상 기반 공 추적 등은 그대로 유지했습니다.  
   - 실제 적용 시, `known_width`, `strike_zone_height` 등은 사용하는 공 및 원하는 존 크기에 맞춰 조정하세요.

---

```python
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
cap = cv2.VideoCapture(0)  # 카메라 인덱스(환경에 맞게 수정)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
```

---

### 코드 사용 시 참고 사항

1. **카메라 인덱스**  
   - `cap = cv2.VideoCapture(0)` 부분에서 `0` 대신 웹캠/외부 카메라 인덱스를 환경에 맞게 설정하세요.

2. **스트라이크 존 크기 조절**  
   - `strike_zone_corners` 및 `strike_zone_height`로 결정되는 3D 존의 크기를 원하는 값으로 조정하세요.  
   - `스트라이크_존에_포함()` 함수에서 `half_length`, `half_width`도 실제 요구 사항에 맞게 수정할 수 있습니다.

3. **공의 실제 지름**  
   - `known_width = 0.073` (7.3cm)는 야구공 기준 예시이므로, 사용하려는 공의 실제 크기로 변경하세요.

4. **조명/색상 환경**  
   - `greenLower`, `greenUpper`, `redLower`, `redUpper` 값은 조명, 배경 등에 따라 달라질 수 있습니다. 환경에 맞춰 HSV 범위를 조정하세요.

5. **마커 크기**  
   - `aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)` 에서 `0.05`는 마커의 한 변 길이(미터)입니다. 실제 마커 크기에 맞춰 조정해야 3D 계산 정확도가 높아집니다.

---

이렇게 수정된 코드에서는 **3D 좌표 변환을 통한 스트라이크 존 판정**만 수행합니다. 2D 다각형 내부 감지 로직이 제거되어, 한 번 공이 검출될 때 3D 판정만 진행됩니다. 원하시는 대로 3D 방식에만 집중하여 스트라이크/아웃을 판정할 수 있습니다. 추가로 환경에 맞춰 파라미터(공 크기, 색상 범위 등)를 조정해보세요. 필요한 내용이 있으면 언제든 말씀해 주세요!