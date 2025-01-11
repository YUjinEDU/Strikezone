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


---

네, **카메라 한 대**만 사용해도 깊이를 어느 정도 추정할 수 있습니다. 

---

## 어떻게 가능한가?

### 1. 단안(싱글) 카메라에서의 깊이 추정 원리

- **단안 카메라**로 촬영된 영상에서, 우리가 어떤 물체(예: 공)의 **실제 크기**(지름 등)를 알고 있다면, 카메라로 본 물체의 **화면상 크기(픽셀)** 와 **카메라 초점거리**(focal length) 등을 활용하여 깊이를 추정할 수 있습니다.
- 흔히 \(\displaystyle Z = \frac{\text{물체의 실제 너비} \times \text{카메라 초점거리}}{\text{물체의 이미지 너비}}\) 와 같은 단안 카메라의 기본적인 거리 추정 공식을 사용합니다.  

### 2. 전제 조건

1. **물체(공)의 실제 크기**를 알고 있어야 함  
   \- 예: 야구공 지름이 7.3cm라는 점을 확실히 알고 있다고 가정.  
2. **카메라가 충분히 보정(캘리브레이션)되어 있어야 함**  
   \- 즉, 카메라 매트릭스(초점거리 \(f_x, f_y\), 광학 중심 \(c_x, c_y\))를 알고 있어야 합니다.  
3. **물체가 정확히 검출되고, 물체의 이미지에서의 너비(또는 반지름)가 제대로 측정**되어야 함  
   \- 색상 추적, 에지 검출, HoughCircles 등을 통해 화면에서 공의 픽셀 크기를 얻어야 합니다.

### 3. 실제 예시

1. **공의 실제 지름** = 0.073m (7.3cm)  
2. **카메라 매트릭스**에서 \(\displaystyle f_x = 1000\) (픽셀 단위 예시)  
3. **이미지에서 공의 반지름** = 20픽셀 → 지름은 40픽셀  
4. **깊이 추정**  
   \[
   Z = \frac{공의 실제 너비 \times f_x}{이미지에서 측정된 너비} 
     = \frac{0.073 \times 1000}{0.04 \times 1000} 
     = \frac{0.073}{0.04} \approx 1.825 \text{(m)}
   \]  
   \- (참고) 여기서는 화면상의 “지름”을 40픽셀이라 했으니, “너비”는 40, “반지름”은 20일 것입니다.  
   \- 실제 사용 시 정확한 값으로 대입해 계산합니다.

### 4. 한계점

1. **정밀도 문제**  
   \- 실제 크기 추정 방식은 물체가 멀어질수록 작은 픽셀 오차가 큰 거리 오차로 이어질 수 있습니다.  
   \- 예: 공이 카메라에서 멀리 있을수록 이미지 상 크기가 작아져, 1~2픽셀의 차이로도 상당한 깊이 오차가 발생할 수 있습니다.
2. **물체 정면 가정**  
   \- 단안 카메라 공식은 물체가 카메라에 대해 정면으로 놓여 있고, 구형·원형 등일 때 더 단순하게 적용할 수 있습니다.  
   \- 공이 기울어져 있거나 부분 가림이 있는 경우, 정확도가 떨어질 수 있습니다.
3. **조명·배경 영향**  
   \- 색상이나 윤곽 기반으로 공을 추적할 때, 조명과 배경에 따라 검출 오차가 발생할 수 있습니다.

### 5. 스테레오(이안) 카메라와 비교

- **스테레오 카메라**(좌/우 두 대)는 삼각 측량 원리를 이용해, 별도의 물체 크기 정보 없이도 더 정확하게 깊이를 구할 수 있습니다.  
- 단안 카메라 방식은 상대적으로 구현이 간단하고 추가 하드웨어 비용이 없지만, **정확도와 안정성** 측면에서는 스테레오보다 떨어질 수 있습니다.

---

## 결론

- **카메라 1대**만으로도, 공의 실제 크기를 알고 있다면 **단안 카메라 거리 추정 공식**을 통해 깊이를 어느 정도 추정할 수 있습니다.  
- 다만, 공이 실제로 어느 정도 크게 보이는지(픽셀 크기)를 정확히 잡아내야 하고, 거리에 따라 오차가 커질 수 있으니 이 점을 유의하면 됩니다. 
- 더 정확하고 안정적인 **깊이 추정**을 원한다면, 스테레오 카메라나 다른 센서(예: 깊이 카메라, LiDAR)를 사용하는 방안도 고려해볼 수 있습니다.