# AR Strike Zone: 증강현실 야구 투구 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Dash](https://img.shields.io/badge/Plotly%20Dash-2.x-brightgreen)](https://dash.plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AR Strike Zone**은 단일 카메라를 사용하여 실시간으로 투구 궤적을 추적하고, 스트라이크 존을 증강현실(AR)로 시각화하며, 투구 데이터를 심층 분석하는 시스템입니다. 야구 선수 및 코치들이 투구 정확도와 구속을 과학적으로 측정하고 개선할 수 있도록 돕습니다.

![시스템 실행 화면](https://user-images.githubusercontent.com/49531398/189498223-9988b495-2c8c-4a37-b459-a681d45c60fd.gif)
*(참고: 위 GIF는 프로젝트의 기능 시연 예시입니다.)*

---

##  주요 기능 (Key Features)

*   **⚾ 실시간 AR 스트라이크 존 시각화**: ArUco 마커를 기준으로 3D 공간에 스트라이크 존과 볼 존을 시각화합니다.
*   **👁️ 정확한 공 추적**: HSV 컬러 모델 기반의 공 검출과 칼만 필터(Kalman Filter)를 결합하여 빠르고 불규칙한 공의 움직임을 안정적으로 추적합니다.
*   **👨‍⚖️ 자동 스트라이크/볼 판정**: 공이 두 개의 가상 평면(볼 존, 스트라이크 존)을 통과하는지 여부와 위치를 분석하여 스트라이크/볼을 자동으로 판정합니다.
*   **🚀 투구 속도 측정**: 3D 공간상에서 공의 이동 거리와 시간을 계산하여 투구 속도(km/h)를 실시간으로 측정하고 표시합니다.
*   **📈 인터랙티브 웹 대시보드**: [Plotly Dash](https://dash.plotly.com/) 기반의 웹 대시보드를 통해 3D 투구 궤적, 투구 위치 분포, 통계 데이터(S/B/O 카운트, 구속 등)를 다각도로 분석할 수 있습니다.
*   **✋ 제스처 컨트롤**: MediaPipe를 활용한 손 인식 기능을 통해, 손바닥을 펴는 동작으로 AR 기능을 활성화할 수 있습니다.
*   **📊 실시간 전광판**: 스트라이크, 볼, 아웃 카운트를 AR 화면에 실시간으로 표시하여 실제 야구 경기와 같은 경험을 제공합니다.

---

## 시스템 아키텍처 (System Architecture)

본 시스템은 여러 모듈이 유기적으로 연동하여 동작합니다. 전체적인 데이터 흐름과 모듈 간의 관계는 아래와 같습니다.

```mermaid
graph TD
    A[카메라 입력<br/>(Live Camera / Video)] --> B{main.py<br/>(Core Logic)};

    subgraph "입력 처리 및 AR 설정"
        C[camera.py<br/>카메라 관리]
        D[aruco_detector.py<br/>ArUco 마커 검출/좌표계 설정]
    end

    subgraph "객체 추적 및 분석"
        E[tracker_v1.py<br/>공/손 감지]
        F[kalman_filter.py<br/>3D 궤적 안정화]
    end

    subgraph "시각화 및 출력"
        G[dashboard.py<br/>웹 대시보드]
        H[baseball_scoreboard.py<br/>AR 전광판]
        I[effects.py<br/>판정 텍스트 효과]
    end
    
    J[config.py</br>전역 설정]

    B --> C & D & E & F & G & H & I & J;

    subgraph "처리 파이프라인"
        P1[1. 프레임 캡처] --> P2[2. ArUco 마커 검출<br/>(3D 좌표계 설정)];
        P2 --> P3[3. 공 검출 및 3D 위치 변환];
        P3 --> P4[4. 칼만 필터 적용<br/>(궤적 보정)];
        P4 --> P5[5. 속도 계산 및 궤적 분석];
        P5 --> P6[6. 스트라이크/볼 판정];
    end

    B -- Orchestrates --> P1;

    subgraph "최종 결과"
        O1[AR 화면 출력<br/>(OpenCV)]
        O2[데이터 분석 대시보드<br/>(Web Browser)]
    end

    P6 --> O1;
    P5 --> O2;
    H --> O1;
    I --> O1;

```

---

## 기술적 상세 (Technical Details)

### 1. ArUco 마커 기반 좌표계 설정

- `aruco_detector.py` 모듈은 ArUco 마커를 검출하여 카메라와 마커 사이의 상대적인 3D 위치 및 회전(`rvecs`, `tvecs`)을 계산합니다. 이는 실제 공간에 안정적인 3D 좌표계를 설정하는 기준이 됩니다.

```python
# src/main.py
# ArUco 마커 검출 및 포즈 추정
corners, ids, rejected = aruco_detector.detect_markers(analysis_frame)
if ids is not None:
    rvecs, tvecs = aruco_detector.estimate_pose(corners)
```

### 2. 공 추적 및 3D 위치 추정

- **(1) 2D 공 검출**: `tracker_v1.py`의 `BallDetector`는 HSV 색상 공간에서 정의된 범위 내의 객체를 찾아 공의 2D 위치(`center`, `radius`)를 식별합니다.
- **(2) 3D 깊이 추정**: 검출된 공의 픽셀 반지름과 실제 공의 반지름, 카메라 초점 거리를 이용하여 3D 깊이(Z 좌표)를 추정합니다.
- **(3) 좌표 변환**: 2D 이미지 좌표와 추정된 깊이를 결합하여 카메라 좌표계 기준 3D 좌표를 계산하고, 이를 다시 ArUco 마커 기준 좌표계로 변환합니다.
- **(4) 칼만 필터 적용**: `kalman_filter.py`의 3D 칼만 필터가 노이즈를 제거하고 공의 궤적을 부드럽게 보정합니다.

```python
# src/main.py
# 공 검출 및 3D 위치 계산
center, radius, _ = ball_detector.detect(analysis_frame)
if center and radius > 0.4:
    # 깊이 추정
    estimated_Z = (camera_matrix[0, 0] * BALL_RADIUS_REAL) / radius
    ball_3d_cam = np.array([...]) # 카메라 좌표계 기준 3D 위치
    
    # 마커 좌표계로 변환
    point_in_marker_coord = aruco_detector.point_to_marker_coord(ball_3d_cam, rvec, tvec)
  
    # 칼만 필터 업데이트
    filtered_point = kalman_tracker.update_with_gating(np.array(point_in_marker_coord, dtype=np.float32))
```

### 3. 스트라이크/볼 판정 로직

- 시스템은 2단계 판정 로직을 사용합니다. 공이 통과해야 하는 두 개의 가상 평면(`ball_zone_corners`, `ball_zone_corners2`)이 마커를 기준으로 정의되어 있습니다.
- `aruco_detector.signed_distance_to_plane` 함수를 이용해 공과 각 평면 사이의 부호 있는 거리를 계산합니다.
- 공이 첫 번째 평면을 통과(`zone_step1 = True`)한 후, 두 번째 평면의 스트라이크 존 영역(`is_in_polygon2`)을 통과하면 **스트라이크**로 판정합니다.

```python
# src/main.py
# 1단계: 첫 번째 평면 통과 감지
if not zone_step1 and prev_distance_to_plane1 > pass_threshold >= distance_to_plane1 and is_in_polygon1:
    zone_step1 = True

# 2단계: 두 번째 평면 통과 및 스트라이크 판정
if zone_step1 and not zone_step2 and prev_distance_to_plane2 > pass_threshold >= distance_to_plane2 and is_in_polygon2:
    print("****** Plane 2 Passed - STRIKE! ******")
    # 스트라이크 처리 로직
```

### 4. 실시간 웹 대시보드

- `dashboard.py`는 `Dash`와 `Plotly`를 사용하여 웹 기반 대시보드를 생성합니다.
- `main.py`에서 계산된 투구 데이터(궤적, S/B 카운트, 구속 등)를 주기적으로 업데이트하여 사용자는 웹 브라우저를 통해 실시간으로 분석 결과를 확인할 수 있습니다.

---

## 사용 방법 (How to Use)

### 1. 요구사항 (Prerequisites)

- Python 3.9+
- 필요한 라이브러리:
  ```
  opencv-python
  numpy
  imutils
  mediapipe
  dash
  plotly
  ```

### 2. 설치 (Installation)

1.  **프로젝트 클론**
    ```bash
    git clone https://github.com/your-username/AR_StrikeZone.git
    cd AR_StrikeZone
    ```
2.  **가상환경 생성 및 활성화 (권장)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **필수 라이브러리 설치**
    ```bash
    pip install opencv-python numpy imutils mediapipe dash plotly
    ```
    (추후 `requirements.txt` 파일을 제공할 예정입니다.)

### 3. 설정 (Configuration)

- `src/config.py` 파일에서 각종 파라미터를 수정할 수 있습니다.
    - `GREEN_LOWER`, `GREEN_UPPER`: 추적할 공의 HSV 색상 범위
    - `ARUCO_MARKER_SIZE`: 사용하는 ArUco 마커의 실제 크기 (미터 단위)
    - `CALIBRATION_PATH`: 카메라 캘리브레이션 파일 경로

### 4. 실행 (Execution)

1.  `src` 디렉토리로 이동합니다.
    ```bash
    cd src
    ```
2.  `main.py`를 실행합니다.
    ```bash
    python main.py
    ```
3.  실행 후 터미널에서 `1: 카메라` 또는 `2: 비디오`를 선택하여 입력 소스를 결정합니다.

**주요 단축키:**
*   `q`: 프로그램 종료
*   `r`: 카운트 및 상태 리셋
*   `c`: 투구 기록 데이터 초기화
*   `(Space)`: (비디오 모드) 재생/일시정지

---

## 코드 구조 (Code Structure)

```
/src
├── main.py                 # 메인 실행 파일, 전체 워크플로우 제어
├── config.py               # 시스템 전역 설정 및 상수 관리
├── camera.py               # 카메라 캡처 및 캘리브레이션 관리
├── aruco_detector.py       # ArUco 마커 검출 및 3D 좌표 변환
├── tracker_v1.py           # 공/손 검출 및 궤적 추적 로직
├── kalman_filter.py        # 3D 칼만 필터 구현
├── baseball_scoreboard.py  # AR 전광판 UI
├── effects.py              # 'STRIKE', 'BALL' 등 시각 효과
├── dashboard.py            # Dash 기반 웹 대시보드 서버
└── ...
```

---

## 향후 개선 방향 (Future Work)

-   [ ] **딥러닝 기반 객체 검출**: YOLO, SSD 등 딥러닝 모델을 도입하여 더 다양한 환경에서 강인한 공 검출 성능 확보
-   [ ] **사용자 맞춤형 존 설정**: 사용자의 키나 자세에 맞춰 스트라이크 존을 동적으로 조절하는 기능


---

## 라이선스 (License)


