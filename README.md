# AR Strike Zone: 증강현실 야구 투구 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Dash](https://img.shields.io/badge/Plotly%20Dash-2.x-brightgreen)](https://dash.plotly.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AR Strike Zone**은 단일 카메라를 사용하여 실시간으로 투구 궤적을 추적하고, 스트라이크 존을 증강현실(AR)로 시각화하며, 투구 데이터를 심층 분석하는 시스템입니다. 야구 선수 및 코치들이 투구 정확도와 구속을 과학적으로 측정하고 개선할 수 있도록 돕습니다.

![시스템 실행 화면](https://user-images.githubusercontent.com/49531398/189498223-9988b495-2c8c-4a37-b459-a681d45c60fd.gif)
*(참고: 위 GIF는 프로젝트의 기능 시연 예시입니다.)*

---

##  주요 기능 (Key Features)

*   **⚾ 실시간 AR 스트라이크 존 시각화**: ArUco 마커를 기준으로 3D 공간에 스트라이크 존과 볼 존을 시각화합니다.
*   **👁️ 정확한 공 추적**: HSV 컬러 모델 기반의 공 검출과 3D 칼만 필터(Kalman Filter)를 결합하여 빠르고 불규칙한 공의 움직임을 안정적으로 추적합니다.
*   **👨‍⚖️ 2단계 자동 스트라이크/볼 판정**: 공이 두 개의 가상 평면(Plane1, Plane2)을 순차적으로 통과하는지 여부와 위치를 분석하여 스트라이크/볼을 자동으로 판정합니다.
*   **🚀 고정밀 투구 속도 측정**: 두 판정면 사이의 거리(20cm)와 고해상도 타이머(`time.perf_counter()`)를 사용하여 투구 속도(km/h)를 정밀하게 측정합니다.
*   **🎥 투구 영상 자동 녹화**: 각 투구 이벤트마다 사전 1.5초 + 사후 1초의 비디오 클립을 자동으로 저장하여 상세 분석이 가능합니다.
*   **📈 인터랙티브 웹 대시보드**: [Plotly Dash](https://dash.plotly.com/) 기반의 웹 대시보드를 통해 3D 투구 궤적, 투구 위치 분포, 통계 데이터(S/B/O 카운트, 구속 등), 그리고 녹화된 영상을 다각도로 분석할 수 있습니다.
*   **✋ MediaPipe 손 인식**: MediaPipe를 활용한 손 인식 기능이 통합되어 있습니다.
*   **📊 실시간 AR 전광판**: 스트라이크, 볼, 아웃 카운트를 AR 화면에 실시간으로 표시하여 실제 야구 경기와 같은 경험을 제공합니다.
*   **🎯 투구 충돌 지점 표시**: 모든 투구의 충돌 지점을 Plane2에 번호와 함께 영구적으로 표시하여 투구 패턴을 시각적으로 분석할 수 있습니다.

---

## 시스템 아키텍처 (System Architecture)

본 시스템은 여러 모듈이 유기적으로 연동하여 동작합니다. 전체적인 데이터 흐름과 모듈 간의 관계는 아래와 같습니다.

```mermaid
graph TD
    A[카메라 입력<br/>(Live Camera / Video)] --> B{main_v7.py<br/>(Core Logic)};

    subgraph "입력 처리 및 AR 설정"
        C[camera.py<br/>카메라 관리]
        D[aruco_detector.py<br/>ArUco 마커 검출/좌표계 설정]
    end

    subgraph "객체 추적 및 분석"
        E[tracker_v1.py<br/>공/손 감지<br/>BallDetector, HandDetector]
        F[kalman_filter.py<br/>KalmanFilter3D<br/>3D 궤적 안정화]
    end

    subgraph "시각화 및 출력"
        G[dashboard.py<br/>웹 대시보드<br/>Plotly Dash 서버]
        H[baseball_scoreboard.py<br/>AR 전광판<br/>S/B/O 카운트 표시]
        I[effects.py<br/>판정 텍스트 효과<br/>STRIKE/BALL 애니메이션]
    end
    
    J[config.py<br/>전역 설정<br/>스트라이크존 좌표/색상 등]

    B --> C & D & E & F & G & H & I & J;

    subgraph "처리 파이프라인"
        P1[1. 프레임 캡처 및 왜곡 보정] --> P2[2. ArUco 마커 검출<br/>(3D 좌표계 설정 - rvec, tvec)];
        P2 --> P3[3. 공 검출 및 3D 위치 변환<br/>(HSV 기반 검출 + 깊이 추정)];
        P3 --> P4[4. 칼만 필터 적용<br/>(궤적 보정 + 마할라노비스 게이팅)];
        P4 --> P5[5. 2단계 평면 교차 감지<br/>(Plane1 → Plane2)];
        P5 --> P6[6. 속도 계산 및 판정<br/>(보간 기반 정밀 타이밍)];
        P6 --> P7[7. 영상 클립 저장<br/>(사전/사후 버퍼 결합)];
    end

    B -- Orchestrates --> P1;

    subgraph "최종 결과"
        O1[AR 화면 출력<br/>(OpenCV - 궤적/존/스코어보드)]
        O2[데이터 분석 대시보드<br/>(Web Browser - 3D 시각화/영상)]
    end

    P7 --> O1;
    P7 --> O2;
    H --> O1;
    I --> O1;

```

---

## 기술적 상세 (Technical Details)

### 1. ArUco 마커 기반 좌표계 설정

- `aruco_detector.py` 모듈은 ArUco 마커를 검출하여 카메라와 마커 사이의 상대적인 3D 위치 및 회전(`rvecs`, `tvecs`)을 계산합니다. 이는 실제 공간에 안정적인 3D 좌표계를 설정하는 기준이 됩니다.
- 마커 좌표계: X(좌우), Y(깊이/앞뒤), Z(높이)

```python
# src/main_v7.py
# ArUco 마커 검출 및 포즈 추정
corners, ids, rejected = aruco_detector.detect_markers(analysis_frame)
if ids is not None:
    rvecs, tvecs = aruco_detector.estimate_pose(corners)
```

### 2. 공 추적 및 3D 위치 추정

현재 시스템은 다층 추적 파이프라인을 사용합니다:

- **(1) 2D 공 검출**: `tracker_v1.py`의 `BallDetector`는 HSV 색상 공간에서 정의된 범위 내의 객체를 찾아 공의 2D 위치(`center`, `radius`)를 식별합니다.
- **(2) 3D 깊이 추정**: 검출된 공의 픽셀 반지름과 실제 공의 반지름(0.036m), 카메라 초점 거리를 이용하여 3D 깊이(Z 좌표)를 추정합니다.
  ```python
  estimated_Z = (camera_matrix[0, 0] * BALL_RADIUS_REAL) / max(1e-6, radius)
  ```
- **(3) 카메라 좌표계 → 마커 좌표계 변환**: 2D 이미지 좌표와 추정된 깊이를 결합하여 카메라 좌표계 기준 3D 좌표를 계산하고, 이를 다시 ArUco 마커 기준 좌표계로 변환합니다.
- **(4) 3D 칼만 필터 적용**: `kalman_filter.py`의 `KalmanFilter3D` 클래스가 마할라노비스 거리 기반 게이팅(임계값 7.81)을 사용하여 이상치를 제거하고 공의 궤적을 부드럽게 보정합니다.

```python
# src/main_v7.py - 공 검출 및 3D 위치 계산
center, radius, _ = ball_detector.detect(analysis_frame)
if center and radius > 0.4:
    # 깊이 추정
    estimated_Z = (camera_matrix[0, 0] * BALL_RADIUS_REAL) / max(1e-6, radius)
    ball_3d_cam = np.array([
        (center[0] - camera_matrix[0,2]) * estimated_Z / camera_matrix[0,0],
        (center[1] - camera_matrix[1,2]) * estimated_Z / camera_matrix[1,1],
        estimated_Z
    ], dtype=np.float32)
    
    # 칼만 필터 업데이트 (카메라 좌표계)
    filtered_point_kalman = kalman_tracker.update_with_gating(ball_3d_cam)
    
    # 마커 좌표계로 변환
    filtered_point = aruco_detector.point_to_marker_coord(
        np.array(filtered_point_kalman, dtype=np.float32), rvec, tvec
    )
```

### 3. 2단계 스트라이크/볼 판정 로직

시스템은 정밀한 2단계 판정 로직을 사용합니다:

- **Plane1** (앞면, Y=0): 첫 번째 판정면
- **Plane2** (뒤면, Y=0.2m): 두 번째 판정면 (Plane1에서 20cm 뒤)
- **판정 방식**: 
  1. 공이 Plane1을 통과하면 `zone_step1 = True` 설정
  2. `zone_step1`이 True인 상태에서 공이 Plane2를 통과할 때, Plane2의 스트라이크 존 폴리곤 내부에 있으면 **스트라이크**, 외부에 있으면 **볼**로 판정
- **정밀 타이밍**: `time.perf_counter()`와 선형 보간을 사용하여 정확한 평면 교차 시점을 계산합니다.

```python
# src/main_v7.py - 2단계 판정 로직
# 서명 거리 계산 (법선 방향 정렬)
d1 = aruco_detector.signed_distance_to_plane_oriented(
    filtered_point, p1_0, p1_1, p1_2, desired_dir=desired_depth_axis
)
d2 = aruco_detector.signed_distance_to_plane_oriented(
    filtered_point, p2_0, p2_1, p2_2, desired_dir=desired_depth_axis
)

# 1단계: Plane1 통과 감지 (+ → 0 → −)
if prev_distance_to_plane1 is not None:
    crossed_p1 = (prev_distance_to_plane1 > 0.0) and (d1 <= 0.0)
    if (not zone_step1) and crossed_p1 and in_poly1:
        # 선형 보간으로 정확한 교차 시점 계산
        alpha1 = prev_distance_to_plane1 / (prev_distance_to_plane1 - d1 + 1e-9)
        t_cross_plane1 = prev_time_perf + alpha1 * (now_perf - prev_time_perf)
        zone_step1 = True

# 2단계: Plane2 통과 및 최종 판정
if prev_distance_to_plane2 is not None:
    crossed_p2 = (prev_distance_to_plane2 > 0.0) and (d2 <= 0.0)
    if crossed_p2 and zone_step1:
        # 속도 계산
        dt = max(1e-6, (t_cross_plane2 - t_cross_plane1))
        v_depth_mps = ZONE_DEPTH / dt  # 0.2m / dt
        v_kmh = v_depth_mps * 3.6
        
        # 스트라이크/볼 판정 (Plane2 폴리곤 기준)
        if in_poly2:
            scoreboard.add_strike()
            text_effect.add_strike_effect()
        else:
            scoreboard.add_ball()
            text_effect.add_ball_effect()
```

### 4. 투구 영상 자동 녹화

- **ClipRecorder** 클래스: 각 투구 이벤트 발생 시 자동으로 비디오 클립을 생성합니다.
- **사전 버퍼**: 1.5초 분량의 프레임을 순환 버퍼(`deque`)에 저장
- **사후 녹화**: 이벤트 발생 후 1초간 추가 녹화
- **비동기 저장**: 별도 스레드에서 MP4로 인코딩하여 메인 루프 성능에 영향 없음

```python
# src/main_v7.py - 클립 녹화
# 사전 버퍼 유지
prebuffer.append(overlay_frame.copy())

# 투구 이벤트 발생 시
ts = int(time.time())
clip_filename = f"pitch_{ts}.mp4"
clip_path = os.path.join(clips_dir, clip_filename)
clip_recorder.start(list(prebuffer), clip_path, cap_fps, post_seconds=1.0)
```

### 5. 실시간 웹 대시보드

- `dashboard.py`는 `Dash`와 `Plotly`를 사용하여 웹 기반 대시보드를 생성합니다.
- **주요 기능**:
  - 3D 투구 궤적 시각화 (Plotly 3D Scatter)
  - 2D 기록지: Plane2 기준 충돌 지점 표시 (X-Z 평면)
  - 투구 목록 테이블 (번호, 시간, 결과, 속도)
  - 선택한 투구의 영상 재생 (HTML5 Video)
  - 실시간 데이터 업데이트 (Dash Callback)

```python
# src/main_v7.py - 대시보드 데이터 업데이트
dashboard.update_data({
    'record_sheet_polygon': [[p[0], p[2]] for p in ball_zone_corners2],
    'append_pitch': {
        'result': result_label,
        'speed_kmh': float(v_kmh),
        'point_3d': [float(point_on_plane2[0]), float(point_on_plane2[1]), float(point_on_plane2[2])],
        'trajectory_3d': [list(map(float, pt)) for pt in trajectory_points],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'video_filename': clip_filename,
        'number': pitch_number
    }
})
```

---

## 사용 방법 (How to Use)

### 1. 요구사항 (Prerequisites)

- Python 3.9+
- 필요한 라이브러리:
  ```
  opencv-python>=4.5.0
  numpy>=1.20.0
  imutils
  mediapipe>=0.10.0
  dash>=2.0.0
  plotly>=5.0.0
  dash-table
  ```

### 2. 설치 (Installation)

1.  **프로젝트 클론**
    ```bash
    git clone https://github.com/YUjinEDU/Strikezone.git
    cd Strikezone
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
    pip install opencv-python numpy imutils mediapipe dash plotly dash-table
    ```

### 3. 설정 (Configuration)

`src/config.py` 파일에서 각종 파라미터를 수정할 수 있습니다:

- **색상 범위**: `GREEN_LOWER`, `GREEN_UPPER` - 추적할 공의 HSV 색상 범위
- **ArUco 마커**: `ARUCO_MARKER_SIZE` - 사용하는 ArUco 마커의 실제 크기 (미터 단위, 기본값: 0.16m)
- **캘리브레이션**: `CALIBRATION_PATH` - 카메라 캘리브레이션 파일 경로 (기본값: "camera_calib.npz")
- **스트라이크 존**: `STRIKE_ZONE_CORNERS` - 스트라이크 존의 3D 좌표 (마커 좌표계 기준)
- **판정면 간격**: `ZONE_DEPTH` - Plane1과 Plane2 사이의 거리 (기본값: 0.2m)
- **공 반지름**: `BALL_RADIUS_REAL` - 실제 공의 반지름 (기본값: 0.036m)

### 4. 실행 (Execution)

1.  `src` 디렉토리로 이동합니다.
    ```bash
    cd src
    ```
2.  메인 프로그램을 실행합니다.
    ```bash
    python main_v7.py
    ```
3.  실행 후 터미널에서 입력 소스를 선택합니다:
    - `1`: 카메라 사용 (GUI에서 카메라 선택 가능)
    - `2`: 비디오 파일 사용 (경로: `./video/video_BBS.mp4`)

4.  웹 대시보드 접속:
    - 브라우저에서 `http://127.0.0.1:8050` 접속
    - 실시간으로 투구 데이터 확인 및 영상 재생

**주요 단축키:**
*   `q`: 프로그램 종료
*   `r`: 카운트 및 상태 리셋 (스코어보드, 궤적, 기록 초기화)
*   `Space` (비디오 모드): 재생/일시정지

### 5. 출력 파일

시스템은 다음과 같은 파일들을 자동으로 생성합니다:

- **로그 파일**: `log/strike_log_YYYYMMDD_HHMMSS.txt` - 투구 이벤트 로그
- **영상 클립**: `clips/pitch_TIMESTAMP.mp4` - 각 투구의 영상 (사전 1.5초 + 사후 1초)

---

## 코드 구조 (Code Structure)

```
/src
├── main_v7.py              # 메인 실행 파일, 전체 워크플로우 제어
│                           # - ClipRecorder: 투구 영상 녹화
│                           # - 2단계 판정 로직 구현
│                           # - 사전/사후 버퍼 관리
├── config.py               # 시스템 전역 설정 및 상수 관리
│                           # - 스트라이크 존 좌표 (STRIKE_ZONE_CORNERS)
│                           # - 색상 범위 (GREEN_LOWER, GREEN_UPPER)
│                           # - 판정면 간격 (ZONE_DEPTH)
├── camera.py               # 카메라 캡처 및 캘리브레이션 관리
│                           # - CameraManager: 카메라 선택/열기/캘리브레이션
├── aruco_detector.py       # ArUco 마커 검출 및 3D 좌표 변환
│                           # - ArucoDetector: 마커 검출, 포즈 추정
│                           # - signed_distance_to_plane_oriented()
│                           # - point_to_marker_coord()
├── tracker_v1.py           # 공/손 검출 및 궤적 추적 로직
│                           # - BallDetector: HSV 기반 공 검출
│                           # - HandDetector: MediaPipe 손 인식
│                           # - KalmanTracker: 2D/3D 칼만 필터
├── kalman_filter.py        # 3D 칼만 필터 구현
│                           # - KalmanFilter3D: 마할라노비스 게이팅
│                           # - update_with_gating(): 이상치 제거
├── baseball_scoreboard.py  # AR 전광판 UI
│                           # - BaseballScoreboard: S/B/O 카운트 표시
│                           # - 3D 좌표 투영 및 렌더링
├── effects.py              # 시각 효과
│                           # - TextEffect: 'STRIKE', 'BALL' 애니메이션
│                           # - PlotlyVisualizer: 3D/2D 그래프 생성
├── dashboard.py            # Dash 기반 웹 대시보드 서버
│                           # - Dashboard: Plotly Dash 앱
│                           # - 3D 궤적, 2D 기록지, 투구 테이블
│                           # - 영상 재생 기능
└── video/                  # 테스트 비디오 파일
    └── video_BBS.mp4

/clips                      # 녹화된 투구 영상 (자동 생성)
    └── pitch_*.mp4

/log                        # 로그 파일 (자동 생성)
    └── strike_log_*.txt
```

### 레거시 파일 (참고용)
- `main_freeze.py`, `main_old.py`, `main_v4.py`, `main_v5.py`, `main_v6.py`: 이전 버전 백업
- `tracker_v1.1.py`, `tracker_v2.py`: 추적 알고리즘 실험 버전

---

## 향후 개선 방향 (Future Work)

상세한 개선 계획은 [TODO_Version2.md](TODO_Version2.md)를 참조하세요.

**주요 개선 항목:**
-   [ ] **색상/광학 개선**: 형광 마젠타 테이프, 무광 표면, 노출/화이트밸런스 고정, 균일 조명
-   [ ] **탐지 로직 강화**: HSV + rg/ratio 병렬 마스크, 원형도/면적 필터링, 히스토그램 백프로젝션
-   [ ] **마커 시스템 업그레이드**: AprilTag + 보드 + KLT 추적 + Health Check
-   [ ] **딥러닝 기반 객체 검출**: YOLO, SSD 등 딥러닝 모델 도입
-   [ ] **사용자 맞춤형 존 설정**: 사용자의 키나 자세에 맞춘 동적 스트라이크 존 조절


---

## 라이선스 (License)

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

---

## 기여 및 문의 (Contributing & Contact)

프로젝트에 기여하고 싶거나 문의 사항이 있으시면 이슈를 생성해 주세요.

**Repository**: [YUjinEDU/Strikezone](https://github.com/YUjinEDU/Strikezone)
