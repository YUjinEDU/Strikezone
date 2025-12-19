# AR Strike Zone: 증강현실 야구 투구 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

**AR Strike Zone**은 단일 카메라를 사용하여 실시간으로 투구 궤적을 추적하고, 스트라이크 존을 증강현실(AR)로 시각화하며, 투구 데이터를 심층 분석하는 시스템입니다.

![시스템 실행 화면](src_v1/video/GIF/스트라이크1.gif)

---

## 📌 주요 기능

1.  **실시간 AR 스트라이크 존**: ArUco 마커를 기준으로 3D 공간에 스트라이크/볼 존 시각화.
2.  **공 추적 및 구속 측정**: HSV 기반 공 검출과 칼만 필터를 이용한 정밀 추적 및 구속(km/h) 계산.
3.  **자동 판정**: 공의 3D 궤적을 분석하여 스트라이크/볼 자동 판정.
4.  **AR 전광판**: 실시간 볼카운트(B/S/O) 디스플레이.

<div align="center">
  <img src="src_v1/video/GIF/볼2.gif" width="45%">
  <img src="src_v1/video/GIF/아웃1.gif" width="45%">
</div>

---

## 🏗️ 시스템 아키텍처

```mermaid
graph TD
    Input[카메라 입력] --> Main{Main System}

    subgraph "Core Modules"
        Main --> Detector[Ball Detector<br/>(HSV + Contour)]
        Main --> Aruco[ArUco Detector<br/>(Pose Estimation)]
        Main --> Kalman[Kalman Filter<br/>(Trajectory Smoothing)]
    end

    subgraph "Visualization"
        Main --> AR[AR Overlay<br/>(Strike Zone & HUD)]
        Main --> Score[Scoreboard<br/>(B/S/O Count)]
    end

    Aruco -- "3D 좌표계 기준 제공" --> Main
    Detector -- "2D 공 위치" --> Main
    Main -- "3D 위치 변환" --> Kalman
    Kalman -- "보정된 3D 좌표" --> AR
```

---

## 🛠️ 기술적 원리

### 1. 좌표계 설정 (ArUco Marker)

단일 카메라 환경에서 깊이(Depth)를 추정하기 위해 ArUco 마커를 사용합니다. 마커의 실제 크기를 바탕으로 카메라와 현실 공간 사이의 **3D Pose(Rotation, Translation)**를 계산하여 기준 좌표계를 생성합니다.

### 2. 공 검출 및 거리 추정 (Pinhole Model)

- **검출**: HSV 색상 공간에서 공의 색상을 추출하여 위치를 찾습니다.
- **거리 계산**: 핀홀 카메라 모델을 응용하여, 이미지 상의 공 크기(radius)와 실제 공 크기의 비율을 통해 카메라로부터의 거리(Z축)를 추정합니다.

### 3. 궤적 보정 (Kalman Filter)

카메라 노이즈와 검출 오차를 줄이기 위해 **칼만 필터(Kalman Filter)**를 적용합니다. 이를 통해 공이 잠시 가려지거나 검출이 튀는 현상을 보정하고 부드러운 궤적을 그립니다.

---

## 🚀 실행 방법

### 요구 사항

- Python 3.9 이상
- OpenCV, NumPy, Imutils

### 설치 및 실행

1. 저장소를 클론합니다.
2. `src_v1` 폴더로 이동합니다.
   ```bash
   cd src_v1
   ```
3. 필요한 라이브러리를 설치합니다.
   ```bash
   pip install opencv-python numpy imutils
   ```
4. 메인 프로그램을 실행합니다.
   ```bash
   python main_v7.py
   ```

---

## 📂 파일 구조 (src_v1)

- `main_v7.py`: 프로그램 메인 실행 파일.
- `aruco_detector.py`: 마커 검출 및 좌표계 변환 모듈.
- `tracker_v1.py`: 공 색상 검출 모듈.
- `kalman_filter.py`: 궤적 보정 필터.
- `baseball_scoreboard.py`: AR 전광판 기능.
- `camera_calib.npz`: 카메라 캘리브레이션 데이터.

---

_본 프로젝트는 2025 오픈소스프로그래밍 과제로 제출되었습니다._
