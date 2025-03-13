# AR 스트라이크 존 프로젝트

## 프로젝트 개요

본 프로젝트는 증강현실(AR) 기술을 활용하여 실시간으로 스트라이크 존을 시각화하고 투구 분석을 수행하는 시스템입니다. 카메라를 통해 공의 궤적을 추적하고, 스트라이크/볼 판정을 자동화하며, 투구 데이터를 실시간으로 분석하여 사용자에게 피드백을 제공합니다.

## 시스템 구성

![시스템 구성도](system_architecture.png)

### 주요 구성 요소

1. **카메라 시스템**: 공의 움직임을 실시간으로 캡처
2. **AR 마커 인식**: ArUco 마커를 통한 좌표계 설정
3. **공 추적 알고리즘**: 컬러 기반 객체 검출 및 칼만 필터 적용
4. **3D 좌표 변환**: 2D 이미지에서 3D 공간으로의 좌표 변환
5. **스트라이크/볼 판정**: 정의된 존 영역 내 공의 위치 판정
6. **대시보드 시각화**: 투구 데이터 실시간 시각화 및 분석

## 기술적 특징

### 1. ArUco 마커 기반 좌표계 설정

- ArUco 마커를 통해 실제 공간에 가상의 좌표계 설정
- 마커 검출 및 포즈 추정을 통한 정확한 3D 공간 매핑
- 카메라 움직임에도 안정적인 좌표계 유지

```python
# ArUco 마커 검출 및 포즈 추정
corners, ids, rejected = aruco_detector.detect_markers(analysis_frame)
if ids is not None:
    rvecs, tvecs = aruco_detector.estimate_pose(corners)
```

### 2. 공 추적 및 3D 위치 추정

- HSV 색 공간에서의 컬러 기반 객체 검출
- 공의 크기를 이용한 깊이(Z축) 추정
- 칼만 필터를 통한 노이즈 제거 및 궤적 안정화

```python
# 공 검출 및 3D 위치 추정
center, radius, _ = ball_detector.detect(analysis_frame)
if center and radius > 0.5:
    estimated_Z = (camera_matrix[0, 0] * BALL_RADIUS_REAL) / radius
    ball_3d_cam = np.array([
        (center[0] - camera_matrix[0,2]) * estimated_Z / camera_matrix[0,0],
        (center[1] - camera_matrix[1,2]) * estimated_Z / camera_matrix[1,1],
        estimated_Z
    ])
    
    # 카메라 → 마커 좌표계 변환
    point_in_marker_coord = aruco_detector.point_to_marker_coord(ball_3d_cam, rvec, tvec)
    
    # 칼만 필터 적용
    filtered_point = kalman_tracker.update_with_gating(np.array(point_in_marker_coord, dtype=np.float32))
```

### 3. 구속 측정 알고리즘

- 3D 공간에서의 위치 변화와 시간을 이용한 속도 계산
- 이상치 필터링을 통한 정확도 향상
- 투구 순간의 최고 속도 기록

```python
# 투구 속도 계산
if len(ball_positions_history) >= 3:
    first_pos = ball_positions_history[0]
    last_pos = ball_positions_history[-1]
    time_diff = ball_times_history[-1] - ball_times_history[0]
    
    if time_diff > 0:
        # 3D 거리 계산
        distance = np.linalg.norm(last_pos - first_pos)
        
        # 속도 계산 (m/s)
        velocity = distance / time_diff
        
        # 이상치 필터링
        if 5 < velocity < 50:  # 18km/h ~ 180km/h 범위
            velocity_kmh = velocity * 3.6
            velocity_buffer.append(velocity_kmh)
```

### 4. 스트라이크/볼 판정 시스템

- 3D 공간에서 정의된 스트라이크 존과 볼 존
- 평면 방정식을 이용한 공의 위치 판정
- 2단계 판정 로직으로 정확도 향상

```python
# 판정 평면 좌표
p_0 = ball_zone_corners[0]
p_1 = ball_zone_corners[1]
p_2 = ball_zone_corners[3]

p2_0 = ball_zone_corners2[0]
p2_1 = ball_zone_corners2[1]
p2_2 = ball_zone_corners2[3]

# 평면과의 거리 및 다각형 내부 여부 계산
distance_to_plane1 = aruco_detector.signed_distance_to_plane(filtered_point, p_0, p_1, p_2)
is_in_polygon1 = aruco_detector.is_point_in_polygon(center, projected_points)

distance_to_plane2 = aruco_detector.signed_distance_to_plane(filtered_point, p2_0, p2_1, p2_2)
is_in_polygon2 = aruco_detector.is_point_in_polygon(center, projected_points2)

# 1단계 판정
if -0.1 <= distance_to_plane1 <= 0.0 and is_in_polygon1:
    zone_step1 = True

# 2단계 판정
if zone_step1:
    if distance_to_plane2 <= 0.0 and is_in_polygon2:
        zone_step2 = True
        # 스트라이크 판정
```

### 5. 실시간 대시보드 시각화

- Dash 프레임워크를 이용한 웹 기반 대시보드
- 3D 궤적 시각화 및 투구 위치 기록
- 투구 통계 및 추천 시스템

```python
# 대시보드 데이터 업데이트
dashboard_data = {
    'record_sheet_points': list(zip(record_sheet_x, record_sheet_y)),
    'record_sheet_polygon': [[p[0], p[2]] for p in ball_zone_corners2],
    'trajectory_3d': list(zip(pitch_points_trace_3d_x, pitch_points_trace_3d_y, pitch_points_trace_3d_z)),
    'strike_zone_corners_3d': ball_zone_corners.tolist(),
    'ball_zone_corners_3d': ball_zone_corners.tolist(),
    'ball_zone_corners2_3d': ball_zone_corners2.tolist(),
    'box_corners_3d': box_corners_3d.tolist(),
    'pitch_count': len(detected_strike_points) + len(detected_ball_points),
    'strike_count': len(detected_strike_points),
    'ball_count': len(detected_ball_points),
    'pitch_speeds': pitch_speeds,
    'pitch_results': pitch_results,
    'pitch_history': pitch_history
}
dashboard.update_data(dashboard_data)
```

## 시스템 아키텍처

### 모듈 구성

1. **main.py**: 메인 실행 파일, 전체 시스템 통합 및 제어
2. **config.py**: 시스템 설정 및 파라미터 관리
3. **camera.py**: 카메라 관리 및 캘리브레이션
4. **aruco_detector.py**: ArUco 마커 검출 및 좌표 변환
5. **tracker.py**: 공 검출 및 추적 알고리즘
6. **kalman_filter.py**: 칼만 필터 구현
7. **effects.py**: 시각 효과 및 UI 요소
8. **dashboard.py**: 데이터 시각화 및 분석 대시보드

### 클래스 다이어그램

![클래스 다이어그램](class_diagram.png)

## 주요 기능 및 UI

### 1. 실시간 AR 시각화

- 스트라이크 존 및 볼 존 시각화
- 3D 박스 및 그리드 표시
- 공 궤적 추적 및 표시

### 2. 투구 분석 대시보드

- 3D 공 궤적 시각화
- 투구 위치 기록 (중심점 기준)
- 투구 통계 (총 투구수, 스트라이크 수, 볼 수, 평균 속도)
- 투구 추천 시스템 (패턴 분석 기반)
- 투구 기록 표 (번호, 결과, 속도)

### 3. 구속 측정 및 분석

- 실시간 구속 측정 및 표시
- 이상치 필터링을 통한 정확도 향상
- 투구별 최고 속도 기록

## 기술적 도전과 해결 방법

### 1. 3D 좌표 변환 정확도

**도전**: 2D 이미지에서 3D 공간으로의 정확한 좌표 변환

**해결 방법**:
- 카메라 캘리브레이션을 통한 내부 파라미터 정확도 향상
- ArUco 마커를 기준으로 한 안정적인 좌표계 설정
- 공의 크기를 이용한 깊이 추정 알고리즘 개선

### 2. 공 추적 안정성

**도전**: 빠르게 움직이는 공의 안정적인 추적

**해결 방법**:
- HSV 색 공간에서의 컬러 기반 객체 검출 최적화
- 칼만 필터를 통한 노이즈 제거 및 궤적 안정화
- 게이팅 기법을 적용한 이상치 필터링

### 3. 구속 측정 정확도

**도전**: 제한된 프레임 레이트에서의 정확한 구속 측정

**해결 방법**:
- 다중 측정점을 이용한 속도 계산
- 이상치 필터링 및 평균화 기법 적용
- 물리적으로 가능한 속도 범위 설정 (5~50 m/s)

### 4. 실시간 처리 성능

**도전**: 영상 처리, 3D 변환, 대시보드 업데이트의 실시간 처리

**해결 방법**:
- 멀티스레딩을 통한 병렬 처리
- 프레임 스킵을 통한 처리 부하 조절
- 대시보드 업데이트 주기 최적화

## 향후 개선 방향

1. **다중 카메라 시스템**: 여러 각도에서의 촬영을 통한 3D 위치 추정 정확도 향상
2. **딥러닝 기반 공 검출**: 다양한 환경에서의 공 검출 정확도 향상
3. **투구 패턴 분석 고도화**: 머신러닝을 활용한 투구 패턴 분석 및 예측
4. **모바일 앱 연동**: 스마트폰 앱을 통한 접근성 향상
5. **사용자 맞춤형 존 설정**: 사용자의 신체 조건에 맞는 스트라이크 존 자동 조정

## 결론

AR 스트라이크 존 프로젝트는 컴퓨터 비전, 증강현실, 데이터 분석 기술을 통합하여 야구 투구 훈련을 위한 혁신적인 솔루션을 제공합니다. 실시간 스트라이크/볼 판정, 구속 측정, 투구 패턴 분석을 통해 사용자는 즉각적인 피드백을 받을 수 있으며, 이를 통해 투구 기술 향상에 도움을 줄 수 있습니다.

본 프로젝트는 스포츠 과학과 기술의 융합을 통해 훈련 방법의 혁신을 이끌어내는 좋은 사례가 될 것이며, 향후 다양한 스포츠 분야로의 확장 가능성을 가지고 있습니다.

## 참고 문헌

1. OpenCV 공식 문서: https://docs.opencv.org/
2. ArUco 마커 라이브러리: https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html
3. Dash 프레임워크: https://dash.plotly.com/
4. 칼만 필터 알고리즘: https://en.wikipedia.org/wiki/Kalman_filter
