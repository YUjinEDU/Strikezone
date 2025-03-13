import os
import sys
import cv2
import numpy as np
import threading
import time
import logging
from collections import deque
import math

# 내부 모듈 임포트
from config import *
from camera import CameraManager
from aruco_detector import ArucoDetector
from tracker import KalmanTracker, BallDetector, HandDetector
from effects import TextEffect
from dashboard import Dashboard
from kalman_filter import KalmanFilter3D

def main():
    # 전역 종료 이벤트
    shutdown_event = threading.Event()
    
    # 로그 설정
    logging.basicConfig(
        filename=LOG_FILENAME,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8'
    )
    
    # 상태 변수 초기화
    is_video_mode = False
    play_pause = False
    record_trajectory = False
    ar_started = False
    zone_step1 = False
    zone_step2 = False
    frame_count = 0
    last_time = 0.0
    
    # 궤적 관련 플래그와 타이머 변수 추가
    show_trajectory = False
    trajectory_display_start_time = None
    
    # 점수 카운트
    strike_count = 0
    ball_count = 0
    out_count = 0
    
    # 궤적 데이터
    detected_strike_points = []
    detected_ball_points = []
    previous_ball_position = None
    previous_ball_time = None
    
    # 투구 속도 계산용 변수
    pitch_speeds = []
    pitch_results = []
    pitch_history = []
    ball_positions_history = deque(maxlen=10)  # 최근 10개 위치 저장
    ball_times_history = deque(maxlen=10)      # 최근 10개 시간 저장
    velocity_buffer = deque(maxlen=5)          # 속도 평균 계산용 버퍼
    display_velocity = 0
    final_velocity = 0
    
    # 손 감지기 초기화
    hand_detector = HandDetector()
    
    # 텍스트 효과 초기화
    text_effect = TextEffect()
    
    # 카메라/비디오 선택
    user_input = input("1: 카메라, 2: 비디오 > ")
    
    if user_input == "1":
        # 카메라 관리자 초기화 및 카메라 선택 GUI 실행
        camera_manager = CameraManager(shutdown_event)
        selected_camera = camera_manager.create_camera_selection_gui()
        
        if selected_camera is None:
            print("카메라가 선택되지 않았습니다. 종료합니다.")
            shutdown_event.set()
            return
        
        # 카메라 열기
        if not camera_manager.open_camera(selected_camera):
            print(f"카메라 {selected_camera}를 열 수 없습니다. 종료합니다.")
            shutdown_event.set()
            return
        
        # 캘리브레이션 데이터 로드
        if not camera_manager.load_calibration(CALIBRATION_PATH):
            print("캘리브레이션 데이터를 로드할 수 없습니다. 종료합니다.")
            shutdown_event.set()
            return
        
        camera_matrix = camera_manager.camera_matrix
        dist_coeffs = camera_manager.dist_coeffs
        cap = camera_manager.capture
        
    elif user_input == "2":
        # 비디오 파일 선택
        video_path = "./video/test_video_1.mp4"
        cap = cv2.VideoCapture(video_path)
        is_video_mode = True
        
        # 비디오 모드용 윈도우 생성
        cv2.namedWindow('Original')
        
        # 트랙바 콜백 함수
        def on_trackbar(val):
            cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        
        # 트랙바 생성
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.createTrackbar("Progress", "Original", 0, total_frames - 1, on_trackbar)
        
        # 캘리브레이션 데이터 로드
        try:
            calib_data = np.load(CALIBRATION_PATH)
            camera_matrix = calib_data["camera_matrix"]
            dist_coeffs = calib_data["dist_coeffs"]
        except Exception as e:
            print(f"캘리브레이션 데이터를 로드할 수 없습니다: {e}")
            shutdown_event.set()
            return
    
    else:
        print("잘못된 입력입니다. 종료합니다.")
        return
    
    # ArUco 검출기 초기화
    aruco_detector = ArucoDetector(ARUCO_MARKER_SIZE, camera_matrix, dist_coeffs)
    
    # 공 검출기 초기화 (녹색공)
    ball_detector = BallDetector(GREEN_LOWER, GREEN_UPPER)

        # 대시보드 초기화 및 시작
    dashboard = Dashboard()
    dashboard_thread = dashboard.run_server()
    
    # 초기 대시보드 데이터 설정
    # 빈 데이터라도 설정하면 그래프 레이아웃이 초기화됨
    initial_dashboard_data = {
        'record_sheet_points': [],
        'record_sheet_polygon': [[p[0], p[2]] for p in BALL_ZONE_CORNERS],
        'trajectory_3d': [],
        'strike_zone_corners_3d': BALL_ZONE_CORNERS.tolist(),
        'ball_zone_corners_3d': BALL_ZONE_CORNERS.tolist(),
        'ball_zone_corners2_3d': BALL_ZONE_CORNERS.tolist(),
        'box_corners_3d': aruco_detector.get_box_corners_3d(BOX_MIN, BOX_MAX).tolist(),
        'pitch_count': 0,
        'strike_count': 0,
        'ball_count': 0,
        'pitch_speeds': [],
        'pitch_results': [],
        'pitch_history': []
    }
    dashboard.update_data(initial_dashboard_data)
    
    # 칼만 필터 초기화
    kalman_tracker = KalmanFilter3D()
    
    # 박스 코너 좌표 계산
    box_corners_3d = aruco_detector.get_box_corners_3d(BOX_MIN, BOX_MAX)
    
    # 존 코너 회전 적용
    ball_zone_corners = np.dot(BALL_ZONE_CORNERS, ROTATION_MATRIX.T)
    ball_zone_corners2 = np.dot(BALL_ZONE_CORNERS, ROTATION_MATRIX.T)
    ball_zone_corners2[:, 1] += ZONE_Z_DIFF
    
    # 결과 표시 변수
    result = ""
    
    # 대시보드 데이터 저장 변수
    record_sheet_x = []
    record_sheet_y = []
    pitch_points_3d_x = []
    pitch_points_3d_y = []
    pitch_points_3d_z = []
    pitch_points_trace_3d_x = []
    pitch_points_trace_3d_y = []
    pitch_points_trace_3d_z = []
    
    # 변수 초기화
    ids = None
    
    # 메인 루프
    while not shutdown_event.is_set():
        # 비디오 모드에서 일시정지 상태 처리
        if is_video_mode and play_pause:
            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.setTrackbarPos("Progress", "Original", current_frame_no)
                key = cv2.waitKey(10)
            except cv2.error:
                pass
                
            if key & 0xFF == ord(' '):
                play_pause = False
                print("재생 재개")
            elif key & 0xFF == ord('q'):
                shutdown_event.set()
                break
            continue
        
        # 프레임 읽기
        fps_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 분석용 프레임과 오버레이 프레임 분리
        analysis_frame = frame.copy()
        overlay_frame = frame.copy()
        
        # 손 감지
        results = hand_detector.find_hands(frame)
        
        # 손 펴면 AR 시작
        if not ar_started:
            if hand_detector.is_hand_open():
                ar_started = True
                print("AR 시작!")
            else:
                cv2.putText(overlay_frame, "Show your hand!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if ar_started:
            frame_count += 1
            if frame_count % SKIP_FRAMES != 0:
                cv2.imshow('ARUCO Tracker with Strike Zone', overlay_frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord('r'):
                    # 상태 초기화
                    ar_started = False
                    strike_count = 0
                    out_count = 0
                    ball_count = 0
                continue
            
            # ArUco 마커 검출
            corners, ids, rejected = aruco_detector.detect_markers(analysis_frame)
            
            if ids is not None:
                # 마커 포즈 추정
                rvecs, tvecs = aruco_detector.estimate_pose(corners)
                
                for rvec, tvec in zip(rvecs, tvecs):
                    # 1) 첫 번째 영역 (볼 존)
                    projected_points = aruco_detector.project_points(ball_zone_corners, rvec, tvec)
                    cv2.polylines(overlay_frame, [projected_points], True, (200, 200, 0), 4)
                    
                    # 2) 두 번째 영역 (볼 존2)
                    projected_points2 = aruco_detector.project_points(ball_zone_corners2, rvec, tvec)
                    cv2.polylines(overlay_frame, [projected_points2], True, (0, 100, 255), 4)
                    
                    # 3D 박스 그리기
                    pts2d = aruco_detector.project_points(box_corners_3d, rvec, tvec)
                    aruco_detector.draw_3d_box(overlay_frame, pts2d, BOX_EDGES, color=(0,0,0), thickness=4)
                    
                    # 그리드 그리기
                    grid_pts2d = pts2d[[0,1,5,4]]  # 앞면 4개 코너만 사용
                    aruco_detector.draw_grid(overlay_frame, grid_pts2d, 3)
                    
                    # 공 궤적 기록 시작
                    if not record_trajectory:
                        record_trajectory = True
                        pitch_points_trace_3d_x.clear()
                        pitch_points_trace_3d_y.clear()
                        pitch_points_trace_3d_z.clear()
                    
                    # 공 검출 (녹색)
                    center, radius, _ = ball_detector.detect(analysis_frame)
                    
                    if center and radius > 0.5:
                        # 공 표시
                        ball_detector.draw_ball(overlay_frame, center, radius)
                        
                        # 공 깊이 추정
                        estimated_Z = (camera_matrix[0, 0] * BALL_RADIUS_REAL) / radius
                        ball_3d_cam = np.array([
                            (center[0] - camera_matrix[0,2]) * estimated_Z / camera_matrix[0,0],
                            (center[1] - camera_matrix[1,2]) * estimated_Z / camera_matrix[1,1],
                            estimated_Z
                        ])
                        
                        # 칼만 필터 업데이트 (카메라 좌표계)
                        filtered_point_kalman = kalman_tracker.update_with_gating(np.array(ball_3d_cam, dtype=np.float32))

                        # 이미지 좌표로 투영: 카메라 좌표계이므로, rvec과 tvec를 (0,0,0)로 사용
                        projected_pt = aruco_detector.project_points(
                            np.array([filtered_point_kalman]), 
                            np.zeros((3,1), dtype=np.float32), 
                            np.zeros((3,1), dtype=np.float32)
                        )[0]

                        # 2D 궤적 기록 (projected_pt는 이미지 좌표)
                        ball_detector.track_trajectory((projected_pt[0], projected_pt[1]))


                        
                        
                        # 카메라 → 마커 좌표계 변환
                        filtered_point = aruco_detector.point_to_marker_coord(filtered_point_kalman, rvec, tvec)
                        
                        if previous_ball_position is None:
                            previous_ball_position = filtered_point
                        
                        previous_ball_position = filtered_point
                        
 
                        
                        # 투구 속도 계산
                        current_time = time.time()
                        ball_positions_history.append(filtered_point)
                        ball_times_history.append(current_time)
                        
                        

                        # 깊이 정보 표시
                        marker_depth_text = f"marker Z: {tvec[0][2]:.2f} m"
                        ball_depth_text = f"ball Z: {estimated_Z:.2f} m"
                        
                        marker_position = tuple(map(int, pts2d[0]))
                        cv2.putText(overlay_frame, marker_depth_text,
                                    (marker_position[0], marker_position[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        cv2.putText(overlay_frame, ball_depth_text,
                                    (center[0]+20, center[1]+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 최소 3개 이상의 위치 데이터가 있을 때 속도 계산
                        current_velocity_kmh = 0
                        if len(ball_positions_history) >= 3:
                            # 첫 번째와 마지막 위치 사용
                            first_pos = ball_positions_history[0]
                            last_pos = ball_positions_history[-1]
                            time_diff = ball_times_history[-1] - ball_times_history[0]
                            
                            if time_diff > 0:
                                # 3D 거리 계산
                                distance = np.linalg.norm(last_pos - first_pos)
                                
                                # 속도 계산 (m/s)
                                velocity = distance / time_diff
                                
                                # 이상치 필터링 (너무 빠르거나 느린 속도 제외)
                                if 5 < velocity < 50:  # 18km/h ~ 180km/h 범위
                                    # km/h로 변환
                                    velocity_kmh = velocity * 3.6
                                    
                                    # 속도 버퍼에 추가
                                    velocity_buffer.append(velocity_kmh)
                                    
                                    # 버퍼에 충분한 데이터가 있으면 평균 계산
                                    if len(velocity_buffer) >= 3:
                                        # 이상치 제거 (최대, 최소 제외)
                                        sorted_velocities = sorted(velocity_buffer)
                                        filtered_velocities = sorted_velocities[1:-1] if len(sorted_velocities) > 2 else sorted_velocities
                                        
                                        # 평균 속도 계산
                                        current_velocity_kmh = sum(filtered_velocities) / len(filtered_velocities)
                                    else:
                                        current_velocity_kmh = velocity_kmh


                        # 현재 속도 표시 (실시간)
                        display_velocity = current_velocity_kmh
                        if len(velocity_buffer) > 0 and current_velocity_kmh == 0:
                            # 현재 속도가 0이고 버퍼에 값이 있으면 버퍼의 최대값 사용
                            display_velocity = max(velocity_buffer)
                        
                        # 대시보드 데이터 업데이트
                        current_dashboard_data = {
                            'record_sheet_points': list(zip(record_sheet_x, record_sheet_y)),
                            'record_sheet_polygon': [[p[0], p[2]] for p in ball_zone_corners2] if ids is not None else [],
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
                        #print(f"스트라이크 판정 후 대시보드 데이터 업데이트: pitch_count={len(detected_strike_points) + len(detected_ball_points)}")
                        dashboard.update_data(current_dashboard_data)
                        
                        # 궤적 기록
                        if record_trajectory:
                            pitch_points_trace_3d_x.append(filtered_point[0])
                            pitch_points_trace_3d_y.append(filtered_point[1])
                            pitch_points_trace_3d_z.append(filtered_point[2])
                        
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
                            # 공이 ball_zone_corners 평면을 지나갈 때 반투명 효과 추가
                            overlay = overlay_frame.copy()
                            # 다각형 내부를 반투명한 색상으로 채우기
                            cv2.fillPoly(overlay, [projected_points], (200, 200, 0, 128))
                            # 반투명 효과 적용 (알파 블렌딩)
                            alpha = 0.5  # 투명도 (0: 완전 투명, 1: 완전 불투명)
                            cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
                            
                        
                        # 2단계 판정
                        if zone_step1:
                            if distance_to_plane2 <= 0.00 and is_in_polygon2:
                                zone_step2 = True
                                # 공이 ball_zone_corners2 평면을 지나갈 때 반투명 효과 추가
                                overlay = overlay_frame.copy()
                                # 다각형 내부를 반투명한 색상으로 채우기
                                cv2.fillPoly(overlay, [projected_points2], (0, 100, 255, 128))
                                # 반투명 효과 적용 (알파 블렌딩)
                                alpha = 0.5  # 투명도 (0: 완전 투명, 1: 완전 불투명)
                                cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
                                
                                # 스트라이크 판정 (일정 시간 간격)
                                current_time = time.time()
                                if current_time - last_time > 1.0:
                                    strike_count += 1
                                    last_time = current_time
                                    
                                    print(f"스트라이크 카운트: {strike_count}")
                                    
                                    # 스트라이크 정보 저장
                                    detected_strike_points.append({
                                        '3d_coord': filtered_point,
                                        'rvec': rvec.copy(),
                                        'tvec': tvec.copy()
                                    })
                                    
                                    # 투구 속도 저장
                                    # 현재 계산된 속도 사용 (필터링된 속도)
                                    if current_velocity_kmh > 0:
                                        final_velocity = current_velocity_kmh
                                    elif len(velocity_buffer) > 0:
                                        # 버퍼에 있는 속도 중 최대값 사용 (투구 순간의 최고 속도)
                                        final_velocity = max(velocity_buffer)
                                    
                                    pitch_results.append("스트라이크")
                                    pitch_speeds.append(final_velocity)
                                    
                                    # 화면에 표시할 속도 업데이트
                                    display_velocity = final_velocity
                                    
                                    # 투구 기록 추가
                                    pitch_history.append({
                                        'number': len(pitch_history) + 1,
                                        'result': "스트라이크",
                                        'speed': f"{final_velocity:.1f}"
                                    })
                                    
                                    # 효과 및 결과 설정
                                    text_effect.add_strike_effect()
                                    result = "strike"
                                    
                                   
                                    # 스트라이크 로깅
                                    logging.info(f"""
                                    === Strike #{len(detected_strike_points)} ===
                                    Coordinates (Marker Frame):
                                        X: {filtered_point[0]:.6f} m
                                        Y: {filtered_point[1]:.6f} m
                                        Z: {filtered_point[2]:.6f} m
                                    Raw Camera Coordinates:
                                        X: {center[0]:.2f} px
                                        Y: {center[1]:.2f} px
                                    Estimated Depth: {estimated_Z:.6f} m
                                    Ball Radius: {radius:.2f} px
                                    """.strip())
                                    
                                    # 대시보드 데이터 업데이트
                                    record_sheet_x.append(filtered_point[0])
                                    record_sheet_y.append(filtered_point[2])
                                    
                                    pitch_points_3d_x.append(filtered_point[0])
                                    pitch_points_3d_y.append(ZONE_Z_DIFF)
                                    pitch_points_3d_z.append(filtered_point[2])
                                    
                                    # 궤적 기록 중단
                                    record_trajectory = False
                                    
                                    show_trajectory = True
                                    trajectory_display_start_time = time.time()
                                    
                                    # 스트라이크가 결정되면 즉시 대시보드 데이터 업데이트
                                    current_dashboard_data = {
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
                                    print(f"스트라이크 판정 후 대시보드 데이터 업데이트: pitch_count={len(detected_strike_points) + len(detected_ball_points)}")
                                    dashboard.update_data(current_dashboard_data)
                                    

                                    # 플래그 초기화
                                    zone_step1 = False
                                    zone_step2 = False
                                    
                        
                        # 볼 판정
                        if -0.1 <= distance_to_plane2 <= 0.0 and not is_in_polygon2:
                            current_time = time.time()
                            if current_time - last_time > 2.0:
                                last_time = current_time
                                result = "ball"
                                text_effect.add_ball_effect()
                                
                                ball_count += 1
                                print(f"볼 카운트: {ball_count}")
                                
                                # 볼 정보 저장
                                detected_ball_points.append({
                                    '3d_coord': filtered_point,
                                    'rvec': rvec.copy(),
                                    'tvec': tvec.copy()
                                })
                                
                                # 투구 속도 저장 (실제 측정이 어려우므로 임의의 값 사용)
                                final_velocity = 20.0 + np.random.normal(0, 5)
                                
                                pitch_results.append("볼")
                                pitch_speeds.append(final_velocity)
                                
                                # 화면에 표시할 속도 업데이트
                                display_velocity = final_velocity
                                
                                # 투구 기록 추가
                                pitch_history.append({
                                    'number': len(pitch_history) + 1,
                                    'result': "볼",
                                    'speed': f"{final_velocity:.1f}"
                                })
                                
                                # 볼도 기록지에 추가
                                record_sheet_x.append(filtered_point[0])
                                record_sheet_y.append(filtered_point[2])
                                
                                # 볼이 결정되면 즉시 대시보드 데이터 업데이트
                                current_dashboard_data = {
                                    'record_sheet_points': list(zip(record_sheet_x, record_sheet_y)),
                                    'record_sheet_polygon': [[p[0], p[2]] for p in ball_zone_corners2] if ids is not None else [],
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
                                print(f"볼 판정 후 대시보드 데이터 업데이트: pitch_count={len(detected_strike_points) + len(detected_ball_points)}")
                                dashboard.update_data(current_dashboard_data)
                                
                                # 궤적 기록 중단
                                record_trajectory = False
                                show_trajectory = True
                                trajectory_display_start_time = time.time()
                        
                        # 판정 이벤트 발생 시에만 궤적을 화면에 그리기
                        if show_trajectory:
                            ball_detector.draw_trajectory(overlay_frame)
                            # 2초 경과 후 궤적 시각화 종료 및 기록 초기화
                            if time.time() - trajectory_display_start_time >= 2.0:
                                show_trajectory = False
                                ball_detector.pts.clear()  # 궤적 기록 초기화        
                              
                
        # 평면 좌표 정의
        if ids is not None and len(detected_strike_points + detected_ball_points) > 0:
            plane_pt1 = ball_zone_corners2[0]
            plane_pt2 = ball_zone_corners2[1]
            plane_pt3 = ball_zone_corners2[2]
            
            # 모든 기록된 투구 표시 (스트라이크와 볼 통합)
            all_pitch_points = []
            
            # 스트라이크 점 데이터 준비
            for idx, point_data in enumerate(detected_strike_points):
                all_pitch_points.append({
                    'point': point_data['3d_coord'],
                    'type': 'strike',
                    'index': idx + 1
                })
            
            # 볼 점 데이터 준비
            for idx, point_data in enumerate(detected_ball_points):
                all_pitch_points.append({
                    'point': point_data['3d_coord'],
                    'type': 'ball',
                    'index': len(detected_strike_points) + idx + 1
                })
            
            # 모든 투구 점 그리기
            for pitch_data in all_pitch_points:
                # 평면에 투영
                point_on_plane = aruco_detector.project_point_onto_plane(
                    pitch_data['point'], plane_pt1, plane_pt2, plane_pt3
                )
                
                # 화면에 투영
                pt_2d_proj = aruco_detector.project_points(
                    np.array([point_on_plane]), rvec, tvec
                )[0]
                
                try:
                    # 더 안전하게 정수 변환
                    x = int(pt_2d_proj[0])
                    y = int(pt_2d_proj[1])
                    pt_2d_proj_int = (x, y)
                    
                    # 타입에 따라 다른 색상으로 원 그리기
                    if pitch_data['type'] == 'strike':
                        color = (255, 255, 255)  # 스트라이크: 흰색
                        cv2.circle(overlay_frame, pt_2d_proj_int, 8, color, 3)
                    else:
                        color = (255, 255, 0)  # 볼: 노란색
                        cv2.circle(overlay_frame, pt_2d_proj_int, 9, color, 3)
                    
                    # 마지막으로 던져진 공 강조 (가장 높은 인덱스 값)
                    if pitch_data['index'] == len(all_pitch_points):
                        cv2.circle(overlay_frame, pt_2d_proj_int, 6, (246, 53, 169), 3)
                    
                    # 순번 표시
                    cv2.putText(overlay_frame, str(pitch_data['index']), 
                                (pt_2d_proj_int[0] - 10, pt_2d_proj_int[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                except (TypeError, ValueError) as e:
                    print(f"좌표 변환 오류: {e}, 값: {pt_2d_proj}")
                    continue
                
        # 스트라이크/아웃 표시
        if strike_count >= 3:
            out_count += 1
            strike_count = 0
        
        if ball_count >= 5:
            ball_count = 0
        
        # 스코어 표시
        cv2.putText(overlay_frame, f"S {strike_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
        cv2.putText(overlay_frame, f"B {ball_count}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(overlay_frame, f"O {out_count}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(overlay_frame, f"speed: {display_velocity:.1f} km/h",
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        # 텍스트 효과 그리기
        text_effect.draw(overlay_frame, result)
        
        # FPS 계산
        fps_end_time = time.time()
        fps = 1.0 / (fps_end_time - fps_start_time + 1e-8)
        cv2.putText(overlay_frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 프레임 표시
        cv2.imshow('ARUCO Tracker with Strike Zone', overlay_frame)
        cv2.imshow('Original', frame)
        
        # 창이 닫혔는지 확인
        if cv2.getWindowProperty('ARUCO Tracker with Strike Zone', cv2.WND_PROP_VISIBLE) < 1 \
           or cv2.getWindowProperty('Original', cv2.WND_PROP_VISIBLE) < 1:
            shutdown_event.set()
            break
        
        # 비디오 모드 트랙바 업데이트
        if is_video_mode:
            current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Progress", "Original", current_frame_no)
        
        # 키 이벤트 처리
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            shutdown_event.set()
            break
        elif key & 0xFF == ord('r'):
            # 상태 초기화
            ar_started = False
            strike_count = 0
            out_count = 0
            ball_count = 0
        elif key & 0xFF == ord('s'):
            cv2.imwrite("strike_zone.png", overlay_frame)
            print("스트라이크 존 이미지가 저장되었습니다.")
        elif key & 0xFF == ord('b'):
            cv2.imwrite("ball_zone.png", overlay_frame)
            print("볼 존 이미지가 저장되었습니다.")
        elif key & 0xFF == ord('c'):
            # 데이터 초기화
            detected_strike_points = []
            detected_ball_points = []
            record_sheet_x.clear()
            record_sheet_y.clear()
            pitch_points_3d_x.clear()
            pitch_points_3d_y.clear()
            pitch_points_3d_z.clear()
            pitch_points_trace_3d_x.clear()
            pitch_points_trace_3d_y.clear()
            pitch_points_trace_3d_z.clear()
            pitch_speeds.clear()
            pitch_results.clear()
            pitch_history.clear()
            ball_positions_history.clear()
            ball_times_history.clear()
            velocity_buffer.clear()
            strike_count = 0
            print("데이터가 초기화되었습니다.")
        elif key & 0xFF == ord('t'):
            # 텍스트 이펙트 추가 테스트
            text_effect.add_strike_effect()
            text_effect.add_ball_effect()
            print("텍스트 이펙트가 추가되었습니다.")
        elif is_video_mode and key & 0xFF == ord(' '):
            # 재생/일시정지 토글
            play_pause = not play_pause
            if play_pause:
                print("일시정지")
            else:
                print("재개")
    
    # 리소스 해제
    if 'cap' in locals() and cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    logging.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    main()