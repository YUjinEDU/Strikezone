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
from tracker_v1 import KalmanTracker, BallDetector, HandDetector
from effects import TextEffect
from dashboard import Dashboard
from kalman_filter import KalmanFilter3D
from baseball_scoreboard import BaseballScoreboard

def main():
    # 전역 종료 이벤트
    global key, rvec
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
    
    prev_distance_to_plane1 = None # 이전 프레임 거리 (초기값 None)
    prev_distance_to_plane2 = None
    
    # 통과 임계값 (0 또는 약간의 음수 값으로 노이즈 제거)
    pass_threshold = 0.0

    frame_count = 0
    last_time = 0.0


    time_freeze_active = False  # 시간 정지 모드 활성화 여부
    frozen_ball_marker_coord = None  # 얼어붙은 공의 마커 기준 3D 좌표
    frozen_trajectory_coords_x = [] # 얼어붙은 궤적의 X 좌표들
    frozen_trajectory_coords_y = [] # 얼어붙은 궤적의 Y 좌표들
    frozen_trajectory_coords_z = [] # 얼어붙은 궤적의 Z 좌표들
    
    # 궤적 관련 플래그와 타이머 변수 추가
    show_trajectory = False
    trajectory_display_start_time = None
    
    # 점수 카운트
    strike_count = 0
    ball_count = 0
    out_count = 0

    # FPS 계산용 변수 초기화
    fps_start_time = time.time()
    last_fps_update_time = time.time()
    display_fps_value = 0
    
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
        video_path = "./video/video_BBS.mp4"
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
    ######################################################################################3
    ######################################################################################3
    ######################################################################################3


    # 1.기본 색상 감지만 적용 (기준) (녹색공)
    ball_detector = BallDetector(GREEN_LOWER, GREEN_UPPER)

    # 2.움직임 감지 추가
    #ball_detector = BallDetector(GREEN_LOWER, GREEN_UPPER, use_motion=True)
    
    # # 3. 다중 프레임 일관성 검사 추가
    #ball_detector = BallDetector(GREEN_LOWER, GREEN_UPPER, use_motion=True, use_consistency=True)

    ######################################################################################3
    ######################################################################################3
    ######################################################################################3


    # 대시보드 초기화 및 시작
    dashboard = Dashboard()
    dashboard_thread = dashboard.run_server()

    # BBS 전광판 추가
    scoreboard = BaseballScoreboard(
        width=0.3, height=0.24, offset_x=-0.4, offset_y=0.0, offset_z=0.1
        )
    
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
                
            if  key & 0xFF == ord(' '):
                play_pause = False
                print("재생 재개")
            elif key & 0xFF == ord('q'):
                shutdown_event.set()
                break
            continue
        
        # 프레임 읽기
        loop_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 분석용 프레임과 오버레이 프레임 분리
        analysis_frame = frame.copy()
        overlay_frame = frame.copy()
        
        # 손 감지
        results = hand_detector.find_hands(frame)
        ar_started = True

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

                    # 전광판 그리기
                    scoreboard.draw(overlay_frame, aruco_detector, rvec, tvec)
                    
                    # 공 궤적 기록 시작
                    if not record_trajectory:
                        record_trajectory = True
                        pitch_points_trace_3d_x.clear()
                        pitch_points_trace_3d_y.clear()
                        pitch_points_trace_3d_z.clear()

                        # 새 투구 감지 시 상태 변수 초기화
                        zone_step1 = False
                        zone_step2 = False
                        prev_distance_to_plane1 = None
                        prev_distance_to_plane2 = None
                    
                    # 공 검출 (녹색)
                    center, radius, _ = ball_detector.detect(analysis_frame)
                    
                    debug_info = {}
                    # debug_info 활용 예시
                    if debug_info:
                        for name, img in debug_info.items():
                            cv2.imshow(f"Debug: {name}", img)

                    if center and radius > 0.4:
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
                        # ball_detector.track_trajectory((projected_pt[0], projected_pt[1]))


                        
                        
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
                        
                        pass_threshold = 0.25
                        
                        # --- 판정 로직 (이전 프레임 정보가 있을 때만 실행) ---
                        if prev_distance_to_plane1 is not None and prev_distance_to_plane2 is not None:
                            # 1단계: plane1 통과 감지
                            print(f"distance_to_plane1: {distance_to_plane1:.4f}, In polygon1: {is_in_polygon1}")
                            # print(f"distance_to_plane2: {distance_to_plane2:.4f}, In polygon2: {is_in_polygon2}")
                            
                            if not zone_step1 and prev_distance_to_plane1 > pass_threshold >= distance_to_plane1 and is_in_polygon1:
                                zone_step1 = True
                                print("1단계 통과")
                                current_time = time.time()
                                
                                # 통과 시각 효과 등 처리
                                try:
                                    overlay = overlay_frame.copy()
                                    cv2.fillPoly(overlay, [projected_points], (200, 200, 0, 128)) # projected_points 필요
                                    alpha = 0.5
                                    cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
                                except Exception as e:
                                    print(f"Overlay effect error: {e}")
                                
                            
                            # 2단계 판정
                            if zone_step1 and not zone_step2 and prev_distance_to_plane2 > pass_threshold >= distance_to_plane2 and is_in_polygon2:
                                    print("****** Plane 2 Passed - STRIKE! ******")


                                    # main.py에 추가
                                    print(f"Distance to plane1: {distance_to_plane1:.4f}, In polygon1: {is_in_polygon1}")
                                    print(f"Distance to plane2: {distance_to_plane2:.4f}, In polygon2: {is_in_polygon2}")
                                    print(f"Ball radius: {radius:.2f}px, Estimated Z: {estimated_Z:.4f}m")
                                    # 스트라이크 판정 (일정 시간 간격)
                                    current_time = time.time()

                                    if current_time - last_time > 2.0:
                                        strike_count += 1
                                        last_time = current_time
                                        
                                        #전광판에 스트라이크 추가
                                        out_added = scoreboard.add_strike()


                                        zone_step2 = True
                                        
                                        print(f"스트라이크 카운트: {scoreboard.strike_count}")
                                        # 공이 ball_zone_corners2 평면을 지나갈 때 반투명 효과 추가
                                        overlay = overlay_frame.copy()
                                        # 다각형 내부를 반투명한 색상으로 채우기
                                        cv2.fillPoly(overlay, [projected_points2], (0, 100, 255, 128))
                                        # 반투명 효과 적용 (알파 블렌딩)
                                        alpha = 0.5  # 투명도 (0: 완전 투명, 1: 완전 불투명)
                                        cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
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
                                        prev_distance_to_plane1 = None
                                        prev_distance_to_plane2 = None
                                        
                            REASONABLE_MIN_DISTANCE = -0.5 # 예시: -0.5m 보다 더 뒤는 오류로 간주
                            if distance_to_plane2 < REASONABLE_MIN_DISTANCE:
                                print(f"Warning: Unlikely distance_to_plane2 ({distance_to_plane2:.2f}), skipping judgment.")
                                zone_step1 = False
                                zone_step2 = False
                                prev_distance_to_plane1 = None
                                prev_distance_to_plane2 = None
                                
                            # 볼 판정
                            elif  distance_to_plane2 <= pass_threshold and not is_in_polygon2:
                                current_time = time.time()
                                if current_time - last_time > 2.0:
                                    
                                    print("****** BALL (Passed P1, Missed P2 Zone) ******")

                                    last_time = current_time
                                    result = "ball"
                                    text_effect.add_ball_effect()

                                    #전광판에 볼 추가
                                    walk_issue = scoreboard.add_ball()
                                    
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

                                    # 볼 판정 후에도 상태 초기화
                                    zone_step1 = False
                                    zone_step2 = False
                                    prev_distance_to_plane1 = None
                                    prev_distance_to_plane2 = None
                        
                        # 이전 거리 값 업데이트
                        # 이전 거리 값이 None이 아닐 때만 업데이트
                        # (첫 번째 프레임에서는 None이므로 업데이트하지 않음)
                        # 이전 거리 값이 None인 경우, 현재 거리 값을 저장
                        
                        prev_distance_to_plane1 = distance_to_plane1
                        prev_distance_to_plane2 = distance_to_plane2

                        
                        # 판정 이벤트 발생 시에만 궤적을 화면에 그리기
                        # if show_trajectory:
                        #     ball_detector.draw_trajectory(overlay_frame)
                        #     # 2초 경과 후 궤적 시각화 종료 및 기록 초기화
                        #     if time.time() - trajectory_display_start_time >= 2.0:
                        #         show_trajectory = False
                        #         ball_detector.pts.clear()  # 궤적 기록 초기화   
                    
                    if show_trajectory and len(pitch_points_trace_3d_x) > 1:

                        trajectory_3d_points_marker_coord = np.array(list(zip(
                            pitch_points_trace_3d_x, 
                            pitch_points_trace_3d_y,
                            pitch_points_trace_3d_z
                        )), dtype=np.float32)

                        if  trajectory_3d_points_marker_coord.ndim ==1:
                            trajectory_3d_points_marker_coord = trajectory_3d_points_marker_coord.reshape(-1, 3)

                        if trajectory_3d_points_marker_coord.shape[0] > 1:
                            projected_trajectory_2d_points = aruco_detector.project_points(
                                trajectory_3d_points_marker_coord, rvec, tvec)
                            
                            if projected_trajectory_2d_points is not None:
                                projected_trajectory_2d_points_int = projected_trajectory_2d_points.reshape((-1, 2)).astype(np.int32)

                                transparent_overlay = overlay_frame.copy()
                               
                                for i in range(1, len(projected_trajectory_2d_points_int)):
                                    pt1  = tuple(projected_trajectory_2d_points_int[i-1])
                                    pt2  = tuple(projected_trajectory_2d_points_int[i])

                                    # 궤적의 오래된 부분은 더 얇게 표현 (선택적)
                                    # thickness = max(1, int(3 * (1.0 - (i / len(projected_trajectory_2d_points_int))) + 1))
                                    thickness = 2 # 여기서는 고정 두께

                                    cv2.line(transparent_overlay, pt1, pt2, (255, 255, 255), thickness)
                                alpah = 0.6
                                cv2.addWeighted(transparent_overlay, alpah, overlay_frame, 1-alpah, 0, overlay_frame)
                                    # 2초 경과 후 궤적 시각화 종료 및 (필요시) 관련 데이터 초기화
                    if show_trajectory and trajectory_display_start_time is not None and \
                    time.time() - trajectory_display_start_time >= 2.0:
                        show_trajectory = False



            # 마커가 감지되지 않았을 때 처리
            else:
                # 마커가 안 보이면 관련 상태 초기화
                record_trajectory = False
                # 이전 거리 값도 리셋하여 다음 감지 시 잘못된 비교 방지
                prev_distance_to_plane1 = None
                prev_distance_to_plane2 = None
                zone_step1 = False
                zone_step2 = False
            

            

 

            
                              
                
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
        
        # 텍스트 효과 그리기
        text_effect.draw(overlay_frame, result)
        
        # FPS 계산 및 표시
        now = time.time()
        delta_time = now - fps_start_time
        if delta_time > 0:
            capture_fps = 1.0 / delta_time

        # 0.5초마다 화면에 표시될 FPS 값 업데이트
        if now - last_fps_update_time >= 0.5:
            display_fps_value = capture_fps # 현재 계산된 FPS로 업데이트
            last_fps_update_time = now

        fps_start_time = now # 다음 프레임 계산을 위해 시작 시간 업데이트

        # 계산된 FPS 값 표시 (0.5초마다 업데이트된 값)
        cv2.putText(overlay_frame, f"FPS: {display_fps_value:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 원본 프레임에도 동일하게 표시 (선택 사항)
        cv2.putText(frame, f"FPS: {display_fps_value:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
            scoreboard.reset()
            
        elif key & 0xFF == ord('s'):
            if not time_freeze_active:
                if ids is not None and 'current_filtered_point' in locals() and current_filtered_point is not None:
                    time_freeze_active = True
                    print("AR Time Freeze : ON!")
                    # 현재 상태를 "얼리기" 위해 정보 저장
                    frozen_ball_marker_coord = current_filtered_point.copy() # 현재 공 위치 (마커 기준) 저장
                    frozen_trajectory_coords_x = list(pitch_points_trace_3d_x) # 현재까지의 궤적 X 복사
                    frozen_trajectory_coords_y = list(pitch_points_trace_3d_y) # 현재까지의 궤적 Y 복사
                    frozen_trajectory_coords_z = list(pitch_points_trace_3d_z) # 현재까지의 궤적 Z 복사
                    # if 'radius' in locals(): frozen_ball_radius = radius # 필요시 현재 공 반지름도 저장

        elif key & 0xFF == ord('b'):
            cv2.imwrite("strike_zone.png", overlay_frame)
            print("스트라이크 존 이미지가 저장되었습니다.")
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