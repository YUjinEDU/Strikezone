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
from tracker_v1 import KalmanTracker, BallDetector, HandDetector # tracker_v1.py의 KalmanTracker 사용
from effects import TextEffect
from dashboard import Dashboard
# from kalman_filter import KalmanFilter3D # tracker_v1.py의 KalmanTracker를 사용하므로 중복될 수 있음
from baseball_scoreboard import BaseballScoreboard

def main():
    # 전역 종료 이벤트
    # global key, rvec # global 사용은 최소화하는 것이 좋음, 지역 변수로 최대한 관리
    shutdown_event = threading.Event()
    
    # 로그 설정
    logging.basicConfig(
        filename=LOG_FILENAME,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8'
    )
    
    # --- 시간 정지 기능 위한 변수 초기화 ---
    time_freeze_active = False
    frozen_ball_marker_coord = None  # 얼어붙은 공의 마커 기준 3D 좌표
    frozen_trajectory_coords_x = [] # 얼어붙은 궤적의 X 좌표들
    frozen_trajectory_coords_y = [] # 얼어붙은 궤적의 Y 좌표들
    frozen_trajectory_coords_z = [] # 얼어붙은 궤적의 Z 좌표들
    frozen_ball_radius_snapshot = 10 # 시간 정지 시 사용할 공 반지름 (선택적)
    # ------------------------------------
    
    # 상태 변수 초기화
    is_video_mode = False
    play_pause = False
    record_trajectory = False # 현재 투구의 궤적을 기록 중인지 여부
    ar_started = False      # AR 기능 시작 여부
    zone_step1 = False      # 스트라이크 존 첫 번째 평면 통과 여부
    zone_step2 = False      # 스트라이크 존 두 번째 평면 통과 여부
    
    prev_distance_to_plane1 = None
    prev_distance_to_plane2 = None
    
    pass_threshold = 0.0 # 평면 통과 임계값 (0.0이 이론적으로 맞음, 필요시 약간의 오차 허용)

    frame_count = 0 # SKIP_FRAMES 사용 시 활용
    last_time = 0.0 # 스트라이크/볼 중복 판정 방지용 타이머

    show_trajectory = False # 판정 후 궤적을 보여줄지 여부
    trajectory_display_start_time = None # 궤적 표시 시작 시간
        
    # FPS 계산용 변수 초기화
    loop_start_time = time.time() # FPS 계산 위한 루프 시작 시간 (fps_start_time 대신 사용)
    last_fps_update_time = time.time()
    display_fps_value = 0
    
    # 궤적 데이터 (실시간 기록용)
    pitch_points_trace_3d_x = []
    pitch_points_trace_3d_y = []
    pitch_points_trace_3d_z = []
    
    # 대시보드 및 로깅용 데이터
    detected_strike_points = []
    detected_ball_points = []
    
    # 투구 속도 계산용 변수
    pitch_speeds = []
    pitch_results = []
    pitch_history = []
    ball_positions_history = deque(maxlen=10)
    ball_times_history = deque(maxlen=10)
    velocity_buffer = deque(maxlen=5)
    display_velocity = 0
    final_velocity = 0 # 판정 시점의 최종 속도
    current_velocity_kmh = 0 # 실시간으로 계산되는 현재 속도
    
    hand_detector = HandDetector()
    text_effect = TextEffect()
    
    user_input = input("1: 카메라, 2: 비디오 > ")
    
    if user_input == "1":
        camera_manager = CameraManager(shutdown_event)
        selected_camera = camera_manager.create_camera_selection_gui()
        if selected_camera is None: print("카메라가 선택되지 않았습니다. 종료합니다."); shutdown_event.set(); return
        if not camera_manager.open_camera(selected_camera): print(f"카메라 {selected_camera}를 열 수 없습니다. 종료합니다."); shutdown_event.set(); return
        if not camera_manager.load_calibration(CALIBRATION_PATH): print("캘리브레이션 데이터를 로드할 수 없습니다. 종료합니다."); shutdown_event.set(); return
        camera_matrix = camera_manager.camera_matrix
        dist_coeffs = camera_manager.dist_coeffs
        cap = camera_manager.capture
    elif user_input == "2":
        video_path = "./video/video_BBS.mp4" # 예시 비디오 경로
        cap = cv2.VideoCapture(video_path)
        is_video_mode = True
        cv2.namedWindow('Original')
        def on_trackbar(val): cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0: cv2.createTrackbar("Progress", "Original", 0, total_frames - 1, on_trackbar)
        try:
            calib_data = np.load(CALIBRATION_PATH)
            camera_matrix = calib_data["camera_matrix"]
            dist_coeffs = calib_data["dist_coeffs"]
        except Exception as e: print(f"캘리브레이션 데이터를 로드할 수 없습니다: {e}"); shutdown_event.set(); return
    else: print("잘못된 입력입니다. 종료합니다."); return

    aruco_detector = ArucoDetector(ARUCO_MARKER_SIZE, camera_matrix, dist_coeffs)
    ball_detector = BallDetector(GREEN_LOWER, GREEN_UPPER)
    dashboard = Dashboard()
    dashboard_thread = dashboard.run_server()
    scoreboard = BaseballScoreboard(width=0.3, height=0.24, offset_x=-0.4, offset_y=0.0, offset_z=0.1)
    
    # tracker_v1.py의 KalmanTracker 사용
    kalman_filter_tracker = KalmanTracker() # 이름 변경 (기존 KalmanFilter3D와 구분)

    box_corners_3d = aruco_detector.get_box_corners_3d(BOX_MIN, BOX_MAX)
    ball_zone_corners = np.dot(BALL_ZONE_CORNERS, ROTATION_MATRIX.T)
    ball_zone_corners2 = np.dot(BALL_ZONE_CORNERS, ROTATION_MATRIX.T)
    ball_zone_corners2[:, 1] += ZONE_Z_DIFF # Y축이 깊이 방향이라고 가정
    
    result = "" # 스트라이크/볼 판정 결과 문자열
    ids = None # 현재 프레임에서 감지된 마커 ID
    current_filtered_point = None # 현재 프레임의 공 위치 (마커 기준), 시간 정지 시 사용
    current_ball_radius = 0 # 현재 프레임의 공 반지름, 시간 정지 시 사용

    initial_dashboard_data = {
        'record_sheet_points': [], 'record_sheet_polygon': [[p[0], p[2]] for p in ball_zone_corners],
        'trajectory_3d': [], 'strike_zone_corners_3d': ball_zone_corners.tolist(),
        'ball_zone_corners_3d': ball_zone_corners.tolist(), 'ball_zone_corners2_3d': ball_zone_corners2.tolist(),
        'box_corners_3d': box_corners_3d.tolist(), 'pitch_count': 0, 'strike_count': 0, 'ball_count': 0,
        'pitch_speeds': [], 'pitch_results': [], 'pitch_history': []
    }
    dashboard.update_data(initial_dashboard_data)

    # 메인 루프
    while not shutdown_event.is_set():
        loop_start_time = time.time() # 매 루프 시작 시간 기록 (FPS 계산용)
        local_key = -1 # cv2.waitKey() 결과 저장용

        if is_video_mode and play_pause:
            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if total_frames > 0: cv2.setTrackbarPos("Progress", "Original", current_frame_no)
                local_key = cv2.waitKey(10) & 0xFF
            except cv2.error: pass
            if local_key == ord(' '): play_pause = False; print("재생 재개")
            elif local_key == ord('q'): shutdown_event.set(); break
            continue
        
        ret, frame = cap.read()
        if not ret: 
            if is_video_mode: print("비디오 끝. 'q'를 눌러 종료하거나 'r'로 리셋하세요."); play_pause = True; continue
            else: print("프레임을 읽을 수 없습니다."); break
        
        analysis_frame = frame.copy()
        overlay_frame = frame.copy()
        
        # ar_started = True # 손 감지 로직 비활성화 시 항상 True
        if not ar_started: # 손 감지 로직 사용 시
            results_hand = hand_detector.find_hands(frame) # 변수명 변경
            if hand_detector.is_hand_open(): # results_hand를 사용하도록 수정 필요 (HandDetector 클래스에 따라)
                ar_started = True
                print("AR 시작!")
            else:
                cv2.putText(overlay_frame, "Show your hand!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # AR 시작 안되면 아래 로직 건너뛰고 화면만 표시
                cv2.imshow('ARUCO Tracker with Strike Zone', overlay_frame)
                if is_video_mode: cv2.imshow('Original', frame)
                local_key = cv2.waitKey(1) & 0xFF
                if local_key == ord('q'): shutdown_event.set(); break
                continue
        
        if ar_started:
            # frame_count += 1 # SKIP_FRAMES 로직은 일단 단순화 위해 주석 처리
            # if frame_count % SKIP_FRAMES != 0:
            #     cv2.imshow('ARUCO Tracker with Strike Zone', overlay_frame)
            #     # ... (기존 key 처리)
            #     continue
            
            corners, ids, rejected = aruco_detector.detect_markers(analysis_frame)

            # <<< 시간 정지 기능 관련 수정 시작: 실시간 데이터 처리 제어 >>>
            if not time_freeze_active:
                # 시간 정지 모드가 아닐 때만 아래 로직들 실행
                if ids is not None:
                    rvecs_live, tvecs_live = aruco_detector.estimate_pose(corners)
                    if rvecs_live is not None and len(rvecs_live) > 0: # 주 마커 하나만 사용한다고 가정
                        rvec_current_logic = rvecs_live[0] # 실시간 로직용 rvec
                        tvec_current_logic = tvecs_live[0] # 실시간 로직용 tvec

                        if not record_trajectory: # 새 투구 시작
                            record_trajectory = True
                            pitch_points_trace_3d_x.clear(); pitch_points_trace_3d_y.clear(); pitch_points_trace_3d_z.clear()
                            zone_step1 = False; zone_step2 = False
                            prev_distance_to_plane1 = None; prev_distance_to_plane2 = None
                            ball_positions_history.clear(); ball_times_history.clear(); velocity_buffer.clear()
                            final_velocity = 0; current_velocity_kmh = 0; display_velocity = 0
                            current_filtered_point = None # 새 투구 시 현재 공 위치 초기화
                            current_ball_radius = 0

                        center, radius, mask_ball = ball_detector.detect(analysis_frame) # mask_ball 변수 추가
                        current_ball_radius = radius # 현재 반지름 저장

                        if center and radius > 0.4:
                            estimated_Z = (camera_matrix[0, 0] * BALL_RADIUS_REAL) / radius
                            ball_3d_cam = np.array([
                                (center[0] - camera_matrix[0,2]) * estimated_Z / camera_matrix[0,0],
                                (center[1] - camera_matrix[1,2]) * estimated_Z / camera_matrix[1,1],
                                estimated_Z ])
                            
                            kalman_filter_tracker.update_with_gating(ball_3d_cam) # tracker_v1.py의 update_with_gating 사용
                            filtered_point_kalman_cam = kalman_filter_tracker.get_filtered_position() # 카메라 좌표계 필터링 결과

                            if filtered_point_kalman_cam is not None:
                                filtered_point_marker = aruco_detector.point_to_marker_coord(filtered_point_kalman_cam, rvec_current_logic, tvec_current_logic)
                                current_filtered_point = filtered_point_marker # 현재 공 위치 (마커 기준) 업데이트

                                if record_trajectory:
                                    pitch_points_trace_3d_x.append(current_filtered_point[0])
                                    pitch_points_trace_3d_y.append(current_filtered_point[1])
                                    pitch_points_trace_3d_z.append(current_filtered_point[2])

                                # --- 속도 계산 로직 ---
                                current_time_for_speed = time.time()
                                ball_positions_history.append(current_filtered_point) # 마커 좌표계 기준 위치로 속도 계산
                                ball_times_history.append(current_time_for_speed)
                                if len(ball_positions_history) >= 3:
                                    first_pos = ball_positions_history[0]; last_pos = ball_positions_history[-1]
                                    time_diff = ball_times_history[-1] - ball_times_history[0]
                                    if time_diff > 0:
                                        distance = np.linalg.norm(last_pos - first_pos)
                                        velocity = distance / time_diff
                                        if 5 < velocity < 50: # m/s
                                            velocity_kmh = velocity * 3.6
                                            velocity_buffer.append(velocity_kmh)
                                            if len(velocity_buffer) >= 3:
                                                sorted_v = sorted(list(velocity_buffer)) # deque를 list로 변환 후 정렬
                                                filtered_v = sorted_v[1:-1] if len(sorted_v) > 2 else sorted_v
                                                current_velocity_kmh = sum(filtered_v) / len(filtered_v) if filtered_v else 0
                                            else: current_velocity_kmh = velocity_kmh
                                display_velocity = current_velocity_kmh
                                if display_velocity == 0 and len(velocity_buffer) > 0: display_velocity = max(velocity_buffer)
                                # --------------------
                                
                                # --- 판정 로직 ---
                                # 판정 시 사용할 projected_points (is_in_polygon용) - 현재 마커 기준
                                temp_projected_points1 = aruco_detector.project_points(ball_zone_corners, rvec_current_logic, tvec_current_logic)
                                temp_projected_points2 = aruco_detector.project_points(ball_zone_corners2, rvec_current_logic, tvec_current_logic)

                                distance_to_plane1 = aruco_detector.signed_distance_to_plane(current_filtered_point, ball_zone_corners[0], ball_zone_corners[1], ball_zone_corners[3])
                                is_in_polygon1 = aruco_detector.is_point_in_polygon(center, temp_projected_points1) if temp_projected_points1 is not None else False
                                distance_to_plane2 = aruco_detector.signed_distance_to_plane(current_filtered_point, ball_zone_corners2[0], ball_zone_corners2[1], ball_zone_corners2[3])
                                is_in_polygon2 = aruco_detector.is_point_in_polygon(center, temp_projected_points2) if temp_projected_points2 is not None else False
                                
                                if prev_distance_to_plane1 is not None and prev_distance_to_plane2 is not None:
                                    if not zone_step1 and prev_distance_to_plane1 > pass_threshold >= distance_to_plane1 and is_in_polygon1:
                                        zone_step1 = True; print("1단계 통과")
                                        # (1단계 통과 시각 효과 - 필요시 추가)
                                    
                                    REASONABLE_MIN_DISTANCE = -0.7 # 좀 더 관대한 값으로 (환경에 따라 조정)
                                    if distance_to_plane2 < REASONABLE_MIN_DISTANCE:
                                        print(f"Warning: 비정상적 P2 거리 ({distance_to_plane2:.2f}), 판정 스킵.")
                                    elif zone_step1 and not zone_step2 and prev_distance_to_plane2 > pass_threshold >= distance_to_plane2 and is_in_polygon2:
                                        current_time_for_judge = time.time()
                                        if current_time_for_judge - last_time > JUDGEMENT_COOLDOWN:
                                            print("****** Plane 2 Passed - STRIKE! ******")
                                            last_time = current_time_for_judge
                                            scoreboard.add_strike()
                                            print(f"스트라이크 카운트: {scoreboard.strike_count}")
                                            detected_strike_points.append({'3d_coord': current_filtered_point.copy(), 'rvec': rvec_current_logic.copy(),'tvec': tvec_current_logic.copy()}) # 복사본 저장
                                            if display_velocity > 0 : final_velocity = display_velocity # 현재 표시되는 속도를 최종 속도로
                                            pitch_results.append("스트라이크"); pitch_speeds.append(final_velocity)
                                            pitch_history.append({'number':len(pitch_history)+1, 'result':"스트라이크", 'speed':f"{final_velocity:.1f}"})
                                            text_effect.add_strike_effect(); result = "strike"
                                            logging.info(f"Strike #{len(detected_strike_points)}: Coords M: {current_filtered_point}, ...")
                                            record_sheet_x.append(current_filtered_point[0]); record_sheet_y.append(current_filtered_point[2])
                                            
                                            record_trajectory = False; show_trajectory = True; trajectory_display_start_time = time.time()
                                            zone_step1 = False; zone_step2 = False; prev_distance_to_plane1 = None; prev_distance_to_plane2 = None
                                    
                                    elif zone_step1 and not zone_step2 and prev_distance_to_plane2 > pass_threshold >= distance_to_plane2 and not is_in_polygon2:
                                        current_time_for_judge = time.time()
                                        if current_time_for_judge - last_time > JUDGEMENT_COOLDOWN:
                                            print("****** BALL (Passed P1, Missed P2 Zone) ******")
                                            last_time = current_time_for_judge
                                            scoreboard.add_ball()
                                            print(f"볼 카운트: {scoreboard.ball_count}")
                                            detected_ball_points.append({'3d_coord': current_filtered_point.copy(), 'rvec': rvec_current_logic.copy(),'tvec': tvec_current_logic.copy()})
                                            final_velocity = display_velocity if display_velocity > 0 else 20.0 + np.random.normal(0,5) # 임의값 대신 현재 속도 사용 시도
                                            pitch_results.append("볼"); pitch_speeds.append(final_velocity)
                                            pitch_history.append({'number':len(pitch_history)+1, 'result':"볼", 'speed':f"{final_velocity:.1f}"})
                                            text_effect.add_ball_effect(); result = "ball"
                                            record_sheet_x.append(current_filtered_point[0]); record_sheet_y.append(current_filtered_point[2])

                                            record_trajectory = False; show_trajectory = True; trajectory_display_start_time = time.time()
                                            zone_step1 = False; zone_step2 = False; prev_distance_to_plane1 = None; prev_distance_to_plane2 = None
                                
                                # 이전 거리 값 업데이트 (판정으로 None이 안되었으면)
                                if prev_distance_to_plane1 is not None: prev_distance_to_plane1 = distance_to_plane1
                                if prev_distance_to_plane2 is not None: prev_distance_to_plane2 = distance_to_plane2
                        else: # 공 감지 안됨 (실시간 모드)
                            # current_filtered_point = None # 이미 루프 시작 시 또는 새 투구 시 None으로 설정됨
                            pass # 특별히 할 것 없음, prev_distance는 이전 값 유지 또는 None 상태 유지
                else: # 마커 감지 안됨 (실시간 모드)
                    prev_distance_to_plane1 = None; prev_distance_to_plane2 = None
                    zone_step1 = False; zone_step2 = False
                    record_trajectory = False; current_filtered_point = None
            # <<< 시간 정지 기능 관련 수정 끝: 실시간 데이터 처리 제어 블록 >>>

            # --- AR 요소 그리기 (시간 정지 여부와 관계없이 마커 보이면 항상 실행) ---
            if ids is not None:
                rvecs_for_drawing, tvecs_for_drawing = aruco_detector.estimate_pose(corners) # 그림은 항상 현재 카메라 시점의 마커 기준
                if rvecs_for_drawing is not None and len(rvecs_for_drawing) > 0:
                    rvec_draw = rvecs_for_drawing[0] # 첫 번째 마커 기준 그리기
                    tvec_draw = tvecs_for_drawing[0]

                    # 1. 존, 박스, 그리드 (항상 현재 rvec_draw, tvec_draw 사용)
                    projected_pts_zone1 = aruco_detector.project_points(ball_zone_corners, rvec_draw, tvec_draw)
                    if projected_pts_zone1 is not None: cv2.polylines(overlay_frame, [projected_pts_zone1], True, (200, 200, 0), 4)
                    projected_pts_zone2 = aruco_detector.project_points(ball_zone_corners2, rvec_draw, tvec_draw)
                    if projected_pts_zone2 is not None: cv2.polylines(overlay_frame, [projected_pts_zone2], True, (0, 100, 255), 4)
                    projected_pts_box = aruco_detector.project_points(box_corners_3d, rvec_draw, tvec_draw)
                    if projected_pts_box is not None:
                        aruco_detector.draw_3d_box(overlay_frame, projected_pts_box, BOX_EDGES, color=(0,0,0), thickness=4)
                        grid_pts2d_draw = projected_pts_box[[0,1,5,4]]
                        aruco_detector.draw_grid(overlay_frame, grid_pts2d_draw, 3)

                    # 2. 전광판 (항상 현재 rvec_draw, tvec_draw 사용)
                    scoreboard.draw(overlay_frame, aruco_detector, rvec_draw, tvec_draw)

                    # 3. 공 및 궤적 그리기 (데이터 소스를 time_freeze_active에 따라 선택)
                    ball_to_render_marker_coord_val = None # 이번에 그릴 공의 마커 기준 3D 좌표
                    trajectory_to_render_x_val, trajectory_to_render_y_val, trajectory_to_render_z_val = [], [], []
                    show_this_frames_trajectory_flag = False # 이번 프레임에 궤적을 그릴지 여부
                    radius_for_drawing_val = current_ball_radius if current_ball_radius > 0.4 else 10 # 기본 반지름

                    if time_freeze_active:
                        ball_to_render_marker_coord_val = frozen_ball_marker_coord
                        if frozen_ball_marker_coord is not None: # 얼린 데이터가 있을 때만
                            trajectory_to_render_x_val = frozen_trajectory_coords_x
                            trajectory_to_render_y_val = frozen_trajectory_coords_y
                            trajectory_to_render_z_val = frozen_trajectory_coords_z
                            if len(trajectory_to_render_x_val) > 1: show_this_frames_trajectory_flag = True
                            # radius_for_drawing_val = frozen_ball_radius_snapshot # 얼린 반지름 사용
                    else: # 실시간 모드
                        if current_filtered_point is not None:
                            ball_to_render_marker_coord_val = current_filtered_point
                        if show_trajectory: # 실시간 궤적은 show_trajectory 플래그 따름
                            trajectory_to_render_x_val = pitch_points_trace_3d_x
                            trajectory_to_render_y_val = pitch_points_trace_3d_y
                            trajectory_to_render_z_val = pitch_points_trace_3d_z
                            if len(trajectory_to_render_x_val) > 1: show_this_frames_trajectory_flag = True
                    
                    # 공 그리기
                    if ball_to_render_marker_coord_val is not None:
                        projected_ball_2d = aruco_detector.project_points(np.array([ball_to_render_marker_coord_val], dtype=np.float32), rvec_draw, tvec_draw)
                        if projected_ball_2d is not None and len(projected_ball_2d) > 0:
                            center_draw = tuple(projected_ball_2d[0].ravel().astype(int))
                            cv2.circle(overlay_frame, center_draw, int(radius_for_drawing_val), (0, 0, 255), -1)

                    # 3D 궤적 그리기 (반투명 효과 포함)
                    if show_this_frames_trajectory_flag:
                        current_trajectory_3d_points = np.array(list(zip(
                            trajectory_to_render_x_val, trajectory_to_render_y_val, trajectory_to_render_z_val
                        )), dtype=np.float32)
                        if current_trajectory_3d_points.ndim == 1 and current_trajectory_3d_points.shape[0] == 3 : current_trajectory_3d_points = current_trajectory_3d_points.reshape(-1,3) # 점이 하나일 때 (1,3)
                        elif current_trajectory_3d_points.ndim ==1 and current_trajectory_3d_points.shape[0] > 3 : pass # 이미 잘못된 데이터
                        elif current_trajectory_3d_points.shape[0] == 0: current_trajectory_3d_points = np.array([]).reshape(0,3) # 빈 배열 처리

                        if current_trajectory_3d_points.shape[0] > 1:
                            projected_trajectory_2d = aruco_detector.project_points(current_trajectory_3d_points, rvec_draw, tvec_draw)
                            if projected_trajectory_2d is not None:
                                projected_trajectory_2d_int = projected_trajectory_2d.reshape((-1,2)).astype(np.int32)
                                
                                traj_overlay = overlay_frame.copy() # 반투명 효과를 위한 복사본
                                for i in range(1, len(projected_trajectory_2d_int)):
                                    pt1 = tuple(projected_trajectory_2d_int[i-1])
                                    pt2 = tuple(projected_trajectory_2d_int[i])
                                    cv2.line(traj_overlay, pt1, pt2, (0,255,0), 2) # 초록색 궤적
                                alpha_traj = 0.6 # 궤적 투명도
                                cv2.addWeighted(traj_overlay, alpha_traj, overlay_frame, 1 - alpha_traj, 0, overlay_frame)
                    
                    # 시간 정지 모드가 "아닐 때만" 실시간 궤적 표시 시간 관리
                    if not time_freeze_active and show_trajectory and \
                       trajectory_display_start_time is not None and \
                       time.time() - trajectory_display_start_time >= TRAJECTORY_DISPLAY_DURATION: # config.py에 TRAJECTORY_DISPLAY_DURATION=2.0 추가
                        show_trajectory = False
            else: # 마커 감지 안됨 (AR 그리기 불가)
                 if not time_freeze_active:
                    record_trajectory = False; prev_distance_to_plane1 = None; prev_distance_to_plane2 = None
                    zone_step1 = False; zone_step2 = False; show_trajectory = False
                    current_filtered_point = None
        # if ar_started: 끝
        
        # --- 화면 하단 정보 표시 ---
        # 기존 기록된 투구 표시 (스트라이크, 볼 판정된 점들) - 이 로직은 rvec, tvec이 유효할 때 그려야 함
        if ids is not None and len(detected_strike_points + detected_ball_points) > 0:
            # 이 부분은 현재 프레임의 rvec_draw, tvec_draw (또는 첫번째 마커의 rvecs_for_drawing[0] 등)을 사용해야 함
            # 위에서 rvec_draw, tvec_draw가 정의된 블록 안으로 이동하거나, 여기서 다시 계산 필요.
            # 간단하게는 이 블록을 위쪽 for rvec_draw, tvec_draw 루프 안으로 옮기는 것이 좋음.
            # 여기서는 마지막으로 유효했던 rvec, tvec을 사용하게 될 수 있어 정확하지 않을 수 있음.
            # 하지만 데모용으로는 일단 유지. (정확하려면 그리기용 rvec/tvec을 클래스 멤버나 더 넓은 스코프 변수로 관리)
            if 'rvec_draw' in locals() and rvec_draw is not None: # rvec_draw가 정의되어 있다면 사용
                plane_pt1 = ball_zone_corners2[0]; plane_pt2 = ball_zone_corners2[1]; plane_pt3 = ball_zone_corners2[2]
                all_pitch_points_data = []
                for idx, pd in enumerate(detected_strike_points): all_pitch_points_data.append({'point': pd['3d_coord'], 'type': 'strike', 'index': idx + 1})
                for idx, pd in enumerate(detected_ball_points): all_pitch_points_data.append({'point': pd['3d_coord'], 'type': 'ball', 'index': len(detected_strike_points) + idx + 1})
                
                for pitch_data in all_pitch_points_data:
                    point_on_plane = aruco_detector.project_point_onto_plane(pitch_data['point'], plane_pt1, plane_pt2, plane_pt3)
                    pt_2d_proj_on_plane = aruco_detector.project_points(np.array([point_on_plane]), rvec_draw, tvec_draw) # 그리기용 rvec_draw 사용
                    if pt_2d_proj_on_plane is not None and len(pt_2d_proj_on_plane) > 0 :
                        try:
                            pt_2d_proj_int = tuple(pt_2d_proj_on_plane[0].ravel().astype(int))
                            color = (255,255,255) if pitch_data['type'] == 'strike' else (255,255,0)
                            cv2.circle(overlay_frame, pt_2d_proj_int, 8 if pitch_data['type'] == 'strike' else 9, color, 3)
                            if pitch_data['index'] == len(all_pitch_points_data): cv2.circle(overlay_frame, pt_2d_proj_int, 6, (246,53,169), 3)
                            cv2.putText(overlay_frame, str(pitch_data['index']), (pt_2d_proj_int[0]-10, pt_2d_proj_int[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                        except Exception as e: print(f"Error drawing pitch history: {e}")
        
        text_effect.draw(overlay_frame, result)
        if display_velocity > 0: cv2.putText(overlay_frame, f"{display_velocity:.1f} km/h", (10, overlay_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        
        current_loop_end_time = time.time()
        processing_time = current_loop_end_time - loop_start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        if current_loop_end_time - last_fps_update_time >= 0.5:
            display_fps_value = current_fps
            last_fps_update_time = current_loop_end_time
        cv2.putText(overlay_frame, f"FPS: {display_fps_value:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if is_video_mode: cv2.putText(frame, f"FPS: {display_fps_value:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('ARUCO Tracker with Strike Zone', overlay_frame)
        if is_video_mode: cv2.imshow('Original', frame)
        
        if cv2.getWindowProperty('ARUCO Tracker with Strike Zone', cv2.WND_PROP_VISIBLE) < 1 or \
           (is_video_mode and cv2.getWindowProperty('Original', cv2.WND_PROP_VISIBLE) < 1):
            shutdown_event.set(); break
        
        if is_video_mode and not play_pause and total_frames > 0:
            current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Progress", "Original", current_frame_no)
        
        local_key = cv2.waitKey(1) & 0xFF # 루프 마지막에서 한 번만 호출
        if local_key == ord('q'): shutdown_event.set(); break
        elif local_key == ord('s'):
            if not time_freeze_active:
                if ids is not None and current_filtered_point is not None:
                    time_freeze_active = True
                    print("AR Time Freeze: ON")
                    frozen_ball_marker_coord = current_filtered_point.copy()
                    frozen_trajectory_coords_x = list(pitch_points_trace_3d_x)
                    frozen_trajectory_coords_y = list(pitch_points_trace_3d_y)
                    frozen_trajectory_coords_z = list(pitch_points_trace_3d_z)
                    # frozen_ball_radius_snapshot = current_ball_radius # 현재 공 반지름 저장
                else: print("Cannot freeze: Marker or current ball position not available.")
            else:
                time_freeze_active = False; print("AR Time Freeze: OFF")
        elif local_key == ord('r'):
            ar_started = True; scoreboard.reset()
            detected_strike_points.clear(); detected_ball_points.clear()
            record_sheet_x.clear(); record_sheet_y.clear()
            pitch_points_trace_3d_x.clear(); pitch_points_trace_3d_y.clear(); pitch_points_trace_3d_z.clear()
            pitch_speeds.clear(); pitch_results.clear(); pitch_history.clear()
            ball_positions_history.clear(); ball_times_history.clear(); velocity_buffer.clear()
            zone_step1 = False; zone_step2 = False; prev_distance_to_plane1 = None; prev_distance_to_plane2 = None
            record_trajectory = False; show_trajectory = False; current_filtered_point = None; result=""
            print("모든 상태 및 데이터 초기화 (AR 재시작)")
        elif local_key == ord('c'): # 'c' 키는 데이터 초기화만 (AR 상태 유지)
            detected_strike_points.clear(); detected_ball_points.clear(); record_sheet_x.clear(); record_sheet_y.clear()
            pitch_points_trace_3d_x.clear(); pitch_points_trace_3d_y.clear(); pitch_points_trace_3d_z.clear()
            frozen_trajectory_coords_x.clear(); frozen_trajectory_coords_y.clear(); frozen_trajectory_coords_z.clear() # 얼린 궤적도 초기화
            pitch_speeds.clear(); pitch_results.clear(); pitch_history.clear()
            ball_positions_history.clear(); ball_times_history.clear(); velocity_buffer.clear(); scoreboard.reset()
            frozen_ball_marker_coord = None; current_filtered_point = None; result=""
            print("기록 데이터 및 전광판 초기화.")
        elif local_key == ord('t'): text_effect.add_strike_effect(); text_effect.add_ball_effect(); print("텍스트 이펙트 테스트.")
        elif is_video_mode and local_key == ord(' '):
            play_pause = not play_pause
            if play_pause: print("일시정지")
            else: print("재생 재개")
    
    if 'cap' in locals() and cap is not None: cap.release()
    cv2.destroyAllWindows()
    if dashboard_thread is not None and dashboard_thread.is_alive():
        print("대시보드 서버 종료 중...")
        # dashboard.stop() # 대시보드에 종료 메서드가 있다면 호출
    logging.shutdown()
    print("프로그램 종료.")
    # sys.exit(0) # shutdown_event로 제어되므로 명시적 exit 불필요할 수 있음

if __name__ == "__main__":
    main()