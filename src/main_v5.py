import os
import sys
import cv2
import numpy as np
import threading
import time
import logging
from collections import deque

# 내부 모듈
from config import *
from camera import CameraManager
from aruco_detector import ArucoDetector
from tracker_v1 import KalmanTracker, BallDetector, HandDetector
from effects import TextEffect
from dashboard import Dashboard
from kalman_filter import KalmanFilter3D
from baseball_scoreboard import BaseballScoreboard

def main():
    global key
    shutdown_event = threading.Event()

    # 로그 설정
    logging.basicConfig(
        filename=LOG_FILENAME,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8'
    )

    # 상태 변수
    is_video_mode = False
    play_pause = False
    record_trajectory = False
    ar_started = False
    debug_mode = False  # 디버그 모드 플래그 추가

    # 교차 거리/시간(고정밀) 추적 변수
    prev_distance_to_plane1 = None
    prev_distance_to_plane2 = None
    prev_time_perf = None
    t_cross_plane1 = None
    t_cross_plane2 = None
    


    # FPS 표시용
    fps_start_time = time.time()
    last_fps_update_time = time.time()
    display_fps_value = 0

    # 속도/기록 관련
    pitch_speeds = []
    pitch_results = []
    pitch_history = []
    display_velocity = 0.0
    final_velocity = 0.0

    # 궤적 기록(마커 좌표계)
    pitch_points_trace_3d_x = []
    pitch_points_trace_3d_y = []
    pitch_points_trace_3d_z = []

    # 손 감지/텍스트 효과
    hand_detector = HandDetector()
    text_effect = TextEffect()

    # 입력 선택
    user_input = input("1: 카메라, 2: 비디오 > ")
    if user_input == "1":
        camera_manager = CameraManager(shutdown_event)
        selected_camera = camera_manager.create_camera_selection_gui()
        if selected_camera is None:
            print("카메라가 선택되지 않았습니다. 종료합니다.")
            shutdown_event.set()
            return
        if not camera_manager.open_camera(selected_camera):
            print(f"카메라 {selected_camera}를 열 수 없습니다. 종료합니다.")
            shutdown_event.set()
            return
        if not camera_manager.load_calibration(CALIBRATION_PATH):
            print("캘리브레이션 데이터를 로드할 수 없습니다. 종료합니다.")
            shutdown_event.set()
            return

        camera_matrix = camera_manager.camera_matrix
        dist_coeffs = camera_manager.dist_coeffs
        cap = camera_manager.capture
    elif user_input == "2":
        video_path = "./video/video_BBS.mp4"
        cap = cv2.VideoCapture(video_path)
        is_video_mode = True

        # 트랙바
        cv2.namedWindow('Original')
        def on_trackbar(val):
            cap.set(cv2.CAP_PROP_POS_FRAMES, val)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.createTrackbar("Progress", "Original", 0, max(0, total_frames - 1), on_trackbar)

        # 캘리 로드
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

    # ArUco
    aruco_detector = ArucoDetector(ARUCO_MARKER_SIZE, camera_matrix, dist_coeffs)

    # 공 검출기
    ball_detector = BallDetector(GREEN_LOWER, GREEN_UPPER)

    # 대시보드
    dashboard = Dashboard()
    dashboard_thread = dashboard.run_server()

    # 스코어보드(AR)
    scoreboard = BaseballScoreboard(width=0.3, height=0.24, offset_x=-0.4, offset_y=0.0, offset_z=0.1)

    # 초기 대시보드 데이터
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

    # 칼만 필터(3D)
    kalman_tracker = KalmanFilter3D()

    # 박스/존 코너(마커 좌표계로 그대로 사용)
    box_corners_3d = aruco_detector.get_box_corners_3d(BOX_MIN, BOX_MAX)
    ball_zone_corners = BALL_ZONE_CORNERS.copy()
    ball_zone_corners2 = BALL_ZONE_CORNERS.copy()
    ball_zone_corners2[:, 1] += ZONE_DEPTH  # 깊이(Y) 방향으로 후면 이동

    # 평면 법선 방향 기준(+Y)
    desired_depth_axis = np.array([0, 1, 0], dtype=np.float32)

    # 왜곡 보정 맵 준비 (실시간 remap)
    undistort_enabled = True
    undistort_map1, undistort_map2 = None, None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_w == 0 or frame_h == 0:
        _ret, _tmp = cap.read()
        if _ret:
            frame_h, frame_w = _tmp.shape[:2]
            if is_video_mode:
                pass
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if undistort_enabled and frame_w > 0 and frame_h > 0:
        new_K, _roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (frame_w, frame_h), alpha=0)
        undistort_map1, undistort_map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_K, (frame_w, frame_h), cv2.CV_16SC2
        )

    # 기록지 용 포인트(x,z)
    record_sheet_x = []
    record_sheet_z = []

    # 루프
    while not shutdown_event.is_set():
        # 비디오 모드 일시정지 처리
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
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 왜곡 보정
        if undistort_enabled and undistort_map1 is not None:
            frame = cv2.remap(frame, undistort_map1, undistort_map2, interpolation=cv2.INTER_LINEAR)

        analysis_frame = frame.copy()
        overlay_frame = frame.copy()

        # 손 감지 (여기서는 항상 시작)
        hand_detector.find_hands(frame)
        ar_started = True

        if ar_started:
            # ArUco 탐지
            corners, ids, rejected = aruco_detector.detect_markers(analysis_frame)
            
            
            if ids is not None:
                # 마커 포즈
                rvecs, tvecs = aruco_detector.estimate_pose(corners)
                
                

                for rvec, tvec in zip(rvecs, tvecs):
                    # 존 폴리곤 투영
                    
                    aruco_detector.draw_axes(overlay_frame, rvec, tvec, size=0.1)
                    
                    projected_points = aruco_detector.project_points(ball_zone_corners, rvec, tvec)
                    cv2.polylines(overlay_frame, [projected_points], True, (0, 255, 255), 5)

                    projected_points2 = aruco_detector.project_points(ball_zone_corners2, rvec, tvec)
                    cv2.polylines(overlay_frame, [projected_points2], True, (255, 100, 0), 5)

                    # 박스 그리기
                    pts2d = aruco_detector.project_points(box_corners_3d, rvec, tvec)
                    aruco_detector.draw_3d_box(overlay_frame, pts2d, BOX_EDGES, color=(0,0,0), thickness=3)

                    # 스코어보드
                    scoreboard.draw(overlay_frame, aruco_detector, rvec, tvec)


                    # 궤적 기록 시작(새 투구)
                    if not record_trajectory:
                        record_trajectory = True
                        pitch_points_trace_3d_x.clear()
                        pitch_points_trace_3d_y.clear()
                        pitch_points_trace_3d_z.clear()

                        prev_distance_to_plane1 = None
                        prev_distance_to_plane2 = None
                        prev_time_perf = None
                        t_cross_plane1 = None
                        t_cross_plane2 = None

                    # 공 검출
                    center, radius, _ = ball_detector.detect(analysis_frame)
                    if center and radius > 0.4:
                        # 공 표시
                        ball_detector.draw_ball(overlay_frame, center, radius)

                        # 공 깊이(Zcam) 추정으로 카메라 좌표계 3D 복원(근사), 이후 마커 좌표계로 변환
                        estimated_Z = (camera_matrix[0, 0] * BALL_RADIUS_REAL) / max(1e-6, radius)
                        ball_3d_cam = np.array([
                            (center[0] - camera_matrix[0,2]) * estimated_Z / camera_matrix[0,0],
                            (center[1] - camera_matrix[1,2]) * estimated_Z / camera_matrix[1,1],
                            estimated_Z
                        ], dtype=np.float32)

                        # 칼만 필터 업데이트(카메라 좌표계 측정 사용)
                        filtered_point_kalman = kalman_tracker.update_with_gating(ball_3d_cam)

                        # 마커 좌표계로 변환
                        filtered_point = aruco_detector.point_to_marker_coord(np.array(filtered_point_kalman, dtype=np.float32), rvec, tvec)

                        # 궤적 기록
                        if record_trajectory:
                            pitch_points_trace_3d_x.append(filtered_point[0])
                            pitch_points_trace_3d_y.append(filtered_point[1])
                            pitch_points_trace_3d_z.append(filtered_point[2])

                        # 평면 정의(연속 코너 3개: 0,1,2)
                        p1_0, p1_1, p1_2 = ball_zone_corners[0], ball_zone_corners[1], ball_zone_corners[2]
                        p2_0, p2_1, p2_2 = ball_zone_corners2[0], ball_zone_corners2[1], ball_zone_corners2[2]

                        # 서명거리 (법선 +Y로 정렬)
                        d1 = aruco_detector.signed_distance_to_plane_oriented(filtered_point, p1_0, p1_1, p1_2, desired_dir=np.array([0,1,0], dtype=np.float32))
                        d2 = aruco_detector.signed_distance_to_plane_oriented(filtered_point, p2_0, p2_1, p2_2, desired_dir=np.array([0,1,0], dtype=np.float32))



                        # 고해상도 타임스탬프
                        now_perf = time.perf_counter()
                        
                        
                         # 디버그 모드: 'b' 키 눌렀을 때 정보 출력
                        if debug_mode:
                            print("\n" + "="*60)
                            print("🔍 공 위치 디버그 정보")
                            print("="*60)
                            print(f"📍 공 3D 좌표 (마커 좌표계):")
                            print(f"   X = {filtered_point[0]:+.4f} m (좌우)")
                            print(f"   Y = {filtered_point[1]:+.4f} m (깊이/전후)")
                            print(f"   Z = {filtered_point[2]:+.4f} m (높이)")
                            print(f"\n📏 평면과의 거리:")
                            print(f"   전면 판정면 (Y=0.00)까지:      d1 = {d1:+.4f} m")
                            print(f"   후면 판정면 (Y={ZONE_DEPTH:.2f})까지: d2 = {d2:+.4f} m")
                            print(f"\n📊 스트라이크존 범위:")
                            print(f"   X: {BOX_MIN[0]:.3f} ~ {BOX_MAX[0]:.3f} m")
                            print(f"   Y: {BOX_Y_MIN:.3f} ~ {BOX_Y_MAX:.3f} m")
                            print(f"   Z: {BOX_Z_MIN:.3f} ~ {BOX_Z_MAX:.3f} m")
                            
                            # 존 내부 판정
                            in_x_range = BOX_MIN[0] <= filtered_point[0] <= BOX_MAX[0]
                            in_y_range = BOX_Y_MIN <= filtered_point[1] <= BOX_Y_MAX
                            in_z_range = BOX_Z_MIN <= filtered_point[2] <= BOX_Z_MAX
                            
                            print(f"\n✅ 존 내부 판정:")
                            print(f"   X 범위: {'✓ IN' if in_x_range else '✗ OUT'}")
                            print(f"   Y 범위: {'✓ IN' if in_y_range else '✗ OUT'}")
                            print(f"   Z 범위: {'✓ IN' if in_z_range else '✗ OUT'}")
                            print(f"   종합: {'🎯 STRIKE' if (in_x_range and in_y_range and in_z_range) else '⚾ BALL'}")
                            
                            print(f"\n🔄 교차 상태:")
                            print(f"   전면 교차: {t_cross_plane1 is not None} (시각: {t_cross_plane1 if t_cross_plane1 else 'None'})")
                            print(f"   후면 교차: {t_cross_plane2 is not None} (시각: {t_cross_plane2 if t_cross_plane2 else 'None'})")
                            print("="*60 + "\n")
                            
                            debug_mode = False  # 한 번만 출력

                        # 교차 시각 보간(+ → 0 → −)
                        if prev_time_perf is not None:
                            if prev_distance_to_plane1 is not None and (prev_distance_to_plane1 > 0.0) and (d1 <= 0.0):
                                alpha1 = prev_distance_to_plane1 / (prev_distance_to_plane1 - d1 + 1e-9)
                                t_cross_plane1 = prev_time_perf + alpha1 * (now_perf - prev_time_perf)
                            if prev_distance_to_plane2 is not None and (prev_distance_to_plane2 > 0.0) and (d2 <= 0.0):
                                alpha2 = prev_distance_to_plane2 / (prev_distance_to_plane2 - d2 + 1e-9)
                                t_cross_plane2 = prev_time_perf + alpha2 * (now_perf - prev_time_perf)

                        # 두 평면 교차 완료 시 깊이 속도 계산
                        # 참고: config에 기존 ZONE_Z_DIFF가 있다면 ZONE_DEPTH로 통일
                        if (t_cross_plane1 is not None) and (t_cross_plane2 is not None):
                            dt = max(1e-6, (t_cross_plane2 - t_cross_plane1))
                            v_depth_mps = ZONE_DEPTH / dt
                            v_kmh = v_depth_mps * 3.6

                            final_velocity = v_kmh
                            display_velocity = v_kmh

                            pitch_speeds.append(v_kmh)
                            pitch_results.append("속도측정")

                            # 기록지 포인트(깊이면 통과 시점의 x,z를 쓰는 게 가장 직관적이지만,
                            # 여기서는 현재 순간의 중심을 기록해도 충분)
                            record_sheet_x.append(filtered_point[0])
                            record_sheet_z.append(filtered_point[2])

                            # 교차 시각 리셋(한 번 계산)
                            t_cross_plane1 = None
                            t_cross_plane2 = None

                        # 이전 값 업데이트
                        prev_distance_to_plane1 = d1
                        prev_distance_to_plane2 = d2
                        prev_time_perf = now_perf

                        # 대시보드 업데이트
                        current_dashboard_data = {
                            'record_sheet_points': list(zip(record_sheet_x, record_sheet_z)),
                            'record_sheet_polygon': [[p[0], p[2]] for p in ball_zone_corners2] if ids is not None else [],
                            'trajectory_3d': list(zip(pitch_points_trace_3d_x, pitch_points_trace_3d_y, pitch_points_trace_3d_z)),
                            'strike_zone_corners_3d': ball_zone_corners.tolist(),
                            'ball_zone_corners_3d': ball_zone_corners.tolist(),
                            'ball_zone_corners2_3d': ball_zone_corners2.tolist(),
                            'box_corners_3d': box_corners_3d.tolist(),
                            'pitch_count': len(pitch_speeds),
                            'strike_count': 0,
                            'ball_count': 0,
                            'pitch_speeds': pitch_speeds,
                            'pitch_results': pitch_results,
                            'pitch_history': pitch_history
                        }
                        dashboard.update_data(current_dashboard_data)

        # FPS 계산/표시
        now = time.time()
        delta_time = now - fps_start_time
        if delta_time > 0:
            capture_fps = 1.0 / delta_time
        if now - last_fps_update_time >= 0.5:
            display_fps_value = capture_fps
            last_fps_update_time = now
        fps_start_time = now

        # 오버레이 표시
        cv2.putText(overlay_frame, f"FPS: {display_fps_value:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if display_velocity > 0:
            cv2.putText(overlay_frame, f"{display_velocity:.1f} km/h",
                        (10, overlay_frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('ARUCO Tracker with Strike Zone', overlay_frame)
        if is_video_mode:
            cv2.imshow('Original', frame)
            current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Progress", "Original", current_frame_no)

        
                    # 창 닫힘 감지
        if cv2.getWindowProperty('ARUCO Tracker with Strike Zone', cv2.WND_PROP_VISIBLE) < 1 \
               or (is_video_mode and cv2.getWindowProperty('Original', cv2.WND_PROP_VISIBLE) < 1):
                shutdown_event.set()
                break


        # 키 입력
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            shutdown_event.set()
            break
        
        elif key & 0xFF == ord('b'):
            # 디버그 모드 활성화 (다음 프레임에서 한 번만 출력)
            debug_mode = True
            print("\n[디버그 모드 활성화] 다음 공 검출 시 정보를 출력합니다...")


        elif key & 0xFF == ord('r'):
            # 상태 초기화
            ar_started = False
            scoreboard.reset()
            pitch_speeds.clear()
            pitch_results.clear()
            pitch_history.clear()
            record_sheet_x.clear()
            record_sheet_z.clear()
            pitch_points_trace_3d_x.clear()
            pitch_points_trace_3d_y.clear()
            pitch_points_trace_3d_z.clear()
            display_velocity = 0
            final_velocity = 0
            prev_distance_to_plane1 = None
            prev_distance_to_plane2 = None
            prev_time_perf = None
            t_cross_plane1 = None
            t_cross_plane2 = None
            print("상태 초기화")
        elif is_video_mode and key & 0xFF == ord(' '):
            play_pause = not play_pause
            print("일시정지" if play_pause else "재개")

    # 자원 해제
    if 'cap' in locals() and cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    logging.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    main()