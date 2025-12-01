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
from tracker_v1 import KalmanTracker
from hybrid_detector import HybridDetector  # 하이브리드 탐지기
from effects import TextEffect
from dashboard import Dashboard
from kalman_filter import KalmanFilter3D
from baseball_scoreboard import BaseballScoreboard

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class ClipRecorder:
    """
    이벤트 발생 시: pre_frames(사전 버퍼) + post_frames(사후 프레임)를 결합해
    비동기로 MP4 저장하는 도우미.
    - start(pre_frames, out_path, fps, post_seconds): 활성화
    - feed(frame): 활성 상태일 때 사후 프레임을 수집
    - 내부에서 사후 프레임 수집이 완료되면 별도 쓰레드에서 파일 저장
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.pre_frames = []
        self.post_frames = []
        self.post_needed = 0
        self.collected = 0
        self.out_path = ""
        self.fps = 60.0
        self.writer_thread = None

    def start(self, pre_frames, out_path, fps, post_seconds=1.0):
        with self.lock:
            if self.active:
                # 이전 작업이 남아있어도 새 작업으로 덮어씀
                pass
            self.pre_frames = list(pre_frames) if pre_frames else []
            self.post_frames = []
            self.post_needed = int(max(1, fps * post_seconds))
            self.collected = 0
            self.out_path = out_path
            self.fps = fps
            self.active = True
            self.writer_thread = None

    def feed(self, frame):
        with self.lock:
            if not self.active:
                return
            # 사후 프레임 수집
            self.post_frames.append(frame.copy())
            self.collected += 1
            if self.collected >= self.post_needed:
                # 저장용 데이터 스냅샷
                frames_to_save = self.pre_frames + self.post_frames
                out_path = self.out_path
                fps = self.fps
                # 비동기 저장
                def _writer(frames, path, fps_val):
                    if not frames:
                        return
                    h, w = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    vw = cv2.VideoWriter(path, fourcc, fps_val, (w, h))
                    for f in frames:
                        vw.write(f)
                    vw.release()
                self.writer_thread = threading.Thread(target=_writer, args=(frames_to_save, out_path, fps), daemon=True)
                self.writer_thread.start()
                # 비활성화
                self.active = False
                self.pre_frames = []
                self.post_frames = []
                self.post_needed = 0
                self.collected = 0
                self.out_path = ""

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
    ar_started = False

    # 교차 거리/시간(고정밀) 추적 변수
    prev_distance_to_plane1 = None
    prev_distance_to_plane2 = None
    prev_time_perf = None
    t_cross_plane1 = None
    t_cross_plane2 = None
    
    # 투구 상태 추적 (카메라 위치에 관계없이 동작)
    plane1_crossed = False      # plane1 통과 여부
    plane2_crossed = False      # plane2 통과 여부 (새로 추가)
    plane1_in_zone = False      # plane1 통과 시 스트라이크존 내부 여부
    plane2_in_zone = False      # plane2 통과 시 스트라이크존 내부 여부 (새로 추가)
    cross_point_p1_saved = None # plane1 통과 지점 저장
    cross_point_p2_saved = None # plane2 통과 지점 저장 (새로 추가)
    first_plane_time = None     # 먼저 통과한 평면 시간 (카메라 위치 무관)
    
    # 판정 쿨다운 (연속 판정 방지)
    last_judgment_time = 0.0    # 마지막 판정 시간
    judgment_cooldown = 2.0     # 판정 쿨다운 (초)

    # FPS 표시용
    fps_start_time = time.time()
    last_fps_update_time = time.time()
    display_fps_value = 0.0

    # 속도/기록 관련
    pitch_speeds = []
    pitch_results = []
    pitch_history = []
    display_velocity = 0.0
    final_velocity = 0.0

    # 손 감지/텍스트 효과
    # hand_detector = HandDetector()
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

    # 공 검출기 (하이브리드: FMO + Color)
    ball_detector = HybridDetector(GREEN_LOWER, GREEN_UPPER)

    # 대시보드
    dashboard = Dashboard()
    dashboard_thread = dashboard.run_server()

    # 스코어보드(AR)
    scoreboard = BaseballScoreboard(width=0.3, height=0.24, offset_x=-0.4, offset_y=0.0, offset_z=0.1)

    # 초기 대시보드 데이터 (plane2 = 뒤 판정면)
    initial_dashboard_data = {
        'record_sheet_points': [],
        'record_sheet_polygon': [[p[0], p[2]] for p in BALL_ZONE_CORNERS],  # 최초엔 앞면으로 초기화, 이후 갱신됨
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

    # 프레임 간 속도 측정용
    prev_ball_pos_marker = None  # 이전 프레임 공 위치 (마커 좌표계)
    prev_ball_time = None        # 이전 프레임 시간
    frame_speed_buffer = []      # 최근 N프레임 속도 저장 (이동평균용)
    MAX_SPEED_BUFFER = 5         # 이동평균 프레임 수

    # 박스/존 코너(마커 좌표계 그대로 사용)
    box_corners_3d = aruco_detector.get_box_corners_3d(BOX_MIN, BOX_MAX)
    ball_zone_corners = BALL_ZONE_CORNERS.copy()       # plane1 (앞)
    ball_zone_corners2 = BALL_ZONE_CORNERS.copy()      # plane2 (뒤)
    ball_zone_corners2[:, 1] += ZONE_DEPTH             # Y(깊이)+

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
            if not is_video_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if undistort_enabled and frame_w > 0 and frame_h > 0:
        new_K, _roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (frame_w, frame_h), alpha=0)
        undistort_map1, undistort_map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_K, (frame_w, frame_h), cv2.CV_16SC2
        )

    # 2D 기록지 데이터 (plane2 기준으로 표시)
    record_sheet_points_xz = []  # [(x,z), ...]

    # ==== 영상 클립(사전/사후) 저장 준비 ====
    clips_dir = os.path.abspath('clips')  # 절대 경로로 설정
    ensure_dir(clips_dir)
    print(f"📂 클립 저장 디렉토리: {clips_dir}")
    
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    if cap_fps <= 1:
        cap_fps = 60.0
    pre_seconds = 1.5
    post_seconds = 1.0
    prebuffer = deque(maxlen=int(cap_fps * pre_seconds))
    clip_recorder = ClipRecorder()

    # 궤적 버퍼(마커 좌표계)
    traj_x, traj_y, traj_z = [], [], []
    
    # 영구 기록용: 각 투구의 plane2 투영 지점 저장
    impact_points_on_plane2 = []  # [{'point_3d': [x,y,z], 'number': n}, ...]

    # 루프
    while not shutdown_event.is_set():
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

        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 왜곡 보정
        if undistort_enabled and undistort_map1 is not None:
            frame = cv2.remap(frame, undistort_map1, undistort_map2, interpolation=cv2.INTER_LINEAR)

        analysis_frame = frame.copy()
        overlay_frame = frame.copy()

        # prebuffer와 clip_recorder에 overlay를 저장(AR 시각 포함 영상 저장 용이)
        prebuffer.append(overlay_frame.copy())
        if clip_recorder.active:
            clip_recorder.feed(overlay_frame)

        # 손 감지(현재는 항상 시작)
        # hand_detector.find_hands(frame)
        ar_started = True

        if ar_started:
            # ArUco 탐지
            corners, ids, rejected = aruco_detector.detect_markers(analysis_frame)

            if ids is not None:
                rvecs, tvecs = aruco_detector.estimate_pose(corners)

                for rvec, tvec in zip(rvecs, tvecs):
                    
                    aruco_detector.draw_axes(overlay_frame, rvec, tvec, size=0.1)
                    
                    # plane1/plane2 폴리곤 투영
                    projected_points = aruco_detector.project_points(ball_zone_corners, rvec, tvec)
                    cv2.polylines(overlay_frame, [projected_points], True, (0, 255, 255), 5)

                    projected_points2 = aruco_detector.project_points(ball_zone_corners2, rvec, tvec)
                    cv2.polylines(overlay_frame, [projected_points2], True, (255, 100, 0), 5)

                    # 박스 그리기
                    pts2d_box = aruco_detector.project_points(box_corners_3d, rvec, tvec)
                    aruco_detector.draw_3d_box(overlay_frame, pts2d_box, BOX_EDGES, color=(0,0,0), thickness=3)

                    # 스코어보드
                    scoreboard.draw(overlay_frame, aruco_detector, rvec, tvec)
                    
                    # ==== 기록된 투구들의 plane2 투영 지점 표시 ====
                    for impact in impact_points_on_plane2:
                        aruco_detector.draw_impact_point_on_plane(
                            overlay_frame,
                            impact['point_3d'],
                            ball_zone_corners2,
                            rvec, tvec,
                            circle_radius=12,
                            circle_color=(255, 255, 0),  # 노란색 원
                            circle_thickness=2,
                            number_text=impact['number'],
                            text_color=(255, 0, 0)  # 빨간색 번호
                        )
                    
                    # ==== 현재 궤적 3D 시각화 ====
                    if len(traj_x) >= 2:
                        trajectory_3d = [[traj_x[i], traj_y[i], traj_z[i]] for i in range(len(traj_x))]
                        aruco_detector.draw_trajectory_3d(
                            overlay_frame,
                            trajectory_3d,
                            rvec, tvec,
                            color=(0, 255, 0),  # 초록색 궤적
                            thickness=2
                        )

                    # 공 검출 (하이브리드: FMO + Color)
                    center, radius, detect_method = ball_detector.detect(analysis_frame)
                    if center and radius > 0.4:
                        # 공 표시 (탐지 방법에 따라 색상 구분)
                        ball_detector.draw_ball(overlay_frame, center, radius, detect_method)

                        # === 깊이 추정 보정 ===
                        # 기본 깊이 추정 (공 크기 기반)
                        estimated_Z_ball = (camera_matrix[0, 0] * BALL_RADIUS_REAL) / max(1e-6, radius)
                        
                        # ArUco 마커 기준 깊이 (tvec[2])를 참조하여 보정
                        marker_depth = float(tvec[2][0]) if tvec is not None else 1.0
                        
                        # 깊이 보정: 마커 깊이 근처로 제한 (마커 ±1m 범위)
                        min_depth = max(0.1, marker_depth - 1.0)
                        max_depth = marker_depth + 1.0
                        estimated_Z = np.clip(estimated_Z_ball, min_depth, max_depth)
                        
                        # 카메라 좌표계 3D 복원
                        ball_3d_cam = np.array([
                            (center[0] - camera_matrix[0,2]) * estimated_Z / camera_matrix[0,0],
                            (center[1] - camera_matrix[1,2]) * estimated_Z / camera_matrix[1,1],
                            estimated_Z
                        ], dtype=np.float32)

                        # 칼만 필터 업데이트
                        filtered_point_kalman = kalman_tracker.update_with_gating(ball_3d_cam)
                        filtered_point = aruco_detector.point_to_marker_coord(
                            np.array(filtered_point_kalman, dtype=np.float32), rvec, tvec
                        )
                        
                        # === 프레임 간 속도 계산 (마커 좌표계 Y축 = 깊이 방향) ===
                        now_time = time.perf_counter()
                        if prev_ball_pos_marker is not None and prev_ball_time is not None:
                            dt_frame = now_time - prev_ball_time
                            if dt_frame > 0.001:  # 1ms 이상일 때만
                                # Y축(깊이) 이동 거리로 속도 계산 (투구 방향)
                                dy = filtered_point[1] - prev_ball_pos_marker[1]
                                # 3D 이동 거리
                                dist_3d = np.linalg.norm(filtered_point - prev_ball_pos_marker)
                                
                                # 주로 Y축 이동이면 유효한 속도
                                if abs(dy) > 0.01:  # 1cm 이상 깊이 이동
                                    speed_mps = abs(dy) / dt_frame  # Y축 속도
                                    frame_speed_kmh = speed_mps * 3.6
                                    
                                    # 합리적 범위 (1~200 km/h)만 버퍼에 추가
                                    if 1.0 < frame_speed_kmh < 200.0:
                                        frame_speed_buffer.append(frame_speed_kmh)
                                        if len(frame_speed_buffer) > MAX_SPEED_BUFFER:
                                            frame_speed_buffer.pop(0)
                        
                        prev_ball_pos_marker = filtered_point.copy()
                        prev_ball_time = now_time
                        
                        # 이동평균 속도
                        if len(frame_speed_buffer) > 0:
                            realtime_speed_kmh = sum(frame_speed_buffer) / len(frame_speed_buffer)
                        else:
                            realtime_speed_kmh = 0.0
                        
                        # === 디버그: 실시간 좌표 + 속도 표시 ===
                        debug_text = f"X:{filtered_point[0]:.2f} Y:{filtered_point[1]:.2f} Z:{filtered_point[2]:.2f}"
                        cv2.putText(overlay_frame, debug_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        # 실시간 속도 표시
                        speed_text = f"Speed: {realtime_speed_kmh:.1f} km/h"
                        cv2.putText(overlay_frame, speed_text, (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        # 스트라이크존 범위 표시
                        zone_text = f"Zone: X[-0.15~0.15] Z[0.25~0.65]"
                        cv2.putText(overlay_frame, zone_text, (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # 궤적 기록
                        traj_x.append(filtered_point[0]); traj_y.append(filtered_point[1]); traj_z.append(filtered_point[2])

                        # 평면 정의(연속 코너 3개: 0,1,2)
                        p1_0, p1_1, p1_2 = ball_zone_corners[0],  ball_zone_corners[1],  ball_zone_corners[2]
                        p2_0, p2_1, p2_2 = ball_zone_corners2[0], ball_zone_corners2[1], ball_zone_corners2[2]

                        # 서명거리 (법선 +Y로 정렬)
                        d1 = aruco_detector.signed_distance_to_plane_oriented(filtered_point, p1_0, p1_1, p1_2, desired_dir=desired_depth_axis)
                        d2 = aruco_detector.signed_distance_to_plane_oriented(filtered_point, p2_0, p2_1, p2_2, desired_dir=desired_depth_axis)

                        # 고해상도 타임스탬프
                        now_perf = time.perf_counter()

                        # === 통과 감지 (같은 프레임에서 plane1, plane2 둘 다 통과할 수 있음) ===
                        crossed_p1 = False
                        crossed_p2 = False
                        cross_point_p1 = None
                        cross_point_p2 = None
                        alpha1 = 0.0
                        alpha2 = 0.0
                        
                        if prev_time_perf is not None and prev_distance_to_plane1 is not None:
                            # plane1 통과 감지: 양방향 통과 감지 (앞→뒤 또는 뒤→앞)
                            # 정방향: 이전 프레임에서 앞(+), 현재 프레임에서 뒤(-)
                            # 역방향: 이전 프레임에서 뒤(-), 현재 프레임에서 앞(+)
                            forward_cross = (prev_distance_to_plane1 > 0.0) and (d1 <= 0.0)
                            backward_cross = (prev_distance_to_plane1 < 0.0) and (d1 >= 0.0)
                            
                            if forward_cross or backward_cross:
                                crossed_p1 = True
                                alpha1 = abs(prev_distance_to_plane1) / (abs(prev_distance_to_plane1) + abs(d1) + 1e-9)
                                
                                if len(traj_x) >= 2:
                                    prev_pt = np.array([traj_x[-2], traj_y[-2], traj_z[-2]], dtype=np.float32)
                                    curr_pt = np.array([traj_x[-1], traj_y[-1], traj_z[-1]], dtype=np.float32)
                                    cross_point_p1 = prev_pt + alpha1 * (curr_pt - prev_pt)
                                else:
                                    cross_point_p1 = filtered_point.copy()
                        
                        if prev_time_perf is not None and prev_distance_to_plane2 is not None:
                            # plane2 통과 감지: 양방향 통과 감지
                            forward_cross2 = (prev_distance_to_plane2 > 0.0) and (d2 <= 0.0)
                            backward_cross2 = (prev_distance_to_plane2 < 0.0) and (d2 >= 0.0)
                            
                            if forward_cross2 or backward_cross2:
                                crossed_p2 = True
                                alpha2 = abs(prev_distance_to_plane2) / (abs(prev_distance_to_plane2) + abs(d2) + 1e-9)
                                
                                if len(traj_x) >= 2:
                                    prev_pt = np.array([traj_x[-2], traj_y[-2], traj_z[-2]], dtype=np.float32)
                                    curr_pt = np.array([traj_x[-1], traj_y[-1], traj_z[-1]], dtype=np.float32)
                                    cross_point_p2 = prev_pt + alpha2 * (curr_pt - prev_pt)
                                else:
                                    cross_point_p2 = filtered_point.copy()

                        # === plane1 통과 처리 ===
                        if crossed_p1 and (not plane1_crossed) and cross_point_p1 is not None:
                            plane1_crossed = True
                            t_cross_plane1 = prev_time_perf + alpha1 * (now_perf - prev_time_perf)
                            plane1_in_zone = aruco_detector.is_point_in_strike_zone_3d(cross_point_p1, ball_zone_corners)
                            cross_point_p1_saved = cross_point_p1.copy()
                            
                            # 첫 번째 통과 시간 기록 (카메라 위치 무관)
                            if first_plane_time is None:
                                first_plane_time = t_cross_plane1
                            
                            if plane1_in_zone:
                                print(f"[plane1 통과] X={cross_point_p1[0]:.3f}, Z={cross_point_p1[2]:.3f} → 스트라이크존 내부 ✓")
                            else:
                                print(f"[plane1 통과] X={cross_point_p1[0]:.3f}, Z={cross_point_p1[2]:.3f} → 스트라이크존 밖 ✗")

                        # === plane2 통과 처리 ===
                        if crossed_p2 and (not plane2_crossed) and cross_point_p2 is not None:
                            plane2_crossed = True
                            t_cross_plane2 = prev_time_perf + alpha2 * (now_perf - prev_time_perf)
                            plane2_in_zone = aruco_detector.is_point_in_strike_zone_3d(cross_point_p2, ball_zone_corners2)
                            cross_point_p2_saved = cross_point_p2.copy()
                            
                            # 첫 번째 통과 시간 기록 (카메라 위치 무관)
                            if first_plane_time is None:
                                first_plane_time = t_cross_plane2
                            
                            print(f"[plane2 통과] X={cross_point_p2[0]:.3f}, Z={cross_point_p2[2]:.3f} → 스트라이크존 내부: {plane2_in_zone}")
                        
                        # === 판정: 두 평면 모두 통과했을 때 (카메라 위치 무관) ===
                        current_time = time.time()
                        cooldown_passed = (current_time - last_judgment_time) >= judgment_cooldown
                        
                        if plane1_crossed and plane2_crossed and cooldown_passed:
                            # 스트라이크 조건: plane1 AND plane2 모두 스트라이크존 내부 통과
                            is_strike = plane1_in_zone and plane2_in_zone
                            
                            # 디버그 로그
                            print(f"[판정] plane1(존내부:{plane1_in_zone}), plane2(존내부:{plane2_in_zone})")
                            print(f"[판정] 결과: {'스트라이크' if is_strike else '볼'}")
                            
                            # 판정 시간 기록 (쿨다운 시작)
                            last_judgment_time = current_time
                            
                            # 속도 계산: 절대값 dt 사용 (카메라 위치 무관)
                            if t_cross_plane1 is not None and t_cross_plane2 is not None:
                                dt = abs(t_cross_plane2 - t_cross_plane1)  # 절대값!
                                
                                MAX_VALID_DT = 0.5  # 500ms 이상이면 왔다갔다 한 것
                                MIN_VALID_DT = 0.003  # 3ms 미만이면 비정상
                                
                                if dt < MIN_VALID_DT:
                                    print(f"[속도] dt 너무 짧음 ({dt*1000:.1f}ms) - 칼만 속도 사용")
                                    v_kmh = realtime_speed_kmh if realtime_speed_kmh > 0 else 0.0
                                elif dt > MAX_VALID_DT:
                                    print(f"[속도] dt 너무 김 ({dt*1000:.1f}ms) - 왔다갔다 감지, 칼만 속도 사용")
                                    v_kmh = realtime_speed_kmh if realtime_speed_kmh > 0 else 0.0
                                else:
                                    # 정상 범위: 평면간 거리 기반 속도 계산
                                    v_depth_mps = ZONE_DEPTH / dt
                                    v_kmh = v_depth_mps * 3.6
                                    
                                    if v_kmh > 200:
                                        print(f"[속도] 비정상 값 ({v_kmh:.1f} km/h) - 칼만 속도 사용")
                                        v_kmh = realtime_speed_kmh if realtime_speed_kmh > 0 else 0.0
                                    else:
                                        print(f"[속도] {v_kmh:.1f} km/h (dt={dt*1000:.1f}ms)")
                                
                                # 최종 속도 저장
                                if v_kmh > 0:
                                    final_velocity = v_kmh
                                    display_velocity = v_kmh
                            else:
                                v_kmh = realtime_speed_kmh if realtime_speed_kmh > 0 else 0.0
                            
                            if is_strike:
                                result_label = "스트라이크"
                                scoreboard.add_strike()
                                text_effect.add_strike_effect()
                            else:
                                result_label = "볼"
                                scoreboard.add_ball()
                                text_effect.add_ball_effect()

                            # 교차 지점의 3D 좌표(plane2 위로 투영)
                            # cross_point_p2_saved 사용 (저장된 통과 지점)
                            if cross_point_p2_saved is not None:
                                try:
                                    point_on_plane2 = aruco_detector.project_point_onto_plane(
                                        cross_point_p2_saved, p2_0, p2_1, p2_2
                                    )
                                except Exception:
                                    point_on_plane2 = cross_point_p2_saved.copy()
                            elif cross_point_p1_saved is not None:
                                # plane2 통과점이 없으면 plane1 통과점 사용
                                point_on_plane2 = cross_point_p1_saved.copy()
                            else:
                                # 둘 다 없으면 현재 위치 사용
                                point_on_plane2 = filtered_point.copy()

                            # ==== 영상 클립 저장(사전+사후 비동기) ====
                            ts = int(time.time())
                            clip_filename = f"pitch_{ts}.mp4"
                            clip_path = os.path.join(clips_dir, clip_filename)
                            clip_recorder.start(list(prebuffer), clip_path, cap_fps, post_seconds=post_seconds)
                            print(f"✅ 영상 클립 저장 시작: {clip_path}")

                            # ==== Dashboard에 피치 append ====
                            trajectory_points = list(zip(traj_x, traj_y, traj_z))
                            pitch_number = len(impact_points_on_plane2) + 1
                            
                            impact_points_on_plane2.append({
                                'point_3d': point_on_plane2,
                                'number': pitch_number
                            })
                            
                            dashboard.update_data({
                                'record_sheet_polygon': [[p[0], p[2]] for p in ball_zone_corners2],
                                'strike_zone_corners_3d': ball_zone_corners.tolist(),
                                'ball_zone_corners_3d': ball_zone_corners.tolist(),
                                'ball_zone_corners2_3d': ball_zone_corners2.tolist(),
                                'box_corners_3d': box_corners_3d.tolist(),
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
                            record_sheet_points_xz.append([float(point_on_plane2[0]), float(point_on_plane2[2])])

                            # 상태 리셋: 다음 투구
                            plane1_crossed = False
                            plane2_crossed = False
                            plane1_in_zone = False
                            plane2_in_zone = False
                            cross_point_p1_saved = None
                            cross_point_p2_saved = None
                            first_plane_time = None
                            t_cross_plane1 = None
                            t_cross_plane2 = None
                            prev_distance_to_plane1 = None
                            prev_distance_to_plane2 = None
                            prev_time_perf = None
                            traj_x.clear(); traj_y.clear(); traj_z.clear()
                            # 속도 버퍼 초기화
                            frame_speed_buffer.clear()
                            prev_ball_pos_marker = None
                            prev_ball_time = None
                            # 탐지기 연속성 리셋 (다음 공 탐지 준비)
                            ball_detector.reset()
                        
                        # === 한쪽 평면만 통과 후 옆으로 빠진 경우 → 볼 판정 ===
                        # 조건1: plane1만 통과하고 plane2를 지나침 (투수 옆 카메라)
                        # 조건2: plane2만 통과하고 plane1을 지나침 (포수 옆 카메라)
                        one_plane_only = (plane1_crossed and not plane2_crossed) or (plane2_crossed and not plane1_crossed)
                        passed_through = (d1 < -0.05) or (d2 < -0.05)  # 어느 한쪽이든 5cm 이상 지나침
                        
                        if one_plane_only and passed_through and cooldown_passed:
                            if plane1_crossed and not plane2_crossed:
                                print(f"[판정] plane1 통과 후 plane2 미통과 (옆으로 빠짐) → 볼")
                            else:
                                print(f"[판정] plane2 통과 후 plane1 미통과 (옆으로 빠짐) → 볼")
                            
                            # 판정 시간 기록 (쿨다운 시작)
                            last_judgment_time = current_time
                            
                            result_label = "볼"
                            scoreboard.add_ball()
                            text_effect.add_ball_effect()
                            v_kmh = 0.0
                            
                            # 마지막 위치를 기록
                            point_on_plane2 = filtered_point.copy()
                            
                            # ==== 영상 클립 저장 ====
                            ts = int(time.time())
                            clip_filename = f"pitch_{ts}.mp4"
                            clip_path = os.path.join(clips_dir, clip_filename)
                            clip_recorder.start(list(prebuffer), clip_path, cap_fps, post_seconds=post_seconds)
                            print(f"✅ 영상 클립 저장 시작: {clip_path}")

                            # ==== Dashboard에 피치 append ====
                            trajectory_points = list(zip(traj_x, traj_y, traj_z))
                            pitch_number = len(impact_points_on_plane2) + 1
                            
                            impact_points_on_plane2.append({
                                'point_3d': point_on_plane2,
                                'number': pitch_number
                            })
                            
                            dashboard.update_data({
                                'record_sheet_polygon': [[p[0], p[2]] for p in ball_zone_corners2],
                                'strike_zone_corners_3d': ball_zone_corners.tolist(),
                                'ball_zone_corners_3d': ball_zone_corners.tolist(),
                                'ball_zone_corners2_3d': ball_zone_corners2.tolist(),
                                'box_corners_3d': box_corners_3d.tolist(),
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
                            record_sheet_points_xz.append([float(point_on_plane2[0]), float(point_on_plane2[2])])

                            # 상태 리셋
                            plane1_crossed = False
                            plane2_crossed = False
                            plane1_in_zone = False
                            plane2_in_zone = False
                            cross_point_p1_saved = None
                            cross_point_p2_saved = None
                            first_plane_time = None
                            t_cross_plane1 = None
                            t_cross_plane2 = None
                            prev_distance_to_plane1 = None
                            prev_distance_to_plane2 = None
                            prev_time_perf = None
                            traj_x.clear(); traj_y.clear(); traj_z.clear()
                            # 속도 버퍼 초기화
                            frame_speed_buffer.clear()
                            prev_ball_pos_marker = None
                            prev_ball_time = None
                            # 탐지기 연속성 리셋 (다음 공 탐지 준비)
                            ball_detector.reset()

                        # 이전 값 업데이트
                        prev_distance_to_plane1 = d1
                        prev_distance_to_plane2 = d2
                        prev_time_perf = now_perf

                        # 대시보드 주기적 데이터(폴리곤, 박스 등) 갱신
                        dashboard.update_data({
                            'record_sheet_points': list(record_sheet_points_xz),  # (x,z)
                            'record_sheet_polygon': [[p[0], p[2]] for p in ball_zone_corners2],
                            'strike_zone_corners_3d': ball_zone_corners.tolist(),
                            'ball_zone_corners_3d': ball_zone_corners.tolist(),
                            'ball_zone_corners2_3d': ball_zone_corners2.tolist(),
                            'box_corners_3d': box_corners_3d.tolist(),
                            'pitch_speeds': pitch_speeds,
                            'pitch_results': pitch_results,
                            'pitch_history': pitch_history
                        })



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
        
        # 속도 표시 (측정된 경우)
        if display_velocity > 0:
            speed_text = f"{display_velocity:.1f} km/h"
            # 배경 박스
            text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            x_pos = overlay_frame.shape[1] - text_size[0] - 20  # 오른쪽 정렬
            y_pos = 50
            cv2.rectangle(overlay_frame, (x_pos - 10, y_pos - text_size[1] - 10), 
                         (x_pos + text_size[0] + 10, y_pos + 10), (0, 0, 0), -1)
            cv2.putText(overlay_frame, speed_text,
                        (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

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
        elif key & 0xFF == ord('r'):
            # 상태 초기화
            ar_started = False
            scoreboard.reset()
            pitch_speeds.clear()
            pitch_results.clear()
            pitch_history.clear()
            record_sheet_points_xz.clear()
            impact_points_on_plane2.clear()  # 충돌 지점 초기화
            traj_x.clear(); traj_y.clear(); traj_z.clear()
            display_velocity = 0.0
            final_velocity = 0.0
            prev_distance_to_plane1 = None
            prev_distance_to_plane2 = None
            prev_time_perf = None
            t_cross_plane1 = None
            t_cross_plane2 = None
            plane1_crossed = False
            plane1_in_zone = False
            cross_point_p1_saved = None
            last_judgment_time = 0.0  # 쿨다운 리셋
            print("상태 초기화")
        elif is_video_mode and key & 0xFF == ord(' '):
            play_pause = not play_pause
            print("일시정지" if play_pause else "재개")
        elif key & 0xFF == ord('f'):
            # FMO 토글
            fmo_state = ball_detector.toggle_fmo()
            print(f"FMO 탐지: {'활성화' if fmo_state else '비활성화'}")

    # 자원 해제
    if 'cap' in locals() and cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    logging.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    main()