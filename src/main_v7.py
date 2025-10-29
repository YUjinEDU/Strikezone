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
    zone_step1 = False  # plane1 통과 완료 여부

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
        hand_detector.find_hands(frame)
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

                    # 공 검출
                    center, radius, _ = ball_detector.detect(analysis_frame)
                    if center and radius > 0.4:
                        # 공 표시
                        ball_detector.draw_ball(overlay_frame, center, radius)

                        # 카메라 좌표계 3D 복원(근사) → 마커 좌표계로 변환
                        estimated_Z = (camera_matrix[0, 0] * BALL_RADIUS_REAL) / max(1e-6, radius)
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

                        # 궤적 기록
                        traj_x.append(filtered_point[0]); traj_y.append(filtered_point[1]); traj_z.append(filtered_point[2])

                        # 평면 정의(연속 코너 3개: 0,1,2)
                        p1_0, p1_1, p1_2 = ball_zone_corners[0],  ball_zone_corners[1],  ball_zone_corners[2]
                        p2_0, p2_1, p2_2 = ball_zone_corners2[0], ball_zone_corners2[1], ball_zone_corners2[2]

                        # 서명거리 (법선 +Y로 정렬)
                        d1 = aruco_detector.signed_distance_to_plane_oriented(filtered_point, p1_0, p1_1, p1_2, desired_dir=desired_depth_axis)
                        d2 = aruco_detector.signed_distance_to_plane_oriented(filtered_point, p2_0, p2_1, p2_2, desired_dir=desired_depth_axis)

                        # 2D 폴리곤 내부 여부(현재 프레임 중심 픽셀 기준)
                        in_poly1 = aruco_detector.is_point_in_polygon(center, projected_points)  if projected_points  is not None else False
                        in_poly2 = aruco_detector.is_point_in_polygon(center, projected_points2) if projected_points2 is not None else False

                        # 고해상도 타임스탬프
                        now_perf = time.perf_counter()

                        # plane1 통과(+ → 0 → −) + polygon 내부
                        if prev_time_perf is not None and prev_distance_to_plane1 is not None:
                            crossed_p1 = (prev_distance_to_plane1 > 0.0) and (d1 <= 0.0)
                            if (not zone_step1) and crossed_p1 and in_poly1:
                                # 보간으로 t_cross_plane1 갱신
                                alpha1 = prev_distance_to_plane1 / (prev_distance_to_plane1 - d1 + 1e-9)
                                t_cross_plane1 = prev_time_perf + alpha1 * (now_perf - prev_time_perf)
                                zone_step1 = True  # 1단계 통과 활성화

                        # plane2 통과(+ → 0 → −)
                        if prev_time_perf is not None and prev_distance_to_plane2 is not None:
                            crossed_p2 = (prev_distance_to_plane2 > 0.0) and (d2 <= 0.0)
                            if crossed_p2:
                                alpha2 = prev_distance_to_plane2 / (prev_distance_to_plane2 - d2 + 1e-9)
                                t_cross_plane2 = prev_time_perf + alpha2 * (now_perf - prev_time_perf)

                                if zone_step1 and (t_cross_plane1 is not None):
                                    # 속도 계산
                                    dt = max(1e-6, (t_cross_plane2 - t_cross_plane1))
                                    v_depth_mps = ZONE_DEPTH / dt
                                    v_kmh = v_depth_mps * 3.6
                                    final_velocity = v_kmh
                                    display_velocity = v_kmh

                                    # 스트/볼 판정: plane2 폴리곤 내부 여부로 결정
                                    if in_poly2:
                                        result_label = "스트라이크"
                                        scoreboard.add_strike()
                                        text_effect.add_strike_effect()
                                    else:
                                        result_label = "볼"
                                        scoreboard.add_ball()
                                        text_effect.add_ball_effect()

                                    # 교차 지점의 3D 좌표(plane2 위로 투영)
                                    # project_point_onto_plane는 aruco_detector에 있음
                                    # (repo의 정의에 따라 호출)
                                    try:
                                        # 일부 버전엔 project_point_onto_plane가 aruco_detector에 정의되어 있음
                                        point_on_plane2 = aruco_detector.project_point_onto_plane(
                                            filtered_point, p2_0, p2_1, p2_2
                                        )
                                    except Exception:
                                        # 정의가 없을 경우, 근사치로 현재 filtered_point 사용
                                        point_on_plane2 = filtered_point.copy()

                                    # ==== 영상 클립 저장(사전+사후 비동기) ====
                                    ts = int(time.time())
                                    clip_filename = f"pitch_{ts}.mp4"
                                    clip_path = os.path.join(clips_dir, clip_filename)
                                    # 사전 프레임 복사 후 recorder 시작
                                    clip_recorder.start(list(prebuffer), clip_path, cap_fps, post_seconds=post_seconds)
                                    print(f"✅ 영상 클립 저장 시작: {clip_path}")

                                    # ==== Dashboard에 피치 append ====
                                    trajectory_points = list(zip(traj_x, traj_y, traj_z))
                                    
                                    # 투구 번호
                                    pitch_number = len(impact_points_on_plane2) + 1
                                    
                                    # plane2에 투영된 충돌 지점 저장
                                    impact_points_on_plane2.append({
                                        'point_3d': point_on_plane2,
                                        'number': pitch_number
                                    })
                                    
                                    dashboard.update_data({
                                        # 2D 기록지는 plane2 폴리곤(x,z)
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
                                    # 2D 기록지 포인트에도 추가(x,z)
                                    record_sheet_points_xz.append([float(point_on_plane2[0]), float(point_on_plane2[2])])

                                    # 상태 리셋: 다음 투구
                                    zone_step1 = False
                                    t_cross_plane1 = None
                                    t_cross_plane2 = None
                                    prev_distance_to_plane1 = None
                                    prev_distance_to_plane2 = None
                                    prev_time_perf = None
                                    traj_x.clear(); traj_y.clear(); traj_z.clear()

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
            zone_step1 = False
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