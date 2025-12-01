import cv2
import cv2.aruco as aruco
import numpy as np

class ArucoDetector:
    def __init__(self, marker_size, camera_matrix, dist_coeffs):
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # ARUCO 사전 및 파라미터 설정
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
    
    def detect_markers(self, frame):
        """프레임에서 ArUco 마커 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected
    
    def estimate_pose(self, corners):
        """마커 포즈 추정 (OpenCV 4.8+ 호환)"""
        if len(corners) > 0:
            rvecs = []
            tvecs = []
            # 각 마커에 대해 solvePnP 사용 (새 OpenCV 버전 호환)
            obj_points = np.array([
                [-self.marker_size/2,  self.marker_size/2, 0],
                [ self.marker_size/2,  self.marker_size/2, 0],
                [ self.marker_size/2, -self.marker_size/2, 0],
                [-self.marker_size/2, -self.marker_size/2, 0]
            ], dtype=np.float32)
            
            for corner in corners:
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, corner, self.camera_matrix, self.dist_coeffs
                )
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
            
            if rvecs:
                return rvecs, tvecs
        return None, None
    
    def draw_axes(self, frame, rvec, tvec, size=0.05):
        """마커 좌표축 그리기"""
        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, size)
    
    def project_points(self, points_3d, rvec, tvec):
        """3D 점들을 영상 평면에 투영"""
        try:
            projected_points, _ = cv2.projectPoints(
                points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            if np.any(np.isnan(projected_points)) or not np.all(np.isfinite(projected_points)):
                print("Warning: NaN or Inf values detected in projection")
                projected_points = np.array([[[0, 0]] * len(points_3d)])
            return projected_points.reshape(-1, 2).astype(int)
        except Exception as e:
            print(f"투영 오류: {e}")
            return np.array([[0, 0]] * len(points_3d), dtype=int)

    @staticmethod
    def _plane_normal(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return (n / n_norm).astype(np.float32)

    def signed_distance_to_plane_oriented(self, point, plane_point1, plane_point2, plane_point3, desired_dir=None):
        """원하는 방향(desired_dir)으로 법선을 정렬한 서명거리"""
        n = self._plane_normal(plane_point1, plane_point2, plane_point3)
        if desired_dir is not None:
            d = desired_dir / (np.linalg.norm(desired_dir) + 1e-9)
            if float(np.dot(n, d)) < 0:
                n = -n
        return float(np.dot(point - plane_point1, n))

    def draw_plane_normal(self, frame, rvec, tvec, plane_point1, plane_point2, plane_point3, scale=0.2, color=(0,0,255)):
        """평면 법선을 영상에 화살표로 시각화(디버깅)"""
        n = self._plane_normal(plane_point1, plane_point2, plane_point3)
        p0 = plane_point1
        p1 = p0 + n * scale
        pts2d = self.project_points(np.array([p0, p1], dtype=np.float32), rvec, tvec)
        cv2.arrowedLine(frame, tuple(pts2d[0]), tuple(pts2d[1]), color, 2, tipLength=0.25)

    def get_box_corners_3d(self, box_min, box_max):
        """3D 박스 코너 좌표 계산 - [x,y,z] 한 좌표계만 사용"""
        x0, y0, z0 = box_min
        x1, y1, z1 = box_max

        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], dtype=np.float32)

        return corners
    
    def draw_3d_box(self, frame, pts2d, box_edges, color=(0,0,0), thickness=2):
        """3D 박스 그리기"""
        try:
            pts = pts2d.astype(np.int32)
            for e in box_edges:
                if 0 <= e[0] < len(pts) and 0 <= e[1] < len(pts):
                    pt1 = tuple(map(int, pts[e[0]]))
                    pt2 = tuple(map(int, pts[e[1]]))
                    cv2.line(frame, pt1, pt2, color, thickness)
        except Exception as ex:
            print(f"3D 박스 그리기 오류: {ex}")
    
    def draw_grid(self, frame, points, num_divisions):
        """사각형 내부에 격자 그리기"""
        for i in range(1, num_divisions):
            # 수평선
            pt1 = tuple((points[0] + (points[3] - points[0]) * i / num_divisions).astype(int))
            pt2 = tuple((points[1] + (points[2] - points[1]) * i / num_divisions).astype(int))
            cv2.line(frame, pt1, pt2, (0, 0, 0), 1)
            
            # 수직선
            pt1 = tuple((points[0] + (points[1] - points[0]) * i / num_divisions).astype(int))
            pt2 = tuple((points[3] + (points[2] - points[3]) * i / num_divisions).astype(int))
            cv2.line(frame, pt1, pt2, (0, 0, 0), 1)
    
    def is_point_in_polygon(self, point, polygon):
        """2D 점이 다각형 내부인지"""
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0
    
    def is_point_in_strike_zone_3d(self, point_3d, zone_corners):
        """
        3D 점이 스트라이크존 영역 내부에 있는지 판정 (X, Z 좌표 기준)
        
        Args:
            point_3d: 마커 좌표계의 3D 점 [x, y, z]
            zone_corners: 스트라이크존 4개 코너 [[x,y,z], ...]
                         순서: Bottom-left, Bottom-right, Top-right, Top-left
        
        Returns:
            bool: 스트라이크존 내부에 있으면 True
        """
        # 스트라이크존 경계 추출 (X, Z 좌표 사용)
        x_coords = zone_corners[:, 0]
        z_coords = zone_corners[:, 2]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        px, py, pz = point_3d[0], point_3d[1], point_3d[2]
        
        # X, Z 좌표가 스트라이크존 경계 내에 있는지 확인
        in_x = x_min <= px <= x_max
        in_z = z_min <= pz <= z_max
        
        return in_x and in_z
    
    def point_to_marker_coord(self, point_3d_cam, rvec, tvec):
        """카메라 좌표계의 점을 마커 좌표계로 변환"""
        R_marker, _ = cv2.Rodrigues(rvec)
        point_in_marker_coord = np.dot(
            R_marker.T,
            (point_3d_cam.reshape(3,1) - tvec.reshape(3,1))
        ).T[0]
        return point_in_marker_coord.astype(np.float32)
    
    def project_point_onto_plane(self, point, plane_point1, plane_point2, plane_point3):
        """3D 점을 평면에 수직 투영"""
        n = self._plane_normal(plane_point1, plane_point2, plane_point3)
        d = np.dot(point - plane_point1, n)
        projected_point = point - d * n
        return projected_point.astype(np.float32)
    
    def draw_trajectory_3d(self, frame, trajectory_points_3d, rvec, tvec, color=(0, 255, 0), thickness=2):
        """3D 궤적을 화면에 투영하여 선으로 그리기"""
        if len(trajectory_points_3d) < 2:
            return
        
        try:
            # 3D 점들을 2D로 투영
            points_array = np.array(trajectory_points_3d, dtype=np.float32)
            projected_2d = self.project_points(points_array, rvec, tvec)
            
            # 연결선 그리기
            for i in range(len(projected_2d) - 1):
                pt1 = (int(projected_2d[i][0]), int(projected_2d[i][1]))
                pt2 = (int(projected_2d[i + 1][0]), int(projected_2d[i + 1][1]))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
                
        except Exception as e:
            print(f"3D 궤적 그리기 오류: {e}")
    
    def draw_impact_point_on_plane(self, frame, point_3d, plane_corners, rvec, tvec, 
                                    circle_radius=15, circle_color=(255, 255, 0), 
                                    circle_thickness=3, number_text=None, text_color=(255, 0, 0)):
        """
        평면(plane_corners)에 투영된 점을 원으로 표시하고, 선택적으로 번호 표시
        
        Args:
            frame: 그릴 프레임
            point_3d: 3D 공간의 점 (마커 좌표계)
            plane_corners: 평면의 4개 코너 (마커 좌표계)
            rvec, tvec: 마커 포즈
            circle_radius: 원 반지름
            circle_color: 원 색상 (BGR)
            circle_thickness: 원 두께
            number_text: 표시할 번호 (None이면 표시 안함)
            text_color: 번호 색상 (BGR)
        """
        try:
            # 평면에 점 투영
            p0, p1, p2 = plane_corners[0], plane_corners[1], plane_corners[2]
            projected_point = self.project_point_onto_plane(point_3d, p0, p1, p2)
            
            # 2D 화면 좌표로 변환
            pts_2d = self.project_points(np.array([projected_point], dtype=np.float32), rvec, tvec)
            center_2d = tuple(pts_2d[0])
            
            # 원 그리기
            cv2.circle(frame, center_2d, circle_radius, circle_color, circle_thickness, cv2.LINE_AA)
            
            # 번호 표시
            if number_text is not None:
                text = str(number_text)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.48  # 0.8 * 0.6 = 0.48 (40% 축소)
                font_thickness = 1
                
                # 텍스트 크기 계산
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = center_2d[0] - text_size[0] // 2
                text_y = center_2d[1] + text_size[1] // 2
                
                # 텍스트 배경 (가독성)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                           (255, 255, 255), font_thickness + 1, cv2.LINE_AA)
                # 텍스트
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                           text_color, font_thickness, cv2.LINE_AA)
                
        except Exception as e:
            print(f"평면 충돌 지점 그리기 오류: {e}")