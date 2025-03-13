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
        """마커 포즈 추정"""
        if len(corners) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            return rvecs, tvecs
        return None, None
    
    def draw_axes(self, frame, rvec, tvec, size=0.05):
        """마커 축 그리기"""
        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, size)
    
    def project_points(self, points_3d, rvec, tvec):
        """3D 점들을 영상 평면에 투영"""
        projected_points, _ = cv2.projectPoints(
            points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        return projected_points.reshape(-1, 2).astype(int)
    
    def get_box_corners_3d(self, box_min, box_max):
        """3D 박스 코너 좌표 계산"""
        x0, y0, z0 = box_min
        x1, y1, z1 = box_max

        # 박스 8개 꼭짓점
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

        # 삼각형 꼭짓점 좌표계를 [x, -z, y]로 변환했기 때문에 z0이 1번쨰에 있음
        triangle_points = np.array([
            [(x0+x1)/2, z0, y1+box_min[2]],  # 8번
            [(x0+x1)/2, z0, y1+box_max[2]], # 9번
        ], dtype=np.float32)

        # 합치기
        corners = np.concatenate((corners, triangle_points), axis=0)
        return corners
    
    def draw_3d_box(self, frame, pts2d, box_edges, color=(0,0,0), thickness=2):
        """3D 박스 그리기"""
        try:
            pts = pts2d.astype(np.int32)  # 명시적으로 int32로 변환
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
        """점이 다각형 내부에 있는지 확인"""
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0
    
    def signed_distance_to_plane(self, point, plane_point1, plane_point2, plane_point3):
        """점과 평면 사이의 부호 있는 거리 계산"""
        v1 = plane_point2 - plane_point1
        v2 = plane_point3 - plane_point1
        n = np.cross(v1, v2)
        n = n / np.linalg.norm(n)
        return np.dot(point - plane_point1, n)
    
    def project_point_onto_plane(self, point, plane_point1, plane_point2, plane_point3):
        """
        point: 평면으로부터 거리를 계산할 점
        plane_point1, plane_point2, plane_point3: 평면을 정의하는 3개의 점
        반환: 부호 있는 점-평면 거리
        """
        v1 = plane_point2 - plane_point1
        v2 = plane_point3 - plane_point1
        n = np.cross(v1, v2)
        n = n / np.linalg.norm(n)
        distance = np.dot(point - plane_point1, n)
        point_proj = point - distance * n
        return point_proj
    
    def point_to_marker_coord(self, point_3d_cam, rvec, tvec):
        """카메라 좌표계의 점을 마커 좌표계로 변환"""
        R_marker, _ = cv2.Rodrigues(rvec)
        point_in_marker_coord = np.dot(
            R_marker.T,
            (point_3d_cam.reshape(3,1) - tvec.reshape(3,1))
        ).T[0]
        return point_in_marker_coord
