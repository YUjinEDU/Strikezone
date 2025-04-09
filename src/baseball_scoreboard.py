import cv2
import numpy as np

class BaseballScoreboard:
    def __init__(self, width=3.0, height=2.0, offset_x=0.0, offset_y=1.0, offset_z=1.5):
        self.width = width
        self.height = height
        self.offset = np.array([offset_x, offset_y, offset_z]) # 마커 중심 기준 오프셋

        # 카운트 초기화
        self.strike_count = 0
        self.ball_count = 0
        self.out_count = 0

        # 색상 설정
        self.bg_color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.border_color = (50, 50, 50)
        self.inactive_color = (20, 20, 20)
        self.ball_color = (0, 200, 0)
        self.strike_color = (0, 215, 255)
        self.out_color = (0, 0, 255)

        # 내부 요소 정보
        self.labels = ["B", "S", "O"]
        self.max_counts = [4, 3, 3]
        self.colors = [self.ball_color, self.strike_color, self.out_color]

        # --- 3D 좌표 계산 ---
        # 전광판 모서리 3D 좌표 (마커 기준 + 오프셋 적용됨)
        self.corners_3d = self._calculate_corners_3d()
        # 내부 요소(레이블, 도트)의 3D 좌표 계산 (마커 기준 + 오프셋 적용됨)
        self.label_positions_3d, self.dot_positions_3d = self._calculate_element_positions_3d()
        # --------------------

    def _calculate_corners_3d(self):
        """전광판 모서리의 3D 좌표 계산"""
        half_width = self.width / 2
        half_height = self.height / 2
        # 좌표계: X(너비), Y(두께/앞뒤), Z(높이/위아래) 가정
        corners = np.array([
            [-half_width, 0,  half_height], # 좌상단 (Z가 양수일 때 위쪽)
            [ half_width, 0,  half_height], # 우상단
            [ half_width, 0, -half_height], # 우하단
            [-half_width, 0, -half_height], # 좌하단
        ], dtype=np.float32)
        corners += self.offset
        return corners

    def _calculate_element_positions_3d(self):
        """레이블과 도트 중심의 3D 좌표 계산"""
        label_positions_3d = []
        dot_positions_3d = [[] for _ in range(len(self.labels))]

        # 비율 설정
        label_area_ratio = 0.25
        dot_area_ratio = 1.0 - label_area_ratio
        dot_area_padding_ratio = 0.1 # 도트 영역 좌우 여백 비율
        max_dots_per_row = 4 # B 기준 최대 도트 수

        # 전광판 로컬 좌표계 기준 계산 (Y=0 평면, 중심이 0,0,0)
        label_x_local = -self.width / 2 + (self.width * label_area_ratio) / 2 # 왼쪽 영역 가로 중앙
        dot_area_start_x_local = -self.width / 2 + self.width * label_area_ratio # 오른쪽 영역 시작 X
        dot_area_width_local = self.width * dot_area_ratio
        dot_start_x_local = dot_area_start_x_local + dot_area_width_local * dot_area_padding_ratio
        dot_effective_width = dot_area_width_local * (1.0 - 2 * dot_area_padding_ratio) # 실제 도트 배치 가능 너비

        # 도트 간 간격 계산 (최대 도트 수 기준)
        # 간격(N-1개) + 도트지름(N개) = 유효너비. 지름=간격이라 가정하면 간격*(2N-1)=유효너비
        dot_spacing_x = dot_effective_width / max(1, (max_dots_per_row * 2 - 1)) # 최대 4개 기준 간격
        # 실제 반지름을 이 간격의 절반으로 사용 가능
        # dot_radius_3d = dot_spacing_x / 2.0

        for i in range(len(self.labels)):
            # 1. 레이블 위치 (세로 3등분 지점)
            # Z 좌표: 위쪽이 양수. 전광판 상단(height/2)에서 아래로 내려옴
            label_z_local = self.height / 2 - self.height * ((i + 0.5) / len(self.labels))
            label_pos = np.array([label_x_local, 0, label_z_local], dtype=np.float32)
            label_positions_3d.append(label_pos + self.offset) # 오프셋 적용

            # 2. 도트 위치 (해당 레이블과 같은 높이, 오른쪽 영역에 가로 배치)
            dot_z_local = label_z_local
            max_count = self.max_counts[i] # 해당 행의 최대 도트 수

            # 해당 행의 도트들을 중앙 정렬하기 위한 시작 오프셋
            current_row_width = dot_spacing_x * max(1, (max_count * 2 - 1))
            start_offset_x = (dot_effective_width - current_row_width) / 2
            current_row_start_x = dot_start_x_local + start_offset_x

            for j in range(max_count):
                # 도트 중심 X 좌표 계산 (간격 * 2 = 도트 지름 + 간격)
                dot_x_local = current_row_start_x + dot_spacing_x * (j * 2 + 0.5) # 각 도트 중심의 X
                dot_pos = np.array([dot_x_local, 0, dot_z_local], dtype=np.float32)
                dot_positions_3d[i].append(dot_pos + self.offset) # 오프셋 적용

        return label_positions_3d, dot_positions_3d

    # reset, add_strike, add_ball, add_out 메서드는 동일 (이전 코드 사용)
    def reset(self):
        self.strike_count = 0
        self.ball_count = 0
        self.out_count = 0
    def add_strike(self):
        self.strike_count += 1
        if self.strike_count >= 3:
            self.strike_count = 0
            self.ball_count = 0
            self.add_out()
            return True
        return False
    def add_ball(self):
        self.ball_count += 1
        if self.ball_count >= 4:
            self.strike_count = 0
            self.ball_count = 0
            return True
        return False
    def add_out(self):
        self.out_count += 1
        if self.out_count >= 3:
            self.reset()
            return True
        return False

    def draw(self, frame, aruco_detector, rvec, tvec):
        """
        전광판 그리기 (3D 좌표 투영 기반)
        """
        try:
            # 1. 배경 그리기 (3D 모서리 투영)
            projected_corners = aruco_detector.project_points(self.corners_3d, rvec, tvec)
            if projected_corners is None or np.any(np.isnan(projected_corners)) or not np.all(np.isfinite(projected_corners)):
                return frame

            pts = projected_corners.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(frame, [pts], self.bg_color)
            cv2.polylines(frame, [pts], True, self.border_color, 2)

            # (선택적) 투영된 크기 계산 (폰트/도트 크기 결정 위해)
            tl = projected_corners[0].ravel()
            tr = projected_corners[1].ravel()
            bl = projected_corners[3].ravel()
            proj_width = max(1, int(np.linalg.norm(tr - tl)))
            proj_height = max(1, int(np.linalg.norm(bl - tl)))

            if proj_width <= 10 or proj_height <= 10:
                return frame

            # --- 크기 설정 (투영된 크기 기반) ---
            font_scale = proj_height / 150
            font_thickness = max(1, int(proj_width / 100))
            # 도트 반지름 계산 (3D 간격 기반 또는 투영 크기 기반)
            # 예: 투영된 너비와 도트 개수 기반
            max_dots_per_row = 4
            dot_area_width_proj = proj_width * (1.0 - 0.25) # 대략적인 오른쪽 영역 너비
            dot_effective_width_proj = dot_area_width_proj * (1.0 - 2 * 0.1)
            dot_spacing_proj = dot_effective_width_proj / max(1, (max_dots_per_row * 2 - 1))
            dot_radius = max(2, int(dot_spacing_proj * 0.8)) # 간격보다 약간 작게
            # ---------------------------------

            # 2. 레이블 그리기 (3D 위치 투영)
            projected_label_centers = aruco_detector.project_points(np.array(self.label_positions_3d), rvec, tvec)
            if projected_label_centers is not None:
                for i, label in enumerate(self.labels):
                    if i < len(projected_label_centers): # 투영 성공 확인
                        label_center_2d = projected_label_centers[i].ravel().astype(int)
                        # 텍스트 중앙 정렬
                        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        label_x_draw = label_center_2d[0] - text_w // 2
                        label_y_draw = label_center_2d[1] + text_h // 2
                        cv2.putText(frame, label, (label_x_draw, label_y_draw), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.text_color, font_thickness, cv2.LINE_AA)

            # 3. 도트 그리기 (3D 위치 투영)
            counts = [self.ball_count, self.strike_count, self.out_count]
            for i in range(len(self.labels)): # 각 행 (B, S, O)
                row_dot_positions_3d = self.dot_positions_3d[i]
                projected_dot_centers = aruco_detector.project_points(np.array(row_dot_positions_3d), rvec, tvec)

                if projected_dot_centers is not None:
                    count = counts[i]
                    color = self.colors[i]
                    max_count = self.max_counts[i]
                    for j in range(max_count): # 각 도트
                         if j < len(projected_dot_centers): # 투영 성공 확인
                            dot_center_2d = projected_dot_centers[j].ravel().astype(int)
                            dot_color = color if j < count else self.inactive_color
                            cv2.circle(frame, tuple(dot_center_2d), dot_radius, dot_color, -1)

        except Exception as e:
            print(f"Error during drawing scoreboard: {e}")

        return frame