# -*- coding: utf-8 -*-
"""
GUI 위젯 모듈
2D 기록지, 스코어보드, 9분할 뷰 등의 커스텀 위젯
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QSizePolicy, QListWidget, QListWidgetItem,
    QScrollArea
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer, QPropertyAnimation, QVariantAnimation
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, 
    QPainterPath, QLinearGradient, QRadialGradient
)
import math

from gui_config import (
    record_config, scoreboard_config, 
    game_config, window_config
)


class RecordSheet2D(QWidget):
    """
    2D 기록지 위젯 (정면 시점 - 야구 중계 스타일)
    투수→포수 방향에서 바라본 시점으로 스트라이크 존 표시
    X = 좌우, Z = 높이, Y(깊이)는 투명도/크기로 표현
    """
    
    pitchSelected = pyqtSignal(int)  # 공 선택 시그널 (번호)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # records: [(x, z, is_strike, number, speed, trajectory_3d), ...]
        # trajectory_3d: [(x, y, z), ...] - 3D 좌표 전체
        self.records = []
        self.setMinimumSize(280, 360)  # 세로가 더 긴 비율
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 확장 가능하게
        
        # 표시 범위 (정규화용) - 스트라이크존이 세로로 길게 보이도록 설정
        # 실제 스트라이크존: 가로 0.3m, 세로 0.4m (세로가 1.33배 더 김)
        # 가로 범위를 더 넓게 잡아 스트라이크존이 세로로 길어 보이게 함
        self.zone_x_min = -0.5   # 미터 (좌우) - 더 넓게
        self.zone_x_max = 0.5
        self.zone_z_min = 0.0    # 미터 (높이)
        self.zone_z_max = 0.9
        
        # 깊이 범위 (Y축 - 투수→포수 방향)
        # 실제 데이터: Y가 음수일수록 투수 방향(멀리), Y가 0~0.2 근처가 포수/판정면
        # 로그에서 -2.7m까지 나와 여유 있게 확장
        self.depth_y_min = -3.0  # 투수 방향 (멀리)
        self.depth_y_max = 0.4   # 포수 방향 (plane2=0.2 기준 여유)
        
        # 실제 스트라이크 존 경계 (변경 없음)
        self.strike_x_min = -0.15
        self.strike_x_max = 0.15
        self.strike_z_min = 0.25
        self.strike_z_max = 0.65
        
        # 선택된 공 번호
        self.selected_pitch = None
        
        # 게임모드 타겟 구역
        self.target_zone = None
        
        # 궤적 표시 개수 (마지막 N개 포인트)
        self.trajectory_points_count = 15
        
        # 애니메이션 관련
        self.animation_progress = 1.0  # 0.0 ~ 1.0 (궤적 그리기 진행도)
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation)
        self.is_animating = False
        
    def add_record(self, x, z, is_strike, speed=None, trajectory=None):
        """기록 추가 (3D 궤적 포함)"""
        number = len(self.records) + 1
        
        # 좌표 유효성 검사 및 클램핑
        # 표시 범위: X(-0.5~0.5), Z(0.0~0.9)
        x_clamped = max(self.zone_x_min, min(self.zone_x_max, x))
        z_clamped = max(self.zone_z_min, min(self.zone_z_max, z))
        
        # 이상한 좌표 감지 및 로그
        if abs(x - x_clamped) > 0.01 or abs(z - z_clamped) > 0.01:
            print(f"[RecordSheet2D] 좌표 클램핑: ({x:.3f}, {z:.3f}) → ({x_clamped:.3f}, {z_clamped:.3f})")
        
        # trajectory: 3D 궤적 전체 저장 (x, y, z)
        traj_3d = []
        if trajectory and len(trajectory) > 0:
            # 마지막 N개 포인트만 사용
            n = min(self.trajectory_points_count, len(trajectory))
            for pt in trajectory[-n:]:
                if len(pt) >= 3:
                    tx, ty, tz = pt[0], pt[1], pt[2]
                    
                    # [중요] 땅(Z=0) 밑으로 가는 궤적은 0으로 클램핑
                    if tz < 0:
                        tz = 0
                    
                    # X는 약간의 여유를 두고 클램핑 (궤적이 존 밖으로 나갈 수 있음)
                    tx = max(self.zone_x_min - 0.1, min(self.zone_x_max + 0.1, tx))
                    # Z는 화면 범위로 클램핑
                    tz = max(self.zone_z_min, min(self.zone_z_max, tz))
                    
                    traj_3d.append((tx, ty, tz))  # x, y, z (y는 깊이)
                    
        self.records.append((x_clamped, z_clamped, is_strike, number, speed, traj_3d))
        
        # 최대 개수 초과시 오래된 것 제거
        if len(self.records) > record_config.MAX_DISPLAY_COUNT:
            self.records.pop(0)
            # 번호 재정렬
            for i, (x, z, is_s, _, spd, traj) in enumerate(self.records):
                self.records[i] = (x, z, is_s, i + 1, spd, traj)
        
        # 선택 해제 및 애니메이션 시작 (새 투구가 추가되면)
        self.selected_pitch = None
        self._start_animation()
        
        self.update()
        return number
        
    def clear_records(self):
        """기록 초기화"""
        self.records = []
        self.selected_pitch = None
        self.target_zone = None
        self.animation_progress = 1.0
        self.is_animating = False
        self.animation_timer.stop()
        self.update()
        
    def _start_animation(self):
        """궤적 애니메이션 시작"""
        self.animation_progress = 0.0
        self.is_animating = True
        self.animation_timer.start(record_config.TRAJECTORY_ANIMATION_SPEED)
        
    def _update_animation(self):
        """애니메이션 업데이트"""
        self.animation_progress += 0.08
        if self.animation_progress >= 1.0:
            self.animation_progress = 1.0
            self.is_animating = False
            self.animation_timer.stop()
        self.update()
        
    def set_target_zone(self, zone):
        """게임모드 타겟 구역 설정 (1~9, None이면 해제)"""
        self.target_zone = zone
        self.update()
        
    def select_pitch(self, number):
        """특정 투구 선택 (하이라이트)"""
        self.selected_pitch = number
        self.update()
        
    def _world_to_widget(self, x, z):
        """월드 좌표(정면 시점)를 위젯 좌표로 변환 (종횡비 유지)"""
        margin = record_config.MARGIN
        available_w = self.width() - 2 * margin
        available_h = self.height() - 2 * margin
        
        # 월드 좌표계의 범위
        world_w = self.zone_x_max - self.zone_x_min  # 0.8m
        world_h = self.zone_z_max - self.zone_z_min  # 1.07m
        world_aspect = world_h / world_w  # 세로/가로 비율 (1.33...)
        
        # 위젯의 종횡비
        widget_aspect = available_h / available_w
        
        # 종횡비 유지하면서 그리기 영역 계산
        if widget_aspect > world_aspect:
            # 위젯이 더 세로로 길다 → 가로 기준
            draw_w = available_w
            draw_h = available_w * world_aspect
            offset_x = 0
            offset_y = (available_h - draw_h) / 2
        else:
            # 위젯이 더 가로로 길다 → 세로 기준
            draw_h = available_h
            draw_w = available_h / world_aspect
            offset_x = (available_w - draw_w) / 2
            offset_y = 0
        
        # 정규화 (0~1)
        nx = (x - self.zone_x_min) / world_w
        nz = (z - self.zone_z_min) / world_h
        
        # 위젯 좌표 (Z는 높이이므로 위아래 반전)
        wx = margin + offset_x + nx * draw_w
        wy = margin + offset_y + (1 - nz) * draw_h
        
        return wx, wy
    
    def _perspective_transform(self, x, y, z):
        """3D 월드 좌표를 원근법 적용하여 2D 위젯 좌표로 변환
        
        투수 시점에서 캐처 방향을 바라보는 뷰:
        - X: 좌우 위치
        - Y: 깊이 (Y가 음수일수록 투수 방향/멀리, Y가 0에 가까울수록 포수 방향/가까이)
        - Z: 높이
        
        원근 투영:
        - 소실점: 화면 상단 중앙 (투수 방향, Y가 작은 쪽)
        - 스트라이크 존: 화면 하단 (포수 방향, Y가 큰 쪽)
        - Y가 작을수록(음수/투수쪽) 소실점에 가깝고 작게
        - Y가 클수록(0에 가까울수록/포수쪽) 스트라이크 존 원래 위치에 크게
        """
        margin = record_config.MARGIN
        w = self.width()
        h = self.height()
        
        # 스트라이크 존 중심 좌표 (위젯 좌표계)
        zone_center_x, zone_center_y = self._world_to_widget(
            (self.strike_x_min + self.strike_x_max) / 2,
            (self.strike_z_min + self.strike_z_max) / 2
        )
        
        # 소실점 (화면 상단 중앙 - 투수 방향)
        vanishing_x = w / 2
        vanishing_y = margin * 0.5  # 상단 마진 부근
        
        # Y 정규화: 0 = 투수쪽(멀리/소실점), 1 = 포수쪽(가까이/스트라이크존)
        # 실제 데이터에서 Y는 음수(투수)→0(포수) 방향으로 증가
        # depth_y_min=-3.0(투수/멀리), depth_y_max=0.4(포수/가까이)
        depth_range = self.depth_y_max - self.depth_y_min
        if depth_range < 0.01:
            depth_ratio = 1.0
        else:
            # 범위를 벗어난 Y는 클램프 후 정규화 (투수쪽=0, 포수쪽=1)
            y_clamped = max(self.depth_y_min, min(self.depth_y_max, y))
            depth_ratio = (y_clamped - self.depth_y_min) / depth_range
        
        # 원근 스케일: 가까울수록(depth_ratio=1) 크게, 멀수록(depth_ratio=0) 작게
        # 비선형 스케일로 더 자연스러운 원근감
        perspective_scale = 0.15 + 0.85 * (depth_ratio ** 0.7)
        
        # 기본 위젯 좌표 계산 (정면 시점)
        base_wx, base_wy = self._world_to_widget(x, z)
        
        # 스트라이크 존 중심 기준으로 스케일 적용
        scaled_x = zone_center_x + (base_wx - zone_center_x) * perspective_scale
        scaled_y_offset = (base_wy - zone_center_y) * perspective_scale
        
        # 소실점과 스트라이크 존 사이를 보간
        # depth_ratio만 쓰면 먼 공이 지나치게 위로 몰리므로 최소 가중치(0.35)를 준다.
        mix_ratio = 0.35 + 0.65 * depth_ratio  # 0.35~1.0
        final_x = vanishing_x + (scaled_x - vanishing_x) * mix_ratio

        # Y 좌표: 소실점에서 스트라이크 존 위치까지 보간
        base_y_at_zone = zone_center_y + scaled_y_offset
        final_y = vanishing_y + (base_y_at_zone - vanishing_y) * mix_ratio
        
        return final_x, final_y, perspective_scale
        
    def paintEvent(self, event):
        """그리기 이벤트"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        margin = record_config.MARGIN
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        # 배경 (다크 테마)
        painter.fillRect(self.rect(), QColor(*record_config.COLOR_BACKGROUND))
        
        # 외곽 테두리
        border_pen = QPen(QColor(60, 60, 70), 2)
        painter.setPen(border_pen)
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        
        # === 원근 가이드라인 (소실점에서 스트라이크 존으로) ===
        self._draw_perspective_guides(painter)
        
        # 스트라이크 존 경계 좌표
        zone_left, zone_top = self._world_to_widget(self.strike_x_min, self.strike_z_max)
        zone_right, zone_bottom = self._world_to_widget(self.strike_x_max, self.strike_z_min)
        zone_w = zone_right - zone_left
        zone_h = zone_bottom - zone_top
        
        # 스트라이크 존 배경 (다크 녹색 톤)
        zone_bg = QColor(*record_config.COLOR_ZONE_FILL, 150)
        painter.fillRect(int(zone_left), int(zone_top), int(zone_w), int(zone_h), zone_bg)
        
        # === 게임모드 타겟 구역 하이라이트 ===
        if self.target_zone is not None and 1 <= self.target_zone <= 9:
            self._draw_target_zone(painter, zone_left, zone_top, zone_w, zone_h)
        
        # 9분할 그리드 (다크 테마)
        grid_pen = QPen(QColor(*record_config.COLOR_GRID), 1, Qt.DashLine)
        painter.setPen(grid_pen)
        
        # 수직선 (3등분)
        for i in range(1, 3):
            x = zone_left + zone_w * i / 3
            painter.drawLine(int(x), int(zone_top), int(x), int(zone_bottom))
            
        # 수평선 (3등분)
        for i in range(1, 3):
            y = zone_top + zone_h * i / 3
            painter.drawLine(int(zone_left), int(y), int(zone_right), int(y))
        
        # 스트라이크 존 테두리 (시안색)
        zone_pen = QPen(QColor(*record_config.COLOR_ZONE_BORDER), 2)
        painter.setPen(zone_pen)
        painter.drawRect(int(zone_left), int(zone_top), int(zone_w), int(zone_h))
        
        # 구역 번호 표시 (다크 테마)
        font = QFont(window_config.FONT_FAMILY, 9)
        painter.setFont(font)
        painter.setPen(QColor(100, 100, 110))
        
        zone_positions = [
            (1, zone_left + zone_w/6, zone_top + zone_h/6),
            (2, zone_left + zone_w/2, zone_top + zone_h/6),
            (3, zone_left + 5*zone_w/6, zone_top + zone_h/6),
            (4, zone_left + zone_w/6, zone_top + zone_h/2),
            (5, zone_left + zone_w/2, zone_top + zone_h/2),
            (6, zone_left + 5*zone_w/6, zone_top + zone_h/2),
            (7, zone_left + zone_w/6, zone_top + 5*zone_h/6),
            (8, zone_left + zone_w/2, zone_top + 5*zone_h/6),
            (9, zone_left + 5*zone_w/6, zone_top + 5*zone_h/6),
        ]
        
        for num, x, y in zone_positions:
            painter.drawText(int(x - 5), int(y + 5), str(num))
        
        # === 궤적 및 공 마커 그리기 ===
        font = QFont(window_config.FONT_FAMILY, record_config.MARKER_FONT_SIZE)
        painter.setFont(font)
        
        # 최신 공 번호 (마지막 투구)
        latest_number = len(self.records) if self.records else 0
        
        for x, z, is_strike, number, speed, trajectory in self.records:
            is_selected = (number == self.selected_pitch)
            is_latest = (number == latest_number)
            
            # 궤적 표시 조건:
            # 1. 선택된 공이 있으면 → 선택된 공의 궤적만 표시
            # 2. 선택된 공이 없으면 → 최신 공의 궤적만 표시
            show_trajectory = False
            if self.selected_pitch is not None:
                show_trajectory = is_selected
            else:
                show_trajectory = is_latest
            
            # 궤적 그리기
            if trajectory and len(trajectory) >= 2 and show_trajectory:
                # 애니메이션 진행도 적용
                anim_progress = self.animation_progress if (is_latest and not self.selected_pitch) else 1.0
                self._draw_trajectory_mlb(painter, trajectory, is_strike, is_selected, anim_progress)
            
            # 마커는 항상 그리기 (모든 공의 위치 표시)
            marker_y = trajectory[-1][1] if trajectory else 0.0
            self._draw_marker_mlb(painter, x, z, marker_y, is_strike, number, is_selected, show_trajectory)
        
        # 타이틀 (다크 테마)
        title_font = QFont(window_config.FONT_FAMILY, 12, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(*record_config.COLOR_TEXT))
        painter.drawText(10, 18, "⚾ 투구 기록")
        
    def _draw_target_zone(self, painter, zone_left, zone_top, zone_w, zone_h):
        """타겟 구역 하이라이트 그리기"""
        zone_idx = self.target_zone - 1
        row = zone_idx // 3
        col = zone_idx % 3
        
        cell_w = zone_w / 3
        cell_h = zone_h / 3
        
        x = zone_left + col * cell_w
        y = zone_top + row * cell_h
        
        # 반투명 주황색 채우기
        target_color = QColor(255, 165, 0, 100)
        painter.fillRect(int(x), int(y), int(cell_w), int(cell_h), target_color)
        
        # 테두리
        target_pen = QPen(QColor(255, 165, 0), 2)
        painter.setPen(target_pen)
        painter.drawRect(int(x), int(y), int(cell_w), int(cell_h))
    
    def _draw_perspective_guides(self, painter):
        """원근 가이드라인 그리기 - 3D 깊이감을 위한 시각적 참조선"""
        margin = record_config.MARGIN
        w = self.width()
        h = self.height()
        
        # 소실점 (화면 상단 중앙)
        vanishing_x = w / 2
        vanishing_y = margin * 0.5
        
        # 스트라이크 존 모서리 좌표
        zone_left, zone_top = self._world_to_widget(self.strike_x_min, self.strike_z_max)
        zone_right, zone_bottom = self._world_to_widget(self.strike_x_max, self.strike_z_min)
        
        # 가이드라인 색상 (매우 연한 색상)
        guide_color = QColor(80, 80, 100, 40)
        guide_pen = QPen(guide_color, 1, Qt.DotLine)
        painter.setPen(guide_pen)
        
        # 소실점에서 스트라이크 존 4개 모서리로 가이드라인
        corners = [
            (zone_left, zone_top),      # 좌상
            (zone_right, zone_top),     # 우상
            (zone_left, zone_bottom),   # 좌하
            (zone_right, zone_bottom),  # 우하
        ]
        
        for cx, cy in corners:
            painter.drawLine(int(vanishing_x), int(vanishing_y), int(cx), int(cy))
        
        # 중간 깊이에 그리드 라인 (3D 바닥/천장 느낌)
        # 여러 깊이 레벨에 수평선 그리기
        for depth_level in [0.2, 0.4, 0.6, 0.8]:
            # 각 깊이에서의 좌우 끝점 계산
            left_x = vanishing_x + (zone_left - vanishing_x) * depth_level
            right_x = vanishing_x + (zone_right - vanishing_x) * depth_level
            line_y = vanishing_y + (zone_bottom - vanishing_y) * depth_level
            
            # 수평 가이드라인 (점점 더 넓어지는)
            alpha = int(20 + 30 * depth_level)
            depth_guide_color = QColor(80, 80, 100, alpha)
            depth_pen = QPen(depth_guide_color, 1, Qt.DotLine)
            painter.setPen(depth_pen)
            painter.drawLine(int(left_x), int(line_y), int(right_x), int(line_y))
        
        # 소실점 표시 (작은 십자)
        crosshair_size = 5
        crosshair_color = QColor(100, 100, 120, 60)
        painter.setPen(QPen(crosshair_color, 1))
        painter.drawLine(int(vanishing_x - crosshair_size), int(vanishing_y),
                        int(vanishing_x + crosshair_size), int(vanishing_y))
        painter.drawLine(int(vanishing_x), int(vanishing_y - crosshair_size),
                        int(vanishing_x), int(vanishing_y + crosshair_size))

    def _depth_to_visual(self, y):
        """깊이(Y)를 시각적 속성으로 변환
        Y가 클수록(멀수록) = 더 작고, 더 투명
        Y가 작을수록(가까울수록) = 더 크고, 더 불투명
        """
        # Y 정규화 (0=가깝다, 1=멀다)
        norm_y = max(0, min(1, (y - self.depth_y_min) / (self.depth_y_max - self.depth_y_min + 0.01)))
        
        # 크기 배율 (멀수록 작게: 0.5 ~ 1.0)
        scale = 1.0 - 0.5 * norm_y
        
        # 투명도 (멀수록 투명: 80 ~ 255)
        alpha = int(80 + (1 - norm_y) * 175)
        
        return scale, alpha
        
    def _draw_trajectory(self, painter, trajectory, is_strike, is_selected):
        """궤적 그리기 (야구 중계 스타일 - 정면 시점, 깊이 효과)
        
        3D 궤적을 정면에서 본 것처럼 표현:
        - X = 좌우 위치
        - Z = 높이
        - Y = 깊이 (투명도/선 굵기로 표현)
        """
        if len(trajectory) < 2:
            return
            
        # 기본 색상
        if is_selected:
            base_color = QColor(255, 200, 100) if is_strike else QColor(255, 150, 150)
            base_pen_width = 4
        else:
            base_color = QColor(100, 200, 100) if is_strike else QColor(200, 100, 100)
            base_pen_width = 3
        
        # 궤적 점들을 연결 (깊이 효과 적용)
        for i in range(len(trajectory) - 1):
            x1, y1, z1 = trajectory[i]
            x2, y2, z2 = trajectory[i + 1]
            
            # 정면 시점: X=좌우, Z=높이
            wx1, wz1 = self._world_to_widget(x1, z1)
            wx2, wz2 = self._world_to_widget(x2, z2)
            
            # 깊이에 따른 시각 효과 (평균 깊이 사용)
            avg_y = (y1 + y2) / 2
            scale, alpha = self._depth_to_visual(avg_y)
            
            # 진행도에 따른 추가 그라데이션 (시작→끝)
            progress = i / max(1, len(trajectory) - 1)
            alpha = int(alpha * (0.3 + 0.7 * progress))
            
            # 선 스타일 설정
            color = QColor(base_color)
            color.setAlpha(alpha)
            pen_width = max(1, int(base_pen_width * scale))
            
            pen = QPen(color, pen_width)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            
            painter.drawLine(int(wx1), int(wz1), int(wx2), int(wz2))
            
            # 깊이를 나타내는 작은 원 그리기 (선택된 궤적만)
            if is_selected and i % 2 == 0:
                circle_radius = max(2, int(4 * scale))
                circle_color = QColor(base_color)
                circle_color.setAlpha(int(alpha * 0.5))
                painter.setBrush(QBrush(circle_color))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(int(wx1 - circle_radius), int(wz1 - circle_radius),
                                   circle_radius * 2, circle_radius * 2)
        
        # 화살표 끝 (마지막 방향 표시 - 정면 시점)
        if len(trajectory) >= 2:
            x1, y1, z1 = trajectory[-2]
            x2, y2, z2 = trajectory[-1]
            wx1, wz1 = self._world_to_widget(x1, z1)
            wx2, wz2 = self._world_to_widget(x2, z2)
            
            # 방향 벡터 계산
            dx = wx2 - wx1
            dy = wz2 - wz1
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # 화살표 머리 그리기
                angle = math.atan2(dy, dx)
                arrow_size = 10
                
                ax1 = wx2 - arrow_size * math.cos(angle - math.pi/6)
                ay1 = wz2 - arrow_size * math.sin(angle - math.pi/6)
                ax2 = wx2 - arrow_size * math.cos(angle + math.pi/6)
                ay2 = wz2 - arrow_size * math.sin(angle + math.pi/6)
                
                arrow_color = QColor(255, 220, 100) if is_selected else base_color
                arrow_color.setAlpha(220)
                painter.setPen(QPen(arrow_color, 2))
                painter.setBrush(QBrush(arrow_color))
                
                # 삼각형 화살표
                arrow_path = QPainterPath()
                arrow_path.moveTo(wx2, wz2)
                arrow_path.lineTo(ax1, ay1)
                arrow_path.lineTo(ax2, ay2)
                arrow_path.closeSubpath()
                painter.drawPath(arrow_path)
            
    def _draw_marker(self, painter, x, z, is_strike, number, is_selected):
        """마커 그리기"""
        wx, wy = self._world_to_widget(x, z)
        
        # 마커 색상
        if is_strike:
            color = QColor(*record_config.COLOR_STRIKE)
        else:
            color = QColor(*record_config.COLOR_BALL)
        
        # 선택된 경우 테두리 강조
        radius = record_config.MARKER_RADIUS
        if is_selected:
            # 외곽 글로우 효과
            glow_color = QColor(255, 255, 0, 100)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(glow_color))
            painter.drawEllipse(int(wx - radius - 4), int(wy - radius - 4),
                               (radius + 4) * 2, (radius + 4) * 2)
            
            # 선택 테두리
            painter.setPen(QPen(QColor(255, 255, 0), 3))
            radius += 2
        else:
            painter.setPen(QPen(color.darker(120), 2))
        
        # 마커
        painter.setBrush(QBrush(color))
        painter.drawEllipse(int(wx - radius), int(wy - radius),
                           radius * 2, radius * 2)
        
        # 번호 표시
        painter.setPen(Qt.white)
        text = str(number)
        painter.drawText(int(wx - 4), int(wy + 4), text)
    
    def _catmull_rom_spline(self, p0, p1, p2, p3, num_points=10):
        """Catmull-Rom 스플라인으로 부드러운 곡선 점 생성"""
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom 계수
            x = 0.5 * ((2 * p1[0]) +
                      (-p0[0] + p2[0]) * t +
                      (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                      (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
            
            y = 0.5 * ((2 * p1[1]) +
                      (-p0[1] + p2[1]) * t +
                      (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                      (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
            
            points.append((x, y))
        return points
    
    def _draw_trajectory_mlb(self, painter, trajectory, is_strike, is_selected, animation_progress=1.0):
        """MLB 스타일 부드러운 곡선 궤적 그리기 (원근법 적용)
        
        - 원근 투영으로 3D 입체감 표현
        - 소실점: 화면 상단 중앙 (투수 방향)
        - 스트라이크 존: 화면 하단 (캐처 방향)
        - 깊이에 따른 크기/투명도 변화
        - Catmull-Rom 스플라인으로 부드러운 곡선
        - 궤적 스무딩으로 노이즈 제거
        - 애니메이션 지원
        """
        if len(trajectory) < 2:
            return
        
        # [노이즈 제거] 궤적 스무딩 (이동 평균)
        # 지그재그 현상을 줄이기 위해 좌표를 부드럽게 만듦
        smoothed_traj = []
        if len(trajectory) >= 3:
            # 첫 점은 그대로
            smoothed_traj.append(trajectory[0])
            
            for i in range(1, len(trajectory) - 1):
                # 이전, 현재, 다음 점의 평균
                prev_p = trajectory[i-1]
                curr_p = trajectory[i]
                next_p = trajectory[i+1]
                
                avg_x = (prev_p[0] + curr_p[0] + next_p[0]) / 3
                avg_y = (prev_p[1] + curr_p[1] + next_p[1]) / 3
                avg_z = (prev_p[2] + curr_p[2] + next_p[2]) / 3
                
                # Z가 0보다 작으면 0으로 클램핑
                if avg_z < 0:
                    avg_z = 0
                
                smoothed_traj.append((avg_x, avg_y, avg_z))
            
            # 마지막 점은 그대로
            smoothed_traj.append(trajectory[-1])
        else:
            smoothed_traj = list(trajectory)
        
        # 색상 선택
        if is_selected:
            colors = record_config.TRAJECTORY_COLORS['selected']
        elif is_strike:
            colors = record_config.TRAJECTORY_COLORS['strike']
        else:
            colors = record_config.TRAJECTORY_COLORS['ball']
        
        start_color = QColor(*colors['start'])
        end_color = QColor(*colors['end'])
        
        # 원근 변환된 위젯 좌표로 변환
        widget_points = []
        scale_values = []  # 깊이에 따른 스케일 저장
        
        for x, y, z in smoothed_traj:
            # Z가 0보다 작으면(땅 밑) 0으로 클램핑
            if z < 0:
                z = 0
            
            # 원근 투영 적용
            wx, wy, scale = self._perspective_transform(x, y, z)
            widget_points.append((wx, wy))
            scale_values.append(scale)
        
        # Catmull-Rom 스플라인으로 부드러운 곡선 점 생성
        smooth_points = []
        smooth_scales = []
        
        if len(widget_points) >= 4:
            for i in range(len(widget_points) - 3):
                p0, p1, p2, p3 = widget_points[i:i+4]
                s0, s1, s2, s3 = scale_values[i:i+4]
                
                segment_points = self._catmull_rom_spline(p0, p1, p2, p3, 8)
                smooth_points.extend(segment_points)
                
                # 스케일도 보간
                for j in range(8):
                    t = j / 7
                    interp_scale = s1 * (1-t) + s2 * t
                    smooth_scales.append(interp_scale)
            
            # 마지막 점 추가
            smooth_points.append(widget_points[-1])
            smooth_scales.append(scale_values[-1])
        else:
            smooth_points = widget_points
            smooth_scales = scale_values
        
        # 애니메이션 진행도에 따라 표시할 점 수 결정
        total_points = len(smooth_points)
        visible_count = max(2, int(total_points * animation_progress))
        visible_points = smooth_points[:visible_count]
        visible_scales = smooth_scales[:visible_count]
        
        if len(visible_points) < 2:
            return
        
        # 선분별로 그리기 (깊이에 따른 굵기/투명도 변화)
        for i in range(len(visible_points) - 1):
            p1 = visible_points[i]
            p2 = visible_points[i + 1]
            
            # 진행도 (0~1)
            progress = i / max(1, len(visible_points) - 1)
            
            # 깊이 기반 스케일 (평균)
            avg_scale = (visible_scales[i] + visible_scales[min(i+1, len(visible_scales)-1)]) / 2
            
            # 색상 보간 (시작→끝)
            r = int(start_color.red() + (end_color.red() - start_color.red()) * progress)
            g = int(start_color.green() + (end_color.green() - start_color.green()) * progress)
            b = int(start_color.blue() + (end_color.blue() - start_color.blue()) * progress)
            
            # 깊이에 따른 투명도 (멀면 더 투명)
            base_alpha = 100 + int(155 * avg_scale)  # 100~255
            
            # 선 굵기 (깊이에 따라 변화)
            base_width = record_config.TRAJECTORY_WIDTH if is_selected else record_config.TRAJECTORY_WIDTH - 1
            line_width = max(1, int(base_width * avg_scale))
            
            # 글로우 효과 (선택된 경우)
            if is_selected:
                glow_color = QColor(r, g, b, int(base_alpha * 0.3))
                glow_width = int(line_width * 2.5)
                glow_pen = QPen(glow_color, glow_width)
                glow_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(glow_pen)
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            
            # 메인 라인
            line_color = QColor(r, g, b, base_alpha)
            pen = QPen(line_color, line_width)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            
            # 깊이 마커 (매 N번째 점마다, 선택된 궤적만)
            if is_selected and i % 5 == 0:
                marker_radius = max(2, int(4 * avg_scale))
                marker_color = QColor(r, g, b, int(base_alpha * 0.6))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(marker_color))
                painter.drawEllipse(int(p1[0] - marker_radius), int(p1[1] - marker_radius),
                                   marker_radius * 2, marker_radius * 2)
        
        # 궤적 끝점에 화살표 효과 (애니메이션 완료 시)
        if animation_progress >= 0.95 and len(visible_points) >= 2:
            self._draw_arrow_head(painter, visible_points[-2], visible_points[-1], end_color)
    
    def _draw_arrow_head(self, painter, p1, p2, color):
        """화살표 머리 그리기"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 1:
            return
            
        angle = math.atan2(dy, dx)
        arrow_size = 12
        
        ax1 = p2[0] - arrow_size * math.cos(angle - math.pi/5)
        ay1 = p2[1] - arrow_size * math.sin(angle - math.pi/5)
        ax2 = p2[0] - arrow_size * math.cos(angle + math.pi/5)
        ay2 = p2[1] - arrow_size * math.sin(angle + math.pi/5)
        
        arrow_color = QColor(color)
        arrow_color.setAlpha(230)
        
        painter.setPen(QPen(arrow_color, 2))
        painter.setBrush(QBrush(arrow_color))
        
        arrow_path = QPainterPath()
        arrow_path.moveTo(p2[0], p2[1])
        arrow_path.lineTo(ax1, ay1)
        arrow_path.lineTo(ax2, ay2)
        arrow_path.closeSubpath()
        painter.drawPath(arrow_path)
    
    def _draw_marker_mlb(self, painter, x, z, y, is_strike, number, is_selected, has_trajectory):
        """MLB 스타일 마커 그리기 (3D 효과 + 그림자)"""
        wx, wy, _ = self._perspective_transform(x, y, z)
        
        # 기본 색상
        if is_strike:
            base_color = QColor(*record_config.COLOR_STRIKE)
        else:
            base_color = QColor(*record_config.COLOR_BALL)
        
        radius = record_config.MARKER_RADIUS
        
        # 그림자 효과
        shadow_offset = record_config.MARKER_SHADOW_OFFSET
        shadow_color = QColor(0, 0, 0, record_config.MARKER_SHADOW_ALPHA)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(shadow_color))
        painter.drawEllipse(int(wx - radius + shadow_offset), int(wy - radius + shadow_offset),
                           radius * 2, radius * 2)
        
        # 선택된 경우 글로우 효과
        if is_selected:
            glow_color = QColor(255, 255, 100, 120)
            painter.setBrush(QBrush(glow_color))
            painter.drawEllipse(int(wx - radius - 5), int(wy - radius - 5),
                               (radius + 5) * 2, (radius + 5) * 2)
            radius += 2
        
        # 궤적이 표시된 공은 더 강조
        if has_trajectory:
            highlight_color = QColor(255, 255, 255, 80)
            painter.setBrush(QBrush(highlight_color))
            painter.drawEllipse(int(wx - radius - 3), int(wy - radius - 3),
                               (radius + 3) * 2, (radius + 3) * 2)
        
        # 3D 그라데이션 효과
        gradient = QRadialGradient(wx - radius/3, wy - radius/3, radius * 1.5)
        gradient.setColorAt(0, base_color.lighter(150))
        gradient.setColorAt(0.5, base_color)
        gradient.setColorAt(1, base_color.darker(130))
        
        painter.setPen(QPen(base_color.darker(150), 1))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(int(wx - radius), int(wy - radius),
                           radius * 2, radius * 2)
        
        # 하이라이트 (3D 효과)
        highlight = QRadialGradient(wx - radius/2, wy - radius/2, radius/2)
        highlight.setColorAt(0, QColor(255, 255, 255, 180))
        highlight.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(highlight))
        painter.drawEllipse(int(wx - radius + 2), int(wy - radius + 2),
                           int(radius * 0.8), int(radius * 0.8))
        
        # 번호 표시
        font = QFont(window_config.FONT_FAMILY, record_config.MARKER_FONT_SIZE - 1, QFont.Bold)
        painter.setFont(font)
        painter.setPen(Qt.white)
        text = str(number)
        # 텍스트 중앙 정렬
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(text)
        text_h = fm.height()
        painter.drawText(int(wx - text_w/2), int(wy + text_h/4), text)
        
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트 - 공 선택/해제"""
        if event.button() == Qt.LeftButton:
            click_x = event.x()
            click_y = event.y()
            
            # 가장 가까운 공 찾기
            min_dist = float('inf')
            closest_number = None
            
            for x, z, is_strike, number, speed, traj in self.records:
                wx, wy = self._world_to_widget(x, z)
                dist = math.sqrt((click_x - wx)**2 + (click_y - wy)**2)
                
                if dist < record_config.MARKER_RADIUS + 10 and dist < min_dist:
                    min_dist = dist
                    closest_number = number
            
            # 같은 공 다시 클릭하면 선택 해제 (전체 보기로 전환)
            if closest_number is not None:
                if self.selected_pitch == closest_number:
                    # 선택 해제 → 최신 궤적 표시로 전환
                    self.selected_pitch = None
                else:
                    # 새로운 공 선택
                    self.selected_pitch = closest_number
                self.pitchSelected.emit(closest_number if self.selected_pitch else 0)
                self.update()
            else:
                # 빈 공간 클릭 시 선택 해제
                if self.selected_pitch is not None:
                    self.selected_pitch = None
                    self.pitchSelected.emit(0)
                    self.update()


class Scoreboard(QFrame):
    """
    스코어보드 위젯 (간소화 버전)
    B-S-O 카운트만 표시 (이닝, 점수 제거)
    """
    
    countChanged = pyqtSignal(int, int, int)  # balls, strikes, outs
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        
        self.balls = 0
        self.strikes = 0
        self.outs = 0
        
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QHBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # 카운트 섹션
        count_frame = QFrame()
        count_layout = QGridLayout(count_frame)
        count_layout.setSpacing(5)
        
        # B-S-O 라벨
        font_label = QFont(window_config.FONT_FAMILY, 14, QFont.Bold)
        font_count = QFont(window_config.FONT_FAMILY, 20, QFont.Bold)
        
        # Ball
        lbl_b = QLabel("B")
        lbl_b.setFont(font_label)
        lbl_b.setStyleSheet("color: #00AA00;")
        count_layout.addWidget(lbl_b, 0, 0)
        
        self.ball_indicators = []
        for i in range(4):
            indicator = QLabel("●" if i < self.balls else "○")
            indicator.setFont(font_count)
            indicator.setStyleSheet("color: #00AA00;")
            count_layout.addWidget(indicator, 0, i + 1)
            self.ball_indicators.append(indicator)
            
        # Strike
        lbl_s = QLabel("S")
        lbl_s.setFont(font_label)
        lbl_s.setStyleSheet("color: #DDDD00;")
        count_layout.addWidget(lbl_s, 1, 0)
        
        self.strike_indicators = []
        for i in range(3):
            indicator = QLabel("●" if i < self.strikes else "○")
            indicator.setFont(font_count)
            indicator.setStyleSheet("color: #DDDD00;")
            count_layout.addWidget(indicator, 1, i + 1)
            self.strike_indicators.append(indicator)
            
        # Out
        lbl_o = QLabel("O")
        lbl_o.setFont(font_label)
        lbl_o.setStyleSheet("color: #DD0000;")
        count_layout.addWidget(lbl_o, 2, 0)
        
        self.out_indicators = []
        for i in range(3):
            indicator = QLabel("●" if i < self.outs else "○")
            indicator.setFont(font_count)
            indicator.setStyleSheet("color: #DD0000;")
            count_layout.addWidget(indicator, 2, i + 1)
            self.out_indicators.append(indicator)
        
        layout.addWidget(count_frame)
        layout.addStretch()
        
    def add_strike(self):
        """스트라이크 추가"""
        self.strikes += 1
        if self.strikes >= 3:
            self.add_out()
            self.strikes = 0
            self.balls = 0
        self._update_display()
        
    def add_ball(self):
        """볼 추가"""
        self.balls += 1
        if self.balls >= 4:
            # 볼넷
            self.balls = 0
            self.strikes = 0
        self._update_display()
        
    def add_out(self):
        """아웃 추가"""
        self.outs += 1
        if self.outs >= 3:
            self.outs = 0
        self._update_display()
        
    def reset_count(self):
        """카운트 리셋"""
        self.balls = 0
        self.strikes = 0
        self._update_display()
        
    def reset_all(self):
        """전체 리셋"""
        self.balls = 0
        self.strikes = 0
        self.outs = 0
        self._update_display()
        
    def _update_display(self):
        """디스플레이 업데이트"""
        # Ball indicators
        for i, indicator in enumerate(self.ball_indicators):
            indicator.setText("●" if i < self.balls else "○")
            
        # Strike indicators
        for i, indicator in enumerate(self.strike_indicators):
            indicator.setText("●" if i < self.strikes else "○")
            
        # Out indicators
        for i, indicator in enumerate(self.out_indicators):
            indicator.setText("●" if i < self.outs else "○")
        
        self.countChanged.emit(self.balls, self.strikes, self.outs)


class GameModeWidget(QFrame):
    """
    게임 모드 위젯
    9구역 타겟 연습 모드
    """
    
    targetHit = pyqtSignal(int, bool)  # zone, is_hit
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        
        self.target_zone = None
        self.score = 0
        self.attempts = 0
        self.max_attempts = 10
        self.hits = 0
        
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 타이틀
        title = QLabel("🎯 타겟 모드")
        title.setFont(QFont(window_config.FONT_FAMILY, 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 현재 타겟 표시
        self.target_label = QLabel("목표: -")
        self.target_label.setFont(QFont(window_config.FONT_FAMILY, 14))
        self.target_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.target_label)
        
        # 점수
        score_layout = QHBoxLayout()
        
        self.score_label = QLabel(f"점수: {self.score}")
        self.score_label.setFont(QFont(window_config.FONT_FAMILY, 14, QFont.Bold))
        score_layout.addWidget(self.score_label)
        
        self.attempts_label = QLabel(f"시도: {self.attempts}/{self.max_attempts}")
        self.attempts_label.setFont(QFont(window_config.FONT_FAMILY, 12))
        score_layout.addWidget(self.attempts_label)
        
        layout.addLayout(score_layout)
        
        # 명중률
        self.accuracy_label = QLabel("명중률: 0%")
        self.accuracy_label.setFont(QFont(window_config.FONT_FAMILY, 12))
        self.accuracy_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.accuracy_label)
        
    def set_target(self, zone):
        """목표 구역 설정"""
        self.target_zone = zone
        zone_name = game_config.ZONE_NAMES.get(zone, str(zone))
        self.target_label.setText(f"목표: {zone} ({zone_name})")
        self.target_label.setStyleSheet("color: #FF6600; font-weight: bold;")
        
    def set_random_target(self):
        """랜덤 목표 설정"""
        import random
        zone = random.randint(1, 9)
        self.set_target(zone)
        return zone
        
    def check_hit(self, actual_zone):
        """명중 체크"""
        self.attempts += 1
        is_hit = (actual_zone == self.target_zone)
        
        if is_hit:
            self.hits += 1
            zone_score = game_config.ZONE_SCORES.get(self.target_zone, 5)
            self.score += zone_score
            
        self._update_display()
        self.targetHit.emit(self.target_zone, is_hit)
        
        return is_hit
        
    def reset(self):
        """리셋"""
        self.target_zone = None
        self.score = 0
        self.attempts = 0
        self.hits = 0
        self.target_label.setText("목표: -")
        self.target_label.setStyleSheet("")
        self._update_display()
        
    def _update_display(self):
        """디스플레이 업데이트"""
        self.score_label.setText(f"점수: {self.score}")
        self.attempts_label.setText(f"시도: {self.attempts}/{self.max_attempts}")
        
        if self.attempts > 0:
            accuracy = (self.hits / self.attempts) * 100
            self.accuracy_label.setText(f"명중률: {accuracy:.1f}%")
        else:
            self.accuracy_label.setText("명중률: 0%")


class StatsWidget(QFrame):
    """
    통계 위젯
    평균 구속, 스트라이크 비율 등 표시
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        
        self.total_pitches = 0
        self.strikes = 0
        self.balls = 0
        self.speeds = []
        
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QGridLayout(self)
        
        font_label = QFont(window_config.FONT_FAMILY, 11)
        font_value = QFont(window_config.FONT_FAMILY, 13, QFont.Bold)
        
        # 총 투구수
        layout.addWidget(QLabel("총 투구:"), 0, 0)
        self.total_label = QLabel("0")
        self.total_label.setFont(font_value)
        layout.addWidget(self.total_label, 0, 1)
        
        # 스트라이크
        layout.addWidget(QLabel("스트라이크:"), 1, 0)
        self.strike_label = QLabel("0")
        self.strike_label.setFont(font_value)
        self.strike_label.setStyleSheet("color: #00AA00;")
        layout.addWidget(self.strike_label, 1, 1)
        
        # 볼
        layout.addWidget(QLabel("볼:"), 2, 0)
        self.ball_label = QLabel("0")
        self.ball_label.setFont(font_value)
        self.ball_label.setStyleSheet("color: #DD0000;")
        layout.addWidget(self.ball_label, 2, 1)
        
        # 스트라이크 비율
        layout.addWidget(QLabel("S%:"), 0, 2)
        self.strike_pct_label = QLabel("0%")
        self.strike_pct_label.setFont(font_value)
        layout.addWidget(self.strike_pct_label, 0, 3)
        
        # 평균 구속
        layout.addWidget(QLabel("평균 구속:"), 1, 2)
        self.avg_speed_label = QLabel("- km/h")
        self.avg_speed_label.setFont(font_value)
        self.avg_speed_label.setStyleSheet("color: #FF6600;")
        layout.addWidget(self.avg_speed_label, 1, 3)
        
        # 최고 구속
        layout.addWidget(QLabel("최고 구속:"), 2, 2)
        self.max_speed_label = QLabel("- km/h")
        self.max_speed_label.setFont(font_value)
        self.max_speed_label.setStyleSheet("color: #FF0000;")
        layout.addWidget(self.max_speed_label, 2, 3)
        
    def add_pitch(self, is_strike, speed=None):
        """투구 기록 추가"""
        self.total_pitches += 1
        
        if is_strike:
            self.strikes += 1
        else:
            self.balls += 1
            
        if speed is not None and speed > 0:
            self.speeds.append(speed)
            
        self._update_display()
        
    def reset(self):
        """리셋"""
        self.total_pitches = 0
        self.strikes = 0
        self.balls = 0
        self.speeds = []
        self._update_display()
        
    def _update_display(self):
        """디스플레이 업데이트"""
        self.total_label.setText(str(self.total_pitches))
        self.strike_label.setText(str(self.strikes))
        self.ball_label.setText(str(self.balls))
        
        if self.total_pitches > 0:
            strike_pct = (self.strikes / self.total_pitches) * 100
            self.strike_pct_label.setText(f"{strike_pct:.1f}%")
        else:
            self.strike_pct_label.setText("0%")
            
        if self.speeds:
            avg_speed = sum(self.speeds) / len(self.speeds)
            max_speed = max(self.speeds)
            self.avg_speed_label.setText(f"{avg_speed:.1f} km/h")
            self.max_speed_label.setText(f"{max_speed:.1f} km/h")
        else:
            self.avg_speed_label.setText("- km/h")
            self.max_speed_label.setText("- km/h")


class PitchListWidget(QFrame):
    """
    투구 리스트 위젯
    던진 공들의 목록을 표시하고 선택 가능
    """
    
    pitchSelected = pyqtSignal(int)  # 선택된 공 번호
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 확장 가능
        
        self.pitches = []  # [(number, is_strike, speed), ...]
        self.selected_number = None  # 현재 선택된 공 번호
        
        self._init_ui()
        
    def _init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 타이틀
        title = QLabel("📜 투구 목록")
        title.setFont(QFont(window_config.FONT_FAMILY, 11, QFont.Bold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)
        
        # 리스트 위젯
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 3px;
                color: #ffffff;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3a3a3a;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
            }
        """)
        # 높이 제한 제거 - 확장 가능하게
        self.list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)
        
    def add_pitch(self, number, is_strike, speed):
        """투구 추가"""
        self.pitches.append((number, is_strike, speed))
        
        # 결과 표시
        result = "S" if is_strike else "B"
        result_color = "#00CC66" if is_strike else "#FF6666"
        
        # 구속 표시
        speed_text = f"{speed:.1f}km/h" if speed and speed > 0 else "-"
        
        # 리스트 아이템 생성
        item_text = f"#{number}  [{result}]  {speed_text}"
        item = QListWidgetItem(item_text)
        
        # 색상 설정
        if is_strike:
            item.setForeground(QColor(0, 200, 100))
        else:
            item.setForeground(QColor(255, 100, 100))
            
        self.list_widget.addItem(item)
        
        # 스크롤을 최신 항목으로
        self.list_widget.scrollToBottom()
        
    def clear_pitches(self):
        """리스트 초기화"""
        self.pitches = []
        self.selected_number = None
        self.list_widget.clear()
        
    def select_pitch(self, number):
        """특정 투구 선택 (0이면 선택 해제)"""
        if number == 0:
            # 선택 해제
            self.list_widget.clearSelection()
            self.selected_number = None
        else:
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if i + 1 == number:
                    self.list_widget.setCurrentItem(item)
                    self.selected_number = number
                    break
                
    def _on_item_clicked(self, item):
        """아이템 클릭 이벤트 - 토글 기능"""
        row = self.list_widget.row(item)
        number = row + 1
        
        # 같은 아이템 다시 클릭하면 선택 해제
        if self.selected_number == number:
            self.list_widget.clearSelection()
            self.selected_number = None
            self.pitchSelected.emit(0)  # 0은 선택 해제 의미
        else:
            self.selected_number = number
            self.pitchSelected.emit(number)
        
    def get_pitch_info(self, number):
        """특정 투구 정보 반환"""
        if 0 < number <= len(self.pitches):
            return self.pitches[number - 1]
        return None
