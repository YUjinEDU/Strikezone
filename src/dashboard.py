import os
import time
import threading
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import dash_table
from flask import send_from_directory

from config import BOX_EDGES

class Dashboard:
    """
    실시간 + 영구 기록 시각화를 위한 대시보드
    좌표계: 마커 좌표계 [x(좌우), y(깊이), z(높이)]
    """

    def __init__(self, port=8050, host='127.0.0.1'):
        self.app = dash.Dash(__name__)
        self.port = port
        self.host = host

        # 정적(영상) 서빙 디렉토리
        self.clips_dir = os.path.abspath('clips')
        os.makedirs(self.clips_dir, exist_ok=True)
        self._setup_static_routes()

        # 데이터 저장
        self.data_lock = threading.Lock()
        self.record_sheet_points = []   # [(x,z)]
        self.record_sheet_polygon = []  # [[x,z], ...] (plane2 권장)
        self.trajectory_3d = []
        self.strike_zone_corners_3d = []
        self.ball_zone_corners_3d = []
        self.ball_zone_corners2_3d = []
        self.box_corners_3d = []

        # 영구 피치 기록
        self.all_pitches = []
        self.next_pitch_id = 1

        # 선택/보기모드
        self.selected_pitch_id = None
        self.view_mode = 'all'

        self.setup_layout()
        self.setup_callbacks()

    def _setup_static_routes(self):
        @self.app.server.route('/clips/<path:filename>')
        def serve_clips(filename):
            print(f"📹 영상 요청: /clips/{filename}")
            print(f"📂 클립 디렉토리: {self.clips_dir}")
            file_path = os.path.join(self.clips_dir, filename)
            if os.path.exists(file_path):
                print(f"✅ 파일 존재: {file_path}")
                return send_from_directory(self.clips_dir, filename)
            else:
                print(f"❌ 파일 없음: {file_path}")
                return "File not found", 404

    def setup_layout(self):
        right_controls = html.Div([
            html.Div([
                html.H3("피치 영상"),
                html.Video(
                    id='pitch-video',
                    src='',
                    controls=True,
                    autoPlay=False,
                    style={
                        'width': '100%', 
                        'height': 'auto', 
                        'backgroundColor': '#000',
                        'maxHeight': '300px'
                    }
                ),
                html.Div(id='video-debug', style={'fontSize': '10px', 'color': 'gray'})
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.H3("피치 목록"),
                dash_table.DataTable(
                    id='pitch-table',
                    columns=[
                        {'name': '번호', 'id': 'number'},
                        {'name': '시간', 'id': 'timestamp'},
                        {'name': '결과', 'id': 'result'},
                        {'name': '속도 (km/h)', 'id': 'speed_kmh'}
                    ],
                    data=[],
                    row_selectable='single',
                    selected_rows=[],
                    style_table={'overflowY': 'auto', 'maxHeight': '400px'},
                    style_cell={'textAlign': 'center', 'padding': '8px', 'fontFamily': 'Arial'},
                    style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'}
                ),
                html.Div([
                    html.Label("보기 모드:"),
                    dcc.RadioItems(
                        id='view-mode',
                        options=[
                            {'label': '전체', 'value': 'all'},
                            {'label': '선택', 'value': 'selected'}
                        ],
                        value='all',
                        labelStyle={'display': 'inline-block', 'marginRight': '15px'}
                    )
                ], style={'marginTop': '10px'})
            ])
        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})

        left_graphs = html.Div([
            html.Div([
                html.H3("2D 기록지 (X 좌우 vs Z 높이)"),
                dcc.Graph(id='record-sheet', style={'height': '360px'})
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.H3("3D 기록지 (X/Y/Z = 좌우/깊이/높이)"),
                dcc.Graph(id='three-d-plot', style={'height': '420px'})
            ])
        ], style={'width': '65%', 'display': 'inline-block', 'padding': '10px'})

        self.app.layout = html.Div([
            html.H1("스트라이크 존 시각화 (영구 기록 + 인터랙션)"),
            html.Div([left_graphs, right_controls], style={'display': 'flex', 'flexDirection': 'row'}),
            dcc.Interval(id='interval-component', interval=500, n_intervals=0),
            dcc.Store(id='selected-pitch-id-store', data=None)
        ], style={'padding': '10px'})

    def setup_callbacks(self):
        @self.app.callback(
            [
                Output('record-sheet', 'figure'),
                Output('three-d-plot', 'figure'),
                Output('pitch-table', 'data'),
                Output('pitch-video', 'src'),
                Output('selected-pitch-id-store', 'data'),
                Output('video-debug', 'children')
            ],
            [
                Input('interval-component', 'n_intervals'),
                Input('pitch-table', 'selected_rows'),
                Input('view-mode', 'value')
            ],
            [State('selected-pitch-id-store', 'data')]
        )
        def update_all(n, selected_rows, view_mode, selected_pitch_id_state):
            with self.data_lock:
                table_data = []
                for p in self.all_pitches:
                    table_data.append({
                        'number': p.get('number', p['id']),
                        'timestamp': p.get('timestamp', ''),
                        'result': p.get('result', ''),
                        'speed_kmh': f"{p.get('speed_kmh', 0):.1f}" if p.get('speed_kmh') is not None else ''
                    })

                new_selected_pitch_id = selected_pitch_id_state
                if selected_rows and len(selected_rows) == 1 and len(self.all_pitches) > 0:
                    idx = selected_rows[0]
                    if 0 <= idx < len(self.all_pitches):
                        new_selected_pitch_id = self.all_pitches[idx]['id']

                video_src = ''
                debug_msg = '영상을 선택하세요'
                if new_selected_pitch_id is not None:
                    sel = next((p for p in self.all_pitches if p['id'] == new_selected_pitch_id), None)
                    if sel and sel.get('video_filename'):
                        video_src = f"/clips/{sel['video_filename']}"
                        print(f"🎥 비디오 소스 설정: {video_src}")
                        print(f"📁 파일명: {sel['video_filename']}")
                        full_path = os.path.join(self.clips_dir, sel['video_filename'])
                        print(f"🔍 전체 경로: {full_path}")
                        print(f"✅ 파일 존재: {os.path.exists(full_path)}")
                        if os.path.exists(full_path):
                            debug_msg = f"✅ {sel['video_filename']}"
                        else:
                            debug_msg = f"❌ 파일 없음: {sel['video_filename']}"

                self.view_mode = view_mode
                self.selected_pitch_id = new_selected_pitch_id

                record_fig = self._create_record_sheet_figure(
                    selected_pitch_id=new_selected_pitch_id if view_mode == 'selected' else None
                )
                three_d_fig = self._create_3d_figure(
                    selected_pitch_id=new_selected_pitch_id if view_mode == 'selected' else None
                )

            return record_fig, three_d_fig, table_data, video_src, new_selected_pitch_id, debug_msg

    # ===== 그래프 생성 (2D) =====
    def _create_record_sheet_figure(self, selected_pitch_id=None):
        fig = go.Figure()

        # 폴리곤(plane2), 동적 범위 계산용
        x_min, x_max, z_min, z_max = -0.25, 0.25, 0.1, 0.8
        if self.record_sheet_polygon:
            polygon_points = np.array(self.record_sheet_polygon, dtype=np.float32)
            pts_closed = np.vstack([polygon_points, polygon_points[0]])
            fig.add_trace(go.Scatter(
                x=pts_closed[:, 0].tolist(),
                y=pts_closed[:, 1].tolist(),
                mode='lines',
                line=dict(color='rgba(0,0,0,0.8)', width=2),
                name='스트라이크 존(뒤)'
            ))
            # 동적 범위(여백 포함) - 실제 좌표계 비율 유지
            x_min = float(np.min(polygon_points[:, 0])) - 0.1
            x_max = float(np.max(polygon_points[:, 0])) + 0.1
            z_min = float(np.min(polygon_points[:, 1])) - 0.1
            z_max = float(np.max(polygon_points[:, 1])) + 0.1

        # 포인트(전체 or 선택)
        if selected_pitch_id is None:
            if self.record_sheet_points:
                x_vals = [p[0] for p in self.record_sheet_points]
                z_vals = [p[1] for p in self.record_sheet_points]
                fig.add_trace(go.Scatter(
                    x=x_vals, y=z_vals,
                    mode='markers+text',
                    marker=dict(size=10, color='blue'),
                    text=[str(i+1) for i in range(len(x_vals))],  # 번호 표시
                    textposition='top center',
                    name='투구 위치(전체)'
                ))
        else:
            sel = next((p for p in self.all_pitches if p['id'] == selected_pitch_id), None)
            if sel and sel.get('point_3d') is not None:
                x = float(sel['point_3d'][0])
                z = float(sel['point_3d'][2])
                fig.add_trace(go.Scatter(
                    x=[x], y=[z],
                    mode='markers+text',
                    marker=dict(size=12, color='crimson'),
                    text=[f"#{sel.get('number', sel['id'])}"],
                    textposition='top center',
                    name='선택 투구'
                ))

        fig.update_layout(
            title='투구 위치 기록 (X 좌우 vs Z 높이)',
            xaxis=dict(
                title='X (좌우, m)', 
                range=[x_min, x_max], 
                showgrid=True, 
                zeroline=True, 
                zerolinecolor='rgba(0,0,0,0.25)',
                scaleanchor="y",  # Y축과 비율 연결
                scaleratio=1      # 1:1 비율 유지
            ),
            yaxis=dict(
                title='Z (높이, m)', 
                range=[z_min, z_max], 
                showgrid=True, 
                zeroline=True, 
                zerolinecolor='rgba(0,0,0,0.25)',
                constrain='domain'  # 도메인 내에서 제한
            ),
            autosize=True,
            margin=dict(l=30, r=30, t=50, b=30),
            plot_bgcolor='rgba(240,240,240,0.5)',
            paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        return fig

    # ===== 그래프 생성 (3D) =====
    def _create_3d_figure(self, selected_pitch_id=None):
        fig = go.Figure()

        if len(self.strike_zone_corners_3d) > 0:
            pts_closed = np.vstack([self.strike_zone_corners_3d, self.strike_zone_corners_3d[0]])
            fig.add_trace(go.Scatter3d(
                x=pts_closed[:, 0], y=pts_closed[:, 1], z=pts_closed[:, 2],
                mode='lines', line=dict(color='blue', width=4), name='앞 판정면'
            ))
        if len(self.ball_zone_corners_3d) > 0:
            pts_closed = np.vstack([self.ball_zone_corners_3d, self.ball_zone_corners_3d[0]])
            fig.add_trace(go.Scatter3d(
                x=pts_closed[:, 0], y=pts_closed[:, 1], z=pts_closed[:, 2],
                mode='lines', line=dict(color='green', width=3), name='볼 존 1(참고)'
            ))
        if len(self.ball_zone_corners2_3d) > 0:
            pts_closed = np.vstack([self.ball_zone_corners2_3d, self.ball_zone_corners2_3d[0]])
            fig.add_trace(go.Scatter3d(
                x=pts_closed[:, 0], y=pts_closed[:, 1], z=pts_closed[:, 2],
                mode='lines', line=dict(color='red', width=3), name='뒤 판정면'
            ))

        if len(self.box_corners_3d) > 0:
            for e in BOX_EDGES:
                if e[0] < len(self.box_corners_3d) and e[1] < len(self.box_corners_3d):
                    p1 = self.box_corners_3d[e[0]]; p2 = self.box_corners_3d[e[1]]
                    fig.add_trace(go.Scatter3d(
                        x=[float(p1[0]), float(p2[0])],
                        y=[float(p1[1]), float(p2[1])],
                        z=[float(p1[2]), float(p2[2])],
                        mode='lines',
                        line=dict(color='gray', width=3),
                        showlegend=False
                    ))

        if selected_pitch_id is None:
            for p in self.all_pitches:
                traj = p.get('trajectory_3d') or []
                if len(traj) >= 2:
                    xs = [float(pt[0]) for pt in traj]
                    ys = [float(pt[1]) for pt in traj]
                    zs = [float(pt[2]) for pt in traj]
                    fig.add_trace(go.Scatter3d(
                        x=xs, y=ys, z=zs,
                        mode='lines',
                        line=dict(color='rgba(255,255,255,0.6)', width=2),
                        name=f"#{p.get('number', p['id'])}"
                    ))
        else:
            sel = next((p for p in self.all_pitches if p['id'] == selected_pitch_id), None)
            if sel:
                traj = sel.get('trajectory_3d') or []
                if len(traj) >= 2:
                    xs = [float(pt[0]) for pt in traj]
                    ys = [float(pt[1]) for pt in traj]
                    zs = [float(pt[2]) for pt in traj]
                    fig.add_trace(go.Scatter3d(
                        x=xs, y=ys, z=zs,
                        mode='lines+markers',
                        line=dict(color='orange', width=5),
                        marker=dict(size=2, color='orange'),
                        name=f"선택 투구 #{sel.get('number', sel['id'])}"
                    ))
                if sel.get('point_3d') is not None:
                    x, y, z = sel['point_3d']
                    fig.add_trace(go.Scatter3d(
                        x=[float(x)], y=[float(y)], z=[float(z)],
                        mode='markers',
                        marker=dict(size=6, color='crimson'),
                        name='교차 지점'
                    ))

        fig.update_layout(
            title='3D 투구 궤적 (X=좌우, Y=깊이, Z=높이)',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (깊이, m)',
                zaxis_title='Z (높이, m)',
                aspectmode='cube',
                camera=dict(eye=dict(x=-1.5, y=-0.6, z=1))
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        return fig

    def update_data(self, data_dict):
        """
        외부(메인 루프)에서 호출.
        - append_pitch: { 'result','speed_kmh','point_3d','trajectory_3d','timestamp','video_filename' }
        - 그 외 키(record_sheet_polygon 등)는 교체 방식으로 세팅
        """
        with self.data_lock:
            if 'append_pitch' in data_dict and data_dict['append_pitch']:
                ap = dict(data_dict['append_pitch'])
                ap['id'] = self.next_pitch_id
                ap['number'] = ap.get('number', self.next_pitch_id)
                if 'timestamp' not in ap or not ap['timestamp']:
                    ap['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                self.all_pitches.append(ap)
                self.next_pitch_id += 1

                # 2D 기록지 포인트(x,z)에도 추가(마지막 포인트 기준)
                pt = ap.get('point_3d')
                if pt is not None and len(pt) == 3:
                    self.record_sheet_points.append([float(pt[0]), float(pt[2])])

            # 교체 속성 업데이트
            for key, value in data_dict.items():
                if key == 'append_pitch':
                    continue
                if hasattr(self, key):
                    setattr(self, key, value)

    def run_server(self, debug=False):
        def run():
            self.app.run_server(
                port=self.port,
                host=self.host,
                debug=debug,
                use_reloader=False
            )
        dashboard_thread = threading.Thread(target=run, daemon=True)
        dashboard_thread.start()
        print(f"대시보드 서버 시작: http://{self.host}:{self.port}")
        return dashboard_thread