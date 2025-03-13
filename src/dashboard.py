import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import threading
import time
import numpy as np
from config import BOX_EDGES
import dash_table


class Dashboard:
    """실시간 데이터 시각화를 위한 대시보드 클래스"""
    
    def __init__(self, port=8050, host='127.0.0.1'):
        """
        Args:
            port: 대시보드 서버 포트
            host: 대시보드 서버 호스트
        """
        # Dash 앱 초기화
        self.app = dash.Dash(__name__)
        self.port = port
        self.host = host
        
        # 데이터 저장 변수
        self.record_sheet_points = []  # 기록지 위의 점들
        self.record_sheet_polygon = [] # 기록지 위의 다각형
        self.trajectory_3d = []        # 3D 궤적
        self.strike_zone_corners_3d = [] # 스트라이크 존 코너
        self.ball_zone_corners_3d = []   # 볼 존 코너
        self.ball_zone_corners2_3d = []  # 볼 존 코너2
        self.box_corners_3d = []       # 박스 코너
        
        # 투구 통계 데이터
        self.pitch_count = 0           # 총 투구 수
        self.strike_count = 0          # 스트라이크 수
        self.ball_count = 0            # 볼 수
        self.pitch_speeds = []         # 투구 속도 기록
        self.pitch_results = []        # 투구 결과 (스트라이크/볼)
        self.pitch_history = []        # 투구 기록 (번호, 결과, 속도)
        
        # 쓰레드 안전을 위한 락
        self.data_lock = threading.Lock()
        
        # 레이아웃 설정
        self.setup_layout()
        
        # 콜백 설정
        self.setup_callbacks()
    
    def setup_layout(self):
        """대시보드 레이아웃 설정"""
        self.app.layout = html.Div([
            html.H1("스트라이크 존 시각화"),
            
            # 레이아웃 컨테이너
            html.Div([
                # 첫 번째 행: 3D 시각화 및 기록지
                html.Div([
                    # 3D 궤적 시각화
                    html.Div([
                        dcc.Graph(id='three-d-plot', style={'height': '500px'})
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    # 기록지 시각화
                    html.Div([
                        dcc.Graph(id='record-sheet', style={'height': '500px'})
                    ], style={'width': '50%', 'display': 'inline-block'})
                ], style={'display': 'flex', 'flexDirection': 'row'}),
                
                # 두 번째 행: 투구 통계 및 추천
                html.Div([
                    # 투구 통계 카드
                    html.Div([
                        html.Div([
                            html.H3("투구 통계", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.Div(id='pitch-stats', style={
                                'display': 'flex', 
                                'justifyContent': 'space-around',
                                'padding': '10px',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '10px'
                            })
                        ], style={'padding': '15px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'})
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
                    
                    # 투구 추천 카드
                    html.Div([
                        html.Div([
                            html.H3("투구 추천", style={'textAlign': 'center', 'marginBottom': '15px'}),
                            html.Div(id='pitch-recommendation', style={
                                'padding': '15px',
                                'backgroundColor': '#e8f4f8',
                                'borderRadius': '10px',
                                'minHeight': '100px'
                            })
                        ], style={'padding': '15px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'})
                    ], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'})
                ], style={'display': 'flex', 'flexDirection': 'row', 'marginTop': '20px'}),
                
                # 세 번째 행: 투구 기록 표
                html.Div([
                    html.Div([
                        html.H3("투구 기록", style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.Div(id='pitch-history-table', style={
                            'padding': '10px',
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '10px'
                        })
                    ], style={'padding': '15px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'})
                ], style={'marginTop': '20px'})
            ]),
            
            # 업데이트 타이머
            dcc.Interval(
                id='interval-component',
                interval=500,  # 0.5초마다 업데이트
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """대시보드 콜백 설정"""
        @self.app.callback(
            [Output('record-sheet', 'figure'),
             Output('three-d-plot', 'figure'),
             Output('pitch-stats', 'children'),
             Output('pitch-recommendation', 'children'),
             Output('pitch-history-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n_intervals):
            with self.data_lock:
                # 기록지 그래프 업데이트
                record_fig = self.create_record_sheet_figure()
                
                # 3D 궤적 그래프 업데이트
                three_d_fig = self.create_3d_figure()
                
                # 투구 통계 업데이트
                pitch_stats = self.create_pitch_stats()
                
                # 투구 추천 업데이트
                pitch_recommendation = self.create_pitch_recommendation()
                
                # 투구 기록 표 업데이트
                pitch_history_table = self.create_pitch_history_table()
            
            return record_fig, three_d_fig, pitch_stats, pitch_recommendation, pitch_history_table
    
    def create_record_sheet_figure(self):
        """
        기록지 그래프 생성
        
        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()
        
        # 중심점 표시 (항상 표시)
        fig.add_trace(go.Scatter(
            x=[0],
            y=[0.2],
            mode='markers',
            marker=dict(
                size=12,
                color='rgba(255, 0, 0, 0.7)',
                symbol='cross',
                line=dict(
                    color='rgba(0, 0, 0, 0.5)',
                    width=1
                )
            ),
            name='중심점'
        ))
        
        # 기록지 다각형 추가 (스트라이크 존 표시)
        if self.record_sheet_polygon:
            # 다각형 닫기 (첫 점을 마지막에 반복)
            polygon_points = np.array(self.record_sheet_polygon)
            pts_closed = np.vstack([polygon_points, polygon_points[0]])
            
            fig.add_trace(go.Scatter(
                x=pts_closed[:, 0].tolist(),
                y=pts_closed[:, 1].tolist(),
                mode='lines',
                line=dict(color='rgba(0, 0, 0, 0.5)', width=2),
                name='스트라이크 존'
            ))
        
        # 기록지 데이터 포인트 추가
        if self.record_sheet_points:
            x_vals = [p[0] for p in self.record_sheet_points]
            y_vals = [p[1] for p in self.record_sheet_points]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='blue',
                ),
                text=[str(i+1) for i in range(len(x_vals))],  # 순서 번호 표시
                textposition='top center',
                name='투구 위치'
            ))
        
        # 레이아웃 설정
        fig.update_layout(
            title='투구 위치 기록',
            xaxis=dict(
                title='좌우 위치',
                range=[-0.2, 0.2],
                showgrid=True,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.5)',
                zerolinewidth=2,
                # 눈금 간격 설정
                dtick=0.1,
                tickmode='linear'
            ),
            yaxis=dict(
                title='상하 위치',
                range=[0.1, 0.8],
                showgrid=True,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.5)',
                zerolinewidth=2,
                # 눈금 간격 설정
                dtick=0.1,
                tickmode='linear'
            ),
            autosize=True,
            margin=dict(l=30, r=30, t=50, b=30),
            # 항상 중심에 표시되도록 설정

            # 그리드 설정
            plot_bgcolor='rgba(240,240,240,0.5)',
            paper_bgcolor='white',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def create_3d_figure(self):
        """
        3D 궤적 그래프 생성
        
        Returns:
            Plotly Figure 객체
        """
        fig = go.Figure()
        
        # 스트라이크 존 추가
        if len(self.strike_zone_corners_3d) > 0:
            pts_closed = np.vstack([self.strike_zone_corners_3d, self.strike_zone_corners_3d[0]])
            fig.add_trace(go.Scatter3d(
                x=pts_closed[:, 0],
                y=pts_closed[:, 1],
                z=pts_closed[:, 2],
                mode='lines',
                line=dict(color='blue', width=4),
                name='스트라이크 존'
            ))
        
        # 볼 존 추가
        if len(self.ball_zone_corners_3d) > 0:
            pts_closed = np.vstack([self.ball_zone_corners_3d, self.ball_zone_corners_3d[0]])
            fig.add_trace(go.Scatter3d(
                x=pts_closed[:, 0],
                y=pts_closed[:, 1],
                z=pts_closed[:, 2],
                mode='lines',
                line=dict(color='green', width=4),
                name='볼 존 1'
            ))
        
        # 볼 존2 추가
        if len(self.ball_zone_corners2_3d) > 0:
            pts_closed = np.vstack([self.ball_zone_corners2_3d, self.ball_zone_corners2_3d[0]])
            fig.add_trace(go.Scatter3d(
                x=pts_closed[:, 0],
                y=pts_closed[:, 1],
                z=pts_closed[:, 2],
                mode='lines',
                line=dict(color='red', width=4),
                name='볼 존 2'
            ))
        
        # 박스 추가
        if len(self.box_corners_3d) > 0:
            for e in BOX_EDGES:
                if e[0] < len(self.box_corners_3d) and e[1] < len(self.box_corners_3d):
                    p1 = self.box_corners_3d[e[0]]
                    p2 = self.box_corners_3d[e[1]]
                    fig.add_trace(go.Scatter3d(
                        x=[float(p1[0]), float(p2[0])],
                        y=[float(p1[1]), float(p2[1])],
                        z=[float(p1[2]), float(p2[2])],
                        mode='lines',
                        line=dict(color='gray', width=4),
                        showlegend=False
                    ))
        
        # 3D 궤적 점 추가 - 데이터 포인트들
        if len(self.trajectory_3d) > 0:
            x_vals = [float(p[0]) for p in self.trajectory_3d]
            y_vals = [float(p[1]) for p in self.trajectory_3d]
            z_vals = [float(p[2]) for p in self.trajectory_3d]
            
            # 궤적 선 추가
            if len(x_vals) > 1:
                fig.add_trace(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='lines',
                    line=dict(color='yellow', width=4),
                    name='공 궤적'
                ))
            
            # 궤적 끝점 추가
            if len(x_vals) > 0:
                fig.add_trace(go.Scatter3d(
                    x=[x_vals[-1]],
                    y=[y_vals[-1]],
                    z=[z_vals[-1]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='orange',
                    ),
                    name='공 위치'
                ))
        
        # 레이아웃 설정
        fig.update_layout(
            title='3D 공 궤적',
            scene=dict(
                xaxis_title='X 축',
                yaxis_title='Y 축',
                zaxis_title='Z 축',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=-1.5, y=-0.6, z=1)
                ),
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def create_pitch_stats(self):
        """
        투구 통계 컴포넌트 생성
        
        Returns:
            HTML 컴포넌트 리스트
        """
        # 평균 속도 계산
        avg_speed = 0
        if self.pitch_speeds:
            avg_speed = sum(self.pitch_speeds) / len(self.pitch_speeds)
        
        # 스트라이크 비율 계산
        strike_ratio = 0
        if self.pitch_count > 0:
            strike_ratio = (self.strike_count / self.pitch_count) * 100
        
        # 통계 카드 생성
        stats_cards = [
            html.Div([
                html.H4("총 투구수", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(f"{self.pitch_count}", style={'textAlign': 'center', 'color': '#007bff', 'margin': '5px 0'})
            ], style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': 'white', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4("스트라이크", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(f"{self.strike_count}", style={'textAlign': 'center', 'color': '#28a745', 'margin': '5px 0'}),
                html.P(f"{strike_ratio:.1f}%", style={'textAlign': 'center', 'margin': '0'})
            ], style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': 'white', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4("볼", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(f"{self.ball_count}", style={'textAlign': 'center', 'color': '#dc3545', 'margin': '5px 0'})
            ], style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': 'white', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4("평균 속도", style={'textAlign': 'center', 'margin': '0'}),
                html.H2(f"{avg_speed:.1f} km/h", style={'textAlign': 'center', 'color': '#6f42c1', 'margin': '5px 0'})
            ], style={'padding': '10px', 'borderRadius': '5px', 'backgroundColor': 'white', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ]
        
        return stats_cards
    
    def create_pitch_recommendation(self):
        """
        투구 추천 컴포넌트 생성
        
        Returns:
            HTML 컴포넌트 리스트
        """
        # 투구 패턴 분석 및 추천
        recommendations = []
        
        # 투구 데이터가 충분한 경우에만 추천
        if self.pitch_count >= 3 and self.record_sheet_points:
            # 최근 투구 위치 분석
            recent_points = self.record_sheet_points[-3:]
            x_vals = [p[0] for p in recent_points]
            y_vals = [p[1] for p in recent_points]
            
            # X축 편향 확인
            x_avg = sum(x_vals) / len(x_vals)
            if x_avg > 0.05:
                recommendations.append(html.P("최근 투구가 오른쪽으로 치우치는 경향이 있습니다. 왼쪽으로 조정해보세요.", 
                                           style={'color': '#dc3545'}))
            elif x_avg < -0.05:
                recommendations.append(html.P("최근 투구가 왼쪽으로 치우치는 경향이 있습니다. 오른쪽으로 조정해보세요.", 
                                           style={'color': '#dc3545'}))
            
            # Y축(높이) 편향 확인
            y_avg = sum(y_vals) / len(y_vals)
            if y_avg > 0.3:
                recommendations.append(html.P("최근 투구가 높게 날아가는 경향이 있습니다. 낮게 조정해보세요.", 
                                           style={'color': '#dc3545'}))
            elif y_avg < 0.15:
                recommendations.append(html.P("최근 투구가 낮게 날아가는 경향이 있습니다. 높게 조정해보세요.", 
                                           style={'color': '#dc3545'}))
            
            # 스트라이크 비율에 따른 추천
            strike_ratio = 0
            if self.pitch_count > 0:
                strike_ratio = (self.strike_count / self.pitch_count) * 100
            
            if strike_ratio < 40:
                recommendations.append(html.P("스트라이크 비율이 낮습니다. 존을 더 정확히 공략해보세요.", 
                                           style={'color': '#007bff'}))
            elif strike_ratio > 80:
                recommendations.append(html.P("스트라이크 비율이 높습니다! 변화구를 섞어보는 것도 좋은 전략입니다.", 
                                           style={'color': '#28a745'}))
        
        # 추천사항이 없는 경우
        if not recommendations:
            if self.pitch_count == 0:
                recommendations.append(html.P("첫 투구를 시작해보세요!", style={'color': '#6c757d'}))
            else:
                recommendations.append(html.P("더 많은 투구 데이터가 필요합니다.", style={'color': '#6c757d'}))
                recommendations.append(html.P("계속해서 투구하면 패턴 분석 및 추천이 제공됩니다.", style={'color': '#6c757d'}))
        
        return recommendations
    
    def create_pitch_history_table(self):
        """
        투구 기록 표 생성
        
        Returns:
            Dash 테이블 컴포넌트
        """
        # 테이블 데이터 준비
        if not self.pitch_history:
            return html.P("투구 기록이 없습니다.", style={'textAlign': 'center', 'color': '#6c757d'})
        
        # 테이블 생성
        table = dash_table.DataTable(
            id='pitch-table',
            columns=[
                {'name': '번호', 'id': 'number'},
                {'name': '결과', 'id': 'result'},
                {'name': '속도 (km/h)', 'id': 'speed'}
            ],
            data=self.pitch_history,
            style_table={
                'overflowX': 'auto',
                'maxHeight': '300px',
                'overflowY': 'auto'
            },
            style_header={
                'backgroundColor': '#007bff',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'fontFamily': 'Arial'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{result} = "스트라이크"'},
                    'backgroundColor': 'rgba(40, 167, 69, 0.2)',
                    'color': '#28a745'
                },
                {
                    'if': {'filter_query': '{result} = "볼"'},
                    'backgroundColor': 'rgba(220, 53, 69, 0.2)',
                    'color': '#dc3545'
                }
            ],
            page_size=10
        )
        
        return table
    
    def update_data(self, data_dict):
        """
        시각화 데이터 업데이트
        
        Args:
            data_dict: 업데이트할 데이터를 담은 딕셔너리
        """
        with self.data_lock:
            # 딕셔너리의 각 키에 해당하는 데이터 업데이트
            for key, value in data_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def run_server(self, debug=False):
        """
        대시보드 서버 실행
        
        Args:
            debug: 디버그 모드 활성화 여부
            
        Returns:
            서버 실행 스레드
        """
        # 비동기로 서버 실행
        def run():
            self.app.run_server(
                port=self.port,
                host=self.host,
                debug=debug,
                use_reloader=False  # 리로더 비활성화 (중복 실행 방지)
            )
        
        # 대시보드 스레드 시작
        dashboard_thread = threading.Thread(target=run, daemon=True)
        dashboard_thread.start()
        
        print(f"대시보드 서버 시작: http://{self.host}:{self.port}")
        return dashboard_thread