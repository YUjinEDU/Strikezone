# pip install dash plotly
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import random
import time
import threading

app = dash.Dash(__name__)

# 전역 리스트 (실시간 데이터)
points_3d = []

app.layout = html.Div([
    dcc.Graph(id='3d-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1000, # 1초마다 콜백
        n_intervals=0
    )
])

@app.callback(
    Output('3d-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_figure(n):
    # points_3d 에 있는 점들로 그래프
    xs = [p[0] for p in points_3d]
    ys = [p[1] for p in points_3d]
    zs = [p[2] for p in points_3d]
    
    scatter = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers', marker=dict(size=5, color='red')
    )
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(aspectmode='cube'),
        title='Real-Time 3D Points'
    )
    return fig

def data_generator():
    """백그라운드에서 임의 점을 추가(시뮬레이션)"""
    while True:
        x_new = random.uniform(-1,1)
        y_new = random.uniform(-1,1)
        z_new = random.uniform(0,1)
        points_3d.append((x_new,y_new,z_new))
        time.sleep(2)  # 2초마다 새 점 추가

if __name__ == '__main__':
    # 백그라운드 스레드에서 점 추가
    threading.Thread(target=data_generator, daemon=True).start()
    
    # 서버 실행
    app.run_server(debug=True)
