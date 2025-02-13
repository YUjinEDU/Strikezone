import streamlit as st
import cv2
import plotly.graph_objs as go
import time
import base64
from streamlit_autorefresh import st_autorefresh


# (A) 전역 변수: 카메라 초기화
if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(2)  # 인덱스=0 카메라

def get_frame_jpeg():
    """카메라에서 한 프레임을 읽어 JPEG로 인코딩 → base64"""
    ret, frame = st.session_state.cap.read()
    if not ret:
        return None
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # JPEG 인코딩
    _, jpeg_data = cv2.imencode('.jpg', frame_rgb)
    b64 = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')
    return b64

def get_3d_figure():
    """예시 3D 그래프 생성 (Plotly)"""
    fig = go.Figure()
    # 예: 간단히 3D scatter
    trace = go.Scatter3d(
        x=[0,1,2], y=[2,1,0], z=[1,2,3],
        mode='markers',
        marker=dict(size=6, color='red')
    )
    fig.add_trace(trace)
    fig.update_layout(
        scene=dict(aspectmode='cube'),
        title="3D Plot"
    )
    return fig

# ----------------------
# Streamlit Layout
# ----------------------
st.title("Streamlit Camera + 3D Graph Demo")

# 1) 카메라 ON/OFF 제어
start_cam = st.checkbox("Start Camera", value=True)

# 2) 화면 레이아웃 분할: 왼쪽(카메라), 오른쪽(그래프)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Camera View")
    if start_cam:
        frame_b64 = get_frame_jpeg()
        if frame_b64 is not None:
            # HTML <img> 태그로 base64 표시
            st.markdown(
                f'<img src="data:image/jpg;base64,{frame_b64}" width="100%" />',
                unsafe_allow_html=True
            )
        else:
            st.write("Failed to read frame")
    else:
        st.write("Camera is OFF")

with col2:
    st.markdown("### 3D Graph")
    fig_3d = get_3d_figure()
    st.plotly_chart(fig_3d, use_container_width=True)

# 3) 자동 리프레시
# Streamlit은 한 번 렌더 후 멈추므로, 실시간 느낌을 내려면 st.experimental_rerun() or st_autorefresh() 사용
refresh_switch = st.checkbox("Auto Refresh every second", value=True)
if st_autorefresh:
    st_autorefresh(interval=1000) 
    #time.sleep(1)
