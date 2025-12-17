# 🎯 AR Strike Zone 구현 가이드 - Part 2

> **Part 1 이어서**: 서버/웹, 물리 기반 추적, 데이터 증강, 참고 논문

---

## 5. Phase 3: 서버 및 웹 대시보드 (3-4주)

### 5.1 서버 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    Backend Server                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│  │ FastAPI  │────▶│  Redis   │────▶│ Postgres │        │
│  │ WebSocket│     │  Cache   │     │Timescale │        │
│  └────┬─────┘     └──────────┘     └──────────┘        │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────┐                                           │
│  │ JWT Auth │                                           │
│  └──────────┘                                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 5.2 FastAPI 서버 구현

#### 5.2.1 프로젝트 구조

```
server/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── pitch.py
│   │   └── session.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── pitch_schema.py
│   │   └── user_schema.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── pitches.py
│   │   └── websocket.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pitch_service.py
│   │   └── analytics_service.py
│   └── db/
│       ├── __init__.py
│       ├── database.py
│       └── redis_client.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

#### 5.2.2 핵심 코드

```python
# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json

from app.config import settings
from app.db.database import engine, Base
from app.db.redis_client import redis_client
from app.api import auth, pitches, websocket

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시
    Base.metadata.create_all(bind=engine)
    await redis_client.connect()
    yield
    # 종료 시
    await redis_client.disconnect()

app = FastAPI(
    title="AR Strike Zone API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(pitches.router, prefix="/api/pitches", tags=["pitches"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

```python
# app/api/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Set
import json
import asyncio

from app.schemas.pitch_schema import PitchData
from app.services.pitch_service import PitchService
from app.db.redis_client import redis_client

router = APIRouter()

class ConnectionManager:
    """WebSocket 연결 관리"""
    
    def __init__(self):
        # user_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
    
    async def send_to_user(self, user_id: str, message: dict):
        """특정 사용자의 모든 연결에 메시지 전송"""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass
    
    async def broadcast_to_user_web(self, user_id: str, data: dict):
        """웹 클라이언트에 실시간 업데이트 전송"""
        # Redis pub/sub을 통해 다른 서버 인스턴스에도 전파
        await redis_client.publish(f"pitch_updates:{user_id}", json.dumps(data))
        await self.send_to_user(user_id, data)

manager = ConnectionManager()

@router.websocket("/pitch/{user_id}")
async def pitch_websocket(websocket: WebSocket, user_id: str):
    """
    모바일 앱에서 투구 데이터를 수신하는 WebSocket 엔드포인트
    """
    await manager.connect(websocket, user_id)
    pitch_service = PitchService()
    
    try:
        while True:
            # 모바일에서 데이터 수신
            data = await websocket.receive_json()
            
            # 데이터 검증 및 저장
            pitch_data = PitchData(**data)
            saved_pitch = await pitch_service.save_pitch(user_id, pitch_data)
            
            # 실시간 통계 업데이트
            stats = await pitch_service.get_session_stats(user_id)
            
            # 웹 클라이언트에 브로드캐스트
            await manager.broadcast_to_user_web(user_id, {
                "type": "new_pitch",
                "pitch": saved_pitch.dict(),
                "stats": stats
            })
            
            # 응답
            await websocket.send_json({
                "status": "saved",
                "pitch_id": saved_pitch.id
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)

@router.websocket("/dashboard/{user_id}")
async def dashboard_websocket(websocket: WebSocket, user_id: str):
    """
    웹 대시보드에서 실시간 업데이트를 수신하는 WebSocket 엔드포인트
    """
    await manager.connect(websocket, f"web_{user_id}")
    
    # Redis pub/sub 구독
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"pitch_updates:{user_id}")
    
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                await websocket.send_json(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket, f"web_{user_id}")
        await pubsub.unsubscribe(f"pitch_updates:{user_id}")
```

```python
# app/schemas/pitch_schema.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Point3D(BaseModel):
    x: float
    y: float
    z: float
    timestamp: int  # milliseconds

class PitchData(BaseModel):
    timestamp: int
    trajectory: List[Point3D]
    speed_kmh: float
    judgment: str  # "STRIKE" or "BALL"
    crossing_point: Optional[Point3D] = None
    pitch_type: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 1702800000000,
                "trajectory": [
                    {"x": 0.0, "y": 18.44, "z": 1.5, "timestamp": 0},
                    {"x": 0.05, "y": 10.0, "z": 1.2, "timestamp": 200},
                    {"x": 0.1, "y": 0.43, "z": 0.8, "timestamp": 450}
                ],
                "speed_kmh": 142.5,
                "judgment": "STRIKE",
                "crossing_point": {"x": 0.1, "y": 0.43, "z": 0.8, "timestamp": 450}
            }
        }

class PitchResponse(PitchData):
    id: int
    user_id: str
    created_at: datetime

class SessionStats(BaseModel):
    total_pitches: int
    strikes: int
    balls: int
    avg_speed: float
    max_speed: float
    strike_rate: float
```

```python
# app/models/pitch.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from app.db.database import Base

class Pitch(Base):
    __tablename__ = "pitches"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    session_id = Column(String, index=True)
    
    timestamp = Column(DateTime, default=func.now())
    trajectory = Column(JSON)  # List of Point3D
    speed_kmh = Column(Float)
    judgment = Column(String)  # STRIKE/BALL
    crossing_point = Column(JSON)  # Point3D
    pitch_type = Column(String, nullable=True)
    
    # 분석 데이터
    release_point = Column(JSON, nullable=True)
    break_amount = Column(Float, nullable=True)  # 변화량
    spin_rate = Column(Float, nullable=True)     # 회전수 (추후 확장)
    
    created_at = Column(DateTime, default=func.now())
```

#### 5.2.3 Docker 설정

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/strikezone
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=your-secret-key
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=strikezone
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

volumes:
  postgres_data:
  redis_data:
```

### 5.3 웹 대시보드 (React)

#### 5.3.1 프로젝트 구조

```
web/
├── src/
│   ├── components/
│   │   ├── Dashboard/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── PitchList.tsx
│   │   │   └── StatsCard.tsx
│   │   ├── Visualization/
│   │   │   ├── Trajectory3D.tsx
│   │   │   ├── StrikeZoneHeatmap.tsx
│   │   │   └── SpeedChart.tsx
│   │   └── common/
│   │       └── ...
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   └── usePitchData.ts
│   ├── services/
│   │   └── api.ts
│   ├── store/
│   │   └── pitchStore.ts
│   ├── types/
│   │   └── pitch.ts
│   ├── App.tsx
│   └── main.tsx
├── package.json
└── vite.config.ts
```

#### 5.3.2 3D 궤적 시각화 (Three.js)

```typescript
// src/components/Visualization/Trajectory3D.tsx
import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Point3D, PitchData } from '../../types/pitch';

interface Props {
  pitches: PitchData[];
  selectedPitchId?: number;
}

export const Trajectory3D: React.FC<Props> = ({ pitches, selectedPitchId }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  
  useEffect(() => {
    if (!containerRef.current) return;
    
    // Scene 설정
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    sceneRef.current = scene;
    
    // Camera 설정 (포수 시점)
    const camera = new THREE.PerspectiveCamera(
      60,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      100
    );
    camera.position.set(0, 1.5, -2);  // 포수 뒤에서 보는 시점
    camera.lookAt(0, 1, 10);
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    
    // 조명
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    scene.add(directionalLight);
    
    // 스트라이크 존 박스
    const strikeZoneGeometry = new THREE.BoxGeometry(0.43, 0.56, 0.43);
    const strikeZoneEdges = new THREE.EdgesGeometry(strikeZoneGeometry);
    const strikeZoneLine = new THREE.LineSegments(
      strikeZoneEdges,
      new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 })
    );
    strikeZoneLine.position.set(0, 0.85, 0);  // 홈플레이트 위치
    scene.add(strikeZoneLine);
    
    // 홈플레이트
    const plateGeometry = new THREE.PlaneGeometry(0.43, 0.43);
    const plateMaterial = new THREE.MeshBasicMaterial({ 
      color: 0xffffff, 
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.5
    });
    const plate = new THREE.Mesh(plateGeometry, plateMaterial);
    plate.rotation.x = -Math.PI / 2;
    plate.position.y = 0.01;
    scene.add(plate);
    
    // 마운드
    const moundGeometry = new THREE.CircleGeometry(0.5, 32);
    const moundMaterial = new THREE.MeshBasicMaterial({ 
      color: 0x8b4513,
      side: THREE.DoubleSide
    });
    const mound = new THREE.Mesh(moundGeometry, moundMaterial);
    mound.rotation.x = -Math.PI / 2;
    mound.position.set(0, 0.3, 18.44);
    scene.add(mound);
    
    // 그리드
    const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    scene.add(gridHelper);
    
    // 애니메이션 루프
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    
    // 리사이즈 핸들러
    const handleResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      containerRef.current?.removeChild(renderer.domElement);
    };
  }, []);
  
  // 투구 궤적 업데이트
  useEffect(() => {
    if (!sceneRef.current) return;
    
    // 기존 궤적 제거
    const toRemove: THREE.Object3D[] = [];
    sceneRef.current.traverse((child) => {
      if (child.userData.isPitchTrajectory) {
        toRemove.push(child);
      }
    });
    toRemove.forEach(obj => sceneRef.current?.remove(obj));
    
    // 새 궤적 추가
    pitches.forEach((pitch, index) => {
      const isSelected = pitch.id === selectedPitchId;
      const color = pitch.judgment === 'STRIKE' ? 0xff4444 : 0x4444ff;
      
      // 궤적 라인
      const points = pitch.trajectory.map(p => 
        new THREE.Vector3(p.x, p.z, p.y)  // Y와 Z 교환 (Three.js 좌표계)
      );
      
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ 
        color: isSelected ? 0xffff00 : color,
        linewidth: isSelected ? 3 : 1,
        transparent: !isSelected,
        opacity: isSelected ? 1 : 0.5
      });
      
      const line = new THREE.Line(geometry, material);
      line.userData.isPitchTrajectory = true;
      sceneRef.current?.add(line);
      
      // 공 위치 (마지막 포인트)
      if (pitch.crossing_point) {
        const sphereGeometry = new THREE.SphereGeometry(0.0365, 16, 16);
        const sphereMaterial = new THREE.MeshBasicMaterial({ color });
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.set(
          pitch.crossing_point.x,
          pitch.crossing_point.z,
          pitch.crossing_point.y
        );
        sphere.userData.isPitchTrajectory = true;
        sceneRef.current?.add(sphere);
      }
    });
  }, [pitches, selectedPitchId]);
  
  return (
    <div 
      ref={containerRef} 
      style={{ width: '100%', height: '500px', borderRadius: '8px', overflow: 'hidden' }}
    />
  );
};
```

#### 5.3.3 스트라이크 존 히트맵

```typescript
// src/components/Visualization/StrikeZoneHeatmap.tsx
import React from 'react';
import Plot from 'react-plotly.js';
import { PitchData } from '../../types/pitch';

interface Props {
  pitches: PitchData[];
}

export const StrikeZoneHeatmap: React.FC<Props> = ({ pitches }) => {
  // 존 통과 위치 추출
  const crossingPoints = pitches
    .filter(p => p.crossing_point)
    .map(p => ({
      x: p.crossing_point!.x,
      y: p.crossing_point!.z,  // 높이
      judgment: p.judgment
    }));
  
  // 스트라이크/볼 분리
  const strikes = crossingPoints.filter(p => p.judgment === 'STRIKE');
  const balls = crossingPoints.filter(p => p.judgment === 'BALL');
  
  // 스트라이크 존 경계 (미터)
  const zoneWidth = 0.43 / 2;  // 홈플레이트 폭의 절반
  const zoneBottom = 0.57;     // 무릎
  const zoneTop = 1.13;        // 가슴 중간
  
  return (
    <Plot
      data={[
        // 스트라이크
        {
          x: strikes.map(p => p.x),
          y: strikes.map(p => p.y),
          mode: 'markers',
          type: 'scatter',
          name: 'Strike',
          marker: {
            color: 'red',
            size: 12,
            symbol: 'circle'
          }
        },
        // 볼
        {
          x: balls.map(p => p.x),
          y: balls.map(p => p.y),
          mode: 'markers',
          type: 'scatter',
          name: 'Ball',
          marker: {
            color: 'blue',
            size: 12,
            symbol: 'circle'
          }
        }
      ]}
      layout={{
        title: '투구 위치 분포',
        width: 400,
        height: 500,
        xaxis: {
          title: '좌우 (m)',
          range: [-0.5, 0.5],
          zeroline: true
        },
        yaxis: {
          title: '높이 (m)',
          range: [0.3, 1.5],
          zeroline: false
        },
        shapes: [
          // 스트라이크 존 박스
          {
            type: 'rect',
            x0: -zoneWidth,
            x1: zoneWidth,
            y0: zoneBottom,
            y1: zoneTop,
            line: { color: 'green', width: 3 },
            fillcolor: 'rgba(0, 255, 0, 0.1)'
          }
        ],
        paper_bgcolor: '#1a1a2e',
        plot_bgcolor: '#1a1a2e',
        font: { color: 'white' }
      }}
    />
  );
};
```

#### 5.3.4 WebSocket Hook

```typescript
// src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback, useState } from 'react';
import { PitchData, SessionStats } from '../types/pitch';

interface WebSocketMessage {
  type: 'new_pitch' | 'stats_update' | 'session_end';
  pitch?: PitchData;
  stats?: SessionStats;
}

export const useWebSocket = (userId: string) => {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastPitch, setLastPitch] = useState<PitchData | null>(null);
  const [stats, setStats] = useState<SessionStats | null>(null);
  
  const connect = useCallback(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/dashboard/${userId}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };
    
    ws.onmessage = (event) => {
      const data: WebSocketMessage = JSON.parse(event.data);
      
      switch (data.type) {
        case 'new_pitch':
          if (data.pitch) setLastPitch(data.pitch);
          if (data.stats) setStats(data.stats);
          break;
        case 'stats_update':
          if (data.stats) setStats(data.stats);
          break;
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      // 자동 재연결
      setTimeout(connect, 3000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    wsRef.current = ws;
  }, [userId]);
  
  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);
  
  return { isConnected, lastPitch, stats };
};
```

---

## 6. Phase 4: 물리 기반 폐루프 추적 (2-3주)

### 6.1 칼만 필터 + 물리 모델

#### 6.1.1 상태 공간 모델

```
상태 벡터 X = [x, y, z, vx, vy, vz]^T

상태 전이 모델 (등가속도 + 중력):
  x(t+1) = x(t) + vx(t)*dt
  y(t+1) = y(t) + vy(t)*dt
  z(t+1) = z(t) + vz(t)*dt - 0.5*g*dt²
  vx(t+1) = vx(t)
  vy(t+1) = vy(t)
  vz(t+1) = vz(t) - g*dt

여기서 g = 9.81 m/s², dt = 1/60 s (60fps)
```

#### 6.1.2 구현 코드

```kotlin
// PhysicsKalmanTracker.kt
class PhysicsKalmanTracker(
    private val processNoise: Float = 0.1f,
    private val measurementNoise: Float = 0.5f,
    private val gravity: Float = 9.81f
) {
    // 상태 벡터: [x, y, z, vx, vy, vz]
    private var state = FloatArray(6) { 0f }
    private var covariance = Array(6) { FloatArray(6) { 0f } }
    
    // 상태 전이 행렬 (dt에 따라 동적 생성)
    private fun getTransitionMatrix(dt: Float): Array<FloatArray> {
        return arrayOf(
            floatArrayOf(1f, 0f, 0f, dt, 0f, 0f),
            floatArrayOf(0f, 1f, 0f, 0f, dt, 0f),
            floatArrayOf(0f, 0f, 1f, 0f, 0f, dt),
            floatArrayOf(0f, 0f, 0f, 1f, 0f, 0f),
            floatArrayOf(0f, 0f, 0f, 0f, 1f, 0f),
            floatArrayOf(0f, 0f, 0f, 0f, 0f, 1f)
        )
    }
    
    // 중력에 의한 제어 입력
    private fun getControlInput(dt: Float): FloatArray {
        return floatArrayOf(
            0f,
            0f,
            -0.5f * gravity * dt * dt,  // 위치 보정
            0f,
            0f,
            -gravity * dt               // 속도 보정
        )
    }
    
    fun predict(dt: Float): FloatArray {
        val F = getTransitionMatrix(dt)
        val u = getControlInput(dt)
        
        // 상태 예측: x = F*x + u
        val newState = FloatArray(6)
        for (i in 0..5) {
            newState[i] = u[i]
            for (j in 0..5) {
                newState[i] += F[i][j] * state[j]
            }
        }
        
        // 공분산 예측: P = F*P*F^T + Q
        val newCov = Array(6) { FloatArray(6) { 0f } }
        // ... 행렬 연산 ...
        
        state = newState
        covariance = newCov
        
        return state.sliceArray(0..2)  // 위치만 반환
    }
    
    fun update(measurement: FloatArray): FloatArray {
        // 측정 행렬 H (위치만 관측)
        val H = arrayOf(
            floatArrayOf(1f, 0f, 0f, 0f, 0f, 0f),
            floatArrayOf(0f, 1f, 0f, 0f, 0f, 0f),
            floatArrayOf(0f, 0f, 1f, 0f, 0f, 0f)
        )
        
        // 칼만 이득 계산
        // K = P*H^T * (H*P*H^T + R)^(-1)
        
        // 상태 업데이트
        // x = x + K*(z - H*x)
        
        // 공분산 업데이트
        // P = (I - K*H)*P
        
        return state.sliceArray(0..2)
    }
    
    fun updateWithGating(
        measurement: FloatArray,
        gatingThreshold: Float = 0.5f  // 미터
    ): FloatArray? {
        // 예측 위치와 측정 위치의 거리
        val predicted = state.sliceArray(0..2)
        val distance = sqrt(
            (predicted[0] - measurement[0]).pow(2) +
            (predicted[1] - measurement[1]).pow(2) +
            (predicted[2] - measurement[2]).pow(2)
        )
        
        return if (distance < gatingThreshold) {
            update(measurement)
        } else {
            // 이상치로 판단, 업데이트 안 함
            Log.w("Tracker", "Gating rejected: distance=$distance")
            null
        }
    }
    
    fun getPredictedTrajectory(numFrames: Int, dt: Float): List<FloatArray> {
        // 현재 상태에서 미래 궤적 예측
        val trajectory = mutableListOf<FloatArray>()
        var tempState = state.copyOf()
        
        for (i in 0 until numFrames) {
            val F = getTransitionMatrix(dt)
            val u = getControlInput(dt)
            
            val newState = FloatArray(6)
            for (j in 0..5) {
                newState[j] = u[j]
                for (k in 0..5) {
                    newState[j] += F[j][k] * tempState[k]
                }
            }
            tempState = newState
            trajectory.add(tempState.sliceArray(0..2))
        }
        
        return trajectory
    }
    
    fun initialize(position: FloatArray, velocity: FloatArray? = null) {
        state[0] = position[0]
        state[1] = position[1]
        state[2] = position[2]
        
        if (velocity != null) {
            state[3] = velocity[0]
            state[4] = velocity[1]
            state[5] = velocity[2]
        } else {
            // 초기 속도 추정 (일반적인 투구 속도)
            state[3] = 0f
            state[4] = -40f  // 약 144 km/h
            state[5] = 0f
        }
        
        // 초기 공분산
        for (i in 0..5) {
            covariance[i][i] = if (i < 3) 0.1f else 5f
        }
    }
    
    fun getSpeed(): Float {
        return sqrt(state[3]*state[3] + state[4]*state[4] + state[5]*state[5])
    }
    
    fun getSpeedKmh(): Float = getSpeed() * 3.6f
}
```

### 6.2 폐루프 검출-추적 통합

```kotlin
// PitchTracker.kt
class PitchTracker(
    private val detector: TFLiteWrapper,
    private val kalman: PhysicsKalmanTracker,
    private val coordinator: CoordinateTransformer
) {
    enum class TrackingState {
        IDLE,           // 대기 중
        TRACKING,       // 추적 중
        LOST,           // 추적 실패
        COMPLETED       // 투구 완료
    }
    
    private var state = TrackingState.IDLE
    private var trajectory = mutableListOf<Point3D>()
    private var missedFrames = 0
    private val maxMissedFrames = 5  // 5프레임 이상 미검출 시 종료
    
    // 의사 검출 (pseudo-detection) 사용 여부
    private var usePseudoDetection = true
    
    fun processFrame(
        frame: Bitmap,
        rvec: FloatArray,
        tvec: FloatArray,
        timestampMs: Long
    ): TrackingResult {
        // 1. 딥러닝 검출
        val detections = detector.detect(frame)
        
        when (state) {
            TrackingState.IDLE -> {
                // 첫 검출 대기
                if (detections.isNotEmpty()) {
                    val det = detections[0]
                    val pos3D = estimate3DPosition(det, rvec, tvec)
                    
                    // 마운드 근처에서 시작했는지 확인
                    if (pos3D[1] > 15f) {  // y > 15m (마운드 근처)
                        kalman.initialize(pos3D)
                        trajectory.add(Point3D(pos3D[0], pos3D[1], pos3D[2], timestampMs))
                        state = TrackingState.TRACKING
                        missedFrames = 0
                    }
                }
            }
            
            TrackingState.TRACKING -> {
                // 예측
                val dt = 1f / 60f
                val predicted = kalman.predict(dt)
                
                if (detections.isNotEmpty()) {
                    // 검출 성공
                    val det = detections[0]
                    val measured = estimate3DPosition(det, rvec, tvec)
                    
                    // 게이팅 적용 업데이트
                    val updated = kalman.updateWithGating(measured)
                    
                    if (updated != null) {
                        trajectory.add(Point3D(updated[0], updated[1], updated[2], timestampMs))
                        missedFrames = 0
                    } else {
                        // 게이팅 실패 → 의사 검출 사용
                        handleMissedDetection(predicted, timestampMs)
                    }
                } else {
                    // 검출 실패
                    handleMissedDetection(predicted, timestampMs)
                }
                
                // 홈플레이트 통과 확인
                val currentPos = trajectory.lastOrNull()
                if (currentPos != null && currentPos.y < 0.5f) {
                    state = TrackingState.COMPLETED
                }
                
                // 너무 많이 놓쳤으면 종료
                if (missedFrames > maxMissedFrames) {
                    state = TrackingState.LOST
                }
            }
            
            else -> { /* LOST, COMPLETED: 처리 없음 */ }
        }
        
        return TrackingResult(
            state = state,
            trajectory = trajectory.toList(),
            currentSpeed = kalman.getSpeedKmh(),
            predicted = if (state == TrackingState.TRACKING) 
                kalman.getPredictedTrajectory(10, 1f/60f) 
                else emptyList()
        )
    }
    
    private fun handleMissedDetection(predicted: FloatArray, timestampMs: Long) {
        missedFrames++
        
        if (usePseudoDetection && missedFrames <= 3) {
            // 의사 검출: 예측 위치를 궤적에 추가
            trajectory.add(Point3D(predicted[0], predicted[1], predicted[2], timestampMs))
            Log.d("Tracker", "Using pseudo-detection at frame $missedFrames")
        }
    }
    
    private fun estimate3DPosition(
        detection: Detection,
        rvec: FloatArray,
        tvec: FloatArray
    ): FloatArray {
        // 깊이 추정
        val pos3D = coordinator.estimateDepth(detection)
        // 마커 좌표계로 변환
        return coordinator.transformToMarkerCoord(pos3D, rvec, tvec)
    }
    
    fun reset() {
        state = TrackingState.IDLE
        trajectory.clear()
        missedFrames = 0
    }
    
    fun getJudgment(strikeZone: StrikeZone): PitchJudgment? {
        if (state != TrackingState.COMPLETED) return null
        
        // 존 통과 지점 찾기
        val crossingPoint = findCrossingPoint(trajectory, strikeZone.frontPlaneY)
        
        return if (crossingPoint != null && strikeZone.isInZone(crossingPoint)) {
            PitchJudgment.STRIKE
        } else {
            PitchJudgment.BALL
        }
    }
    
    private fun findCrossingPoint(trajectory: List<Point3D>, planeY: Float): Point3D? {
        for (i in 1 until trajectory.size) {
            val prev = trajectory[i - 1]
            val curr = trajectory[i]
            
            // Y가 planeY를 통과했는지 확인
            if (prev.y > planeY && curr.y <= planeY) {
                // 선형 보간
                val t = (planeY - prev.y) / (curr.y - prev.y)
                return Point3D(
                    x = prev.x + t * (curr.x - prev.x),
                    y = planeY,
                    z = prev.z + t * (curr.z - prev.z),
                    timestamp = (prev.timestamp + t * (curr.timestamp - prev.timestamp)).toLong()
                )
            }
        }
        return null
    }
}

data class TrackingResult(
    val state: PitchTracker.TrackingState,
    val trajectory: List<Point3D>,
    val currentSpeed: Float,
    val predicted: List<FloatArray>
)
```

---

## 7. Phase 5: 고급 데이터 증강 (2주)

### 7.1 모션 블러 합성

```python
# augmentation/motion_blur.py
import cv2
import numpy as np
from typing import Tuple

def apply_motion_blur(
    image: np.ndarray,
    ball_center: Tuple[int, int],
    ball_radius: int,
    blur_length: int = 15,
    blur_angle: float = 0.0
) -> np.ndarray:
    """
    공 영역에만 모션 블러 적용
    
    Args:
        image: 입력 이미지
        ball_center: 공 중심 좌표 (x, y)
        ball_radius: 공 반지름 (픽셀)
        blur_length: 블러 길이 (픽셀)
        blur_angle: 블러 방향 (라디안)
    """
    h, w = image.shape[:2]
    
    # 모션 블러 커널 생성
    kernel_size = blur_length
    kernel = np.zeros((kernel_size, kernel_size))
    
    # 방향에 따른 커널 생성
    center = kernel_size // 2
    cos_a = np.cos(blur_angle)
    sin_a = np.sin(blur_angle)
    
    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * cos_a)
        y = int(center + offset * sin_a)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    kernel = kernel / kernel.sum()
    
    # 공 영역 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, ball_center, int(ball_radius * 1.5), 255, -1)
    
    # 블러 적용
    blurred = cv2.filter2D(image, -1, kernel)
    
    # 마스크 영역만 블러 적용
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = image * (1 - mask_3ch) + blurred * mask_3ch
    
    return result.astype(np.uint8)


def generate_motion_blur_sequence(
    background: np.ndarray,
    ball_image: np.ndarray,
    start_pos: Tuple[int, int],
    end_pos: Tuple[int, int],
    num_frames: int = 30,
    ball_radius: int = 15
) -> list:
    """
    모션 블러가 적용된 공 이동 시퀀스 생성
    """
    frames = []
    positions = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        # 포물선 궤적 (중력 효과)
        x = int(start_pos[0] + t * (end_pos[0] - start_pos[0]))
        y = int(start_pos[1] + t * (end_pos[1] - start_pos[1]) + 
                0.5 * 9.81 * (t * 0.5) ** 2 * 100)  # 중력 효과
        
        # 배경에 공 합성
        frame = background.copy()
        
        # 공 붙여넣기
        ball_h, ball_w = ball_image.shape[:2]
        y1 = max(0, y - ball_h // 2)
        y2 = min(frame.shape[0], y + ball_h // 2)
        x1 = max(0, x - ball_w // 2)
        x2 = min(frame.shape[1], x + ball_w // 2)
        
        # 알파 블렌딩
        if ball_image.shape[2] == 4:
            alpha = ball_image[:, :, 3:4] / 255.0
            frame[y1:y2, x1:x2] = (
                frame[y1:y2, x1:x2] * (1 - alpha[:y2-y1, :x2-x1]) +
                ball_image[:y2-y1, :x2-x1, :3] * alpha[:y2-y1, :x2-x1]
            )
        
        # 모션 블러 적용
        if i > 0:
            prev_pos = positions[-1]
            angle = np.arctan2(y - prev_pos[1], x - prev_pos[0])
            distance = np.sqrt((x - prev_pos[0])**2 + (y - prev_pos[1])**2)
            blur_length = min(int(distance * 0.8), 20)
            
            if blur_length > 3:
                frame = apply_motion_blur(
                    frame, (x, y), ball_radius, blur_length, angle
                )
        
        frames.append(frame)
        positions.append((x, y))
    
    return frames, positions
```

### 7.2 배경 합성 (Copy-Paste)

```python
# augmentation/copy_paste.py
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

class BallCopyPaste:
    """공을 다양한 배경에 복사-붙여넣기"""
    
    def __init__(self, ball_templates_dir: str, backgrounds_dir: str):
        self.ball_templates = self._load_templates(ball_templates_dir)
        self.backgrounds = self._load_backgrounds(backgrounds_dir)
        
        # 색상/밝기 증강
        self.color_aug = A.Compose([
            A.RandomBrightnessContrast(p=0.8),
            A.HueSaturationValue(p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
        ])
    
    def _load_templates(self, dir_path: str) -> list:
        templates = []
        for p in Path(dir_path).glob("*.png"):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None:
                templates.append(img)
        return templates
    
    def _load_backgrounds(self, dir_path: str) -> list:
        backgrounds = []
        for p in Path(dir_path).glob("*.jpg"):
            img = cv2.imread(str(p))
            if img is not None:
                backgrounds.append(img)
        return backgrounds
    
    def generate(
        self,
        num_samples: int = 100,
        output_size: Tuple[int, int] = (1920, 1080)
    ) -> list:
        """합성 이미지 및 라벨 생성"""
        
        samples = []
        
        for _ in range(num_samples):
            # 랜덤 배경 선택
            bg = np.random.choice(self.backgrounds).copy()
            bg = cv2.resize(bg, output_size)
            
            # 랜덤 공 템플릿 선택
            ball = np.random.choice(self.ball_templates).copy()
            
            # 공 크기 조정 (거리에 따른 크기 변화 시뮬레이션)
            scale = np.random.uniform(0.5, 2.0)
            new_size = (int(ball.shape[1] * scale), int(ball.shape[0] * scale))
            ball = cv2.resize(ball, new_size)
            
            # 랜덤 위치
            max_x = output_size[0] - ball.shape[1]
            max_y = output_size[1] - ball.shape[0]
            x = np.random.randint(0, max(1, max_x))
            y = np.random.randint(0, max(1, max_y))
            
            # 색상 증강
            ball_rgb = ball[:, :, :3]
            ball_rgb = self.color_aug(image=ball_rgb)['image']
            ball[:, :, :3] = ball_rgb
            
            # 합성
            result = self._paste_ball(bg, ball, x, y)
            
            # 라벨 생성 (YOLO 형식)
            cx = (x + ball.shape[1] / 2) / output_size[0]
            cy = (y + ball.shape[0] / 2) / output_size[1]
            w = ball.shape[1] / output_size[0]
            h = ball.shape[0] / output_size[1]
            
            samples.append({
                'image': result,
                'label': f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            })
        
        return samples
    
    def _paste_ball(
        self,
        background: np.ndarray,
        ball: np.ndarray,
        x: int,
        y: int
    ) -> np.ndarray:
        """알파 블렌딩으로 공 합성"""
        
        result = background.copy()
        bh, bw = ball.shape[:2]
        
        # 경계 체크
        x1, x2 = max(0, x), min(background.shape[1], x + bw)
        y1, y2 = max(0, y), min(background.shape[0], y + bh)
        
        bx1, bx2 = max(0, -x), bw - max(0, x + bw - background.shape[1])
        by1, by2 = max(0, -y), bh - max(0, y + bh - background.shape[0])
        
        if ball.shape[2] == 4:
            alpha = ball[by1:by2, bx1:bx2, 3:4] / 255.0
            result[y1:y2, x1:x2] = (
                result[y1:y2, x1:x2] * (1 - alpha) +
                ball[by1:by2, bx1:bx2, :3] * alpha
            ).astype(np.uint8)
        else:
            result[y1:y2, x1:x2] = ball[by1:by2, bx1:bx2]
        
        return result
```

### 7.3 프레임 차분 채널

```python
# augmentation/frame_difference.py
import cv2
import numpy as np

class FrameDifferenceChannel:
    """
    연속 프레임 차분을 추가 입력 채널로 사용
    - 움직이는 공을 강조
    - 정적 배경 제거 효과
    """
    
    def __init__(self, threshold: int = 30):
        self.prev_frame = None
        self.threshold = threshold
    
    def compute(self, frame: np.ndarray) -> np.ndarray:
        """
        현재 프레임과 이전 프레임의 차분 계산
        
        Returns:
            4채널 이미지 (RGB + Difference)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            diff = np.zeros_like(gray)
        else:
            # 절대 차분
            diff = cv2.absdiff(gray, self.prev_frame)
            
            # 임계값 적용
            _, diff = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = np.ones((3, 3), np.uint8)
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
            diff = cv2.dilate(diff, kernel, iterations=1)
            
            self.prev_frame = gray
        
        # 4채널로 결합
        result = np.dstack([frame, diff])
        return result
    
    def reset(self):
        self.prev_frame = None


def create_4channel_dataset(
    video_path: str,
    output_dir: str,
    labels_dir: str
):
    """비디오에서 4채널 학습 데이터 생성"""
    
    cap = cv2.VideoCapture(video_path)
    diff_channel = FrameDifferenceChannel()
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 4채널 이미지 생성
        frame_4ch = diff_channel.compute(frame)
        
        # RGB 이미지 저장
        rgb_path = f"{output_dir}/frame_{frame_idx:06d}.jpg"
        cv2.imwrite(rgb_path, frame)
        
        # 차분 채널 저장 (별도 파일)
        diff_path = f"{output_dir}/frame_{frame_idx:06d}_diff.jpg"
        cv2.imwrite(diff_path, frame_4ch[:, :, 3])
        
        frame_idx += 1
    
    cap.release()
```

### 7.4 증강 파이프라인 통합

```python
# augmentation/pipeline.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size: int = 416):
    """학습용 증강 파이프라인"""
    
    return A.Compose([
        # 기하학적 변환
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=0,  # 회전은 안 함 (공은 구형)
            p=0.5
        ),
        
        # 색상 변환
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30),
        ], p=0.8),
        
        # 날씨/조명 시뮬레이션
        A.OneOf([
            A.RandomSunFlare(src_radius=100, p=0.2),
            A.RandomShadow(p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        ], p=0.3),
        
        # 노이즈
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.ISONoise(),
            A.MultiplicativeNoise(),
        ], p=0.3),
        
        # 블러 (모션 블러 포함)
        A.OneOf([
            A.MotionBlur(blur_limit=7),
            A.GaussianBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        
        # 리사이즈 및 정규화
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))


def get_val_transforms(img_size: int = 416):
    """검증용 변환 (증강 없음)"""
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))
```

---

## 8. 필수 참고 논문 목록

### 8.1 공 검출 및 추적

| 논문 | 핵심 내용 | 관련성 |
|------|----------|--------|
| **TrackNet (AVSS 2019)** | 테니스/배드민턴 공 추적용 CNN, 히트맵 출력 | ⭐⭐⭐⭐⭐ |
| "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications" | 연속 프레임 입력, 가우시안 히트맵 예측 | 공 추적 특화 |
| **MonoTrack (CVPR 2023)** | 단일 카메라 3D 공 추적 | ⭐⭐⭐⭐⭐ |
| "MonoTrack: Shuttle Trajectory Reconstruction from Monocular Badminton Video" | 핀홀 모델 + 물리 기반 궤적 추정 | 깊이 추정 참고 |
| **Ball 3D Localization (WACV 2024)** | 단일 이미지에서 공 3D 위치 추정 | ⭐⭐⭐⭐ |

### 8.2 소형 객체 검출

| 논문 | 핵심 내용 | 관련성 |
|------|----------|--------|
| **SEMA-YOLO (2025)** | 얕은 층 강화, RFA 모듈 | ⭐⭐⭐⭐⭐ |
| "SEMA-YOLO: Small Object Detection Enhanced with Multi-Scale Attention" | P4 헤드 추가, 다중스케일 적응 | 아키텍처 참고 |
| **MDSF-YOLO (2024)** | 다중 스케일 팽창 융합 | ⭐⭐⭐⭐ |
| "Small Object Detection with Multi-scale Dilated Sequence Fusion" | 팽창 합성곱으로 컨텍스트 확장 | 피처 융합 참고 |
| **TPH-YOLOv5 (2021)** | 트랜스포머 + YOLO | ⭐⭐⭐ |

### 8.3 물리 기반 추적

| 논문 | 핵심 내용 | 관련성 |
|------|----------|--------|
| **PhyOT (NeurIPS 2023)** | 물리 엔진 + 칼만 필터 + 딥러닝 | ⭐⭐⭐⭐⭐ |
| "Physics-Informed Object Tracking" | 뉴턴 역학 통합, 오탐 제거 | 핵심 참고 |
| **KalmanFormer (2025)** | 트랜스포머로 칼만 필터 보정 | ⭐⭐⭐⭐ |
| "KalmanFormer: SORT with Deep Learning Motion Model" | 비선형 운동 학습, 의사검출 생성 | 폐루프 참고 |
| **Singh et al. (2025)** | YOLO + 운동학 모델 | ⭐⭐⭐⭐⭐ |
| "Hybrid CNN-Kinematics Tracker for Fast Moving Objects" | 70% 추적 오차 감소 | 직접 관련 |

### 8.4 데이터 증강

| 논문 | 핵심 내용 | 관련성 |
|------|----------|--------|
| **Hiemann et al. (2021)** | 스포츠 공 전용 증강 | ⭐⭐⭐⭐⭐ |
| "Ball Detection in Beach Volleyball with Domain-specific Augmentation" | 프레임 차분 채널, 물리 기반 합성 | 직접 참고 |
| **Copy-Paste (CVPR 2021)** | 인스턴스 복사-붙여넣기 증강 | ⭐⭐⭐⭐ |
| **MixUp / Mosaic** | YOLO 기본 증강 | ⭐⭐⭐ |

### 8.5 모바일 딥러닝 최적화

| 논문/자료 | 핵심 내용 | 관련성 |
|------|----------|--------|
| **YOLOv8 (Ultralytics 2023)** | 최신 YOLO 아키텍처 | ⭐⭐⭐⭐⭐ |
| **TensorFlow Lite 가이드** | INT8 양자화, GPU delegate | ⭐⭐⭐⭐⭐ |
| **EfficientDet (CVPR 2020)** | 효율적인 피처 융합 | ⭐⭐⭐⭐ |
| **MobileNetV3 (ICCV 2019)** | 모바일 백본 | ⭐⭐⭐ |

### 8.6 논문 다운로드 링크

```markdown
## 핵심 논문 (필독)

1. TrackNet
   - arXiv: https://arxiv.org/abs/1907.03698
   - GitHub: https://github.com/ChgygLin/TrackNet

2. PhyOT (Physics-Informed Object Tracking)
   - arXiv: https://arxiv.org/abs/2312.08650

3. Ball Detection with Domain-specific Augmentation
   - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8124271/

4. YOLOv8
   - Docs: https://docs.ultralytics.com/
   - GitHub: https://github.com/ultralytics/ultralytics

5. SEMA-YOLO
   - MDPI: https://www.mdpi.com/2072-4292/17/11/1917

## 보조 자료

6. TensorFlow Lite 최적화 가이드
   - https://www.tensorflow.org/lite/performance/best_practices

7. Android CameraX 문서
   - https://developer.android.com/training/camerax

8. BaseballCV 오픈소스 데이터셋
   - https://github.com/BaseballCV/BaseballCV
```

---

## 9. 기술 스택 상세

### 9.1 전체 스택 요약

```
┌─────────────────────────────────────────────────────────────┐
│                     기술 스택 Overview                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📱 Mobile (Android)                                         │
│  ├── Language: Kotlin                                        │
│  ├── Camera: CameraX (Jetpack)                               │
│  ├── ML: TensorFlow Lite 2.14 + GPU Delegate                 │
│  ├── CV: OpenCV 4.8 (ArUco)                                  │
│  ├── Network: OkHttp + WebSocket                             │
│  └── TTS: Android TextToSpeech (Offline)                     │
│                                                              │
│  🖥️ Backend                                                   │
│  ├── Framework: FastAPI (Python 3.11)                        │
│  ├── WebSocket: Starlette                                    │
│  ├── Database: PostgreSQL 15 + TimescaleDB                   │
│  ├── Cache: Redis 7                                          │
│  ├── Auth: JWT (PyJWT)                                       │
│  └── Container: Docker + Docker Compose                      │
│                                                              │
│  🌐 Frontend (Web)                                            │
│  ├── Framework: React 18 + TypeScript                        │
│  ├── 3D: Three.js                                            │
│  ├── Charts: Plotly.js                                       │
│  ├── State: Zustand                                          │
│  ├── Styling: TailwindCSS                                    │
│  └── Build: Vite                                             │
│                                                              │
│  🔬 ML Training                                               │
│  ├── Framework: PyTorch + Ultralytics                        │
│  ├── Augmentation: Albumentations                            │
│  ├── Experiment: Weights & Biases                            │
│  └── Export: ONNX → TFLite                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 버전 및 의존성

```yaml
# 모바일 (build.gradle)
android:
  compileSdk: 34
  minSdk: 26
  targetSdk: 34

dependencies:
  camerax: 1.3.1
  tensorflow-lite: 2.14.0
  tensorflow-lite-gpu: 2.14.0
  opencv: 4.8.0
  okhttp: 4.12.0
  gson: 2.10.1
  coroutines: 1.7.3

# 백엔드 (requirements.txt)
python: ">=3.11"
fastapi: ">=0.104.0"
uvicorn: ">=0.24.0"
sqlalchemy: ">=2.0.0"
asyncpg: ">=0.29.0"
redis: ">=5.0.0"
pyjwt: ">=2.8.0"
pydantic: ">=2.5.0"

# 프론트엔드 (package.json)
node: ">=18"
react: "^18.2.0"
three: "^0.159.0"
plotly.js: "^2.27.0"
zustand: "^4.4.0"
tailwindcss: "^3.3.0"

# ML 학습 (requirements-ml.txt)
torch: ">=2.1.0"
ultralytics: ">=8.0.200"
albumentations: ">=1.3.0"
opencv-python: ">=4.8.0"
wandb: ">=0.16.0"
```

---

## 10. 실험 설계 및 평가

### 10.1 실험 체크리스트

```markdown
## 필수 실험 (Ablation Study)

### A. 검출 모델 비교
- [ ] YOLOv8n vs YOLOv8s (정확도/속도 trade-off)
- [ ] 입력 해상도: 416 vs 512 vs 640
- [ ] 양자화: FP32 vs FP16 vs INT8

### B. 증강 효과
- [ ] 기본 증강만 vs +모션블러 vs +배경합성 vs +프레임차분
- [ ] 증강 강도별 비교

### C. 물리 기반 추적
- [ ] 단순 칼만 vs 중력 포함 칼만 vs 폐루프
- [ ] 게이팅 임계값별 비교
- [ ] 의사검출 사용 여부

### D. 시스템 통합
- [ ] End-to-end 지연 시간
- [ ] 네트워크 상태별 성능 (좋음/보통/나쁨)
```

### 10.2 평가 시나리오

```markdown
## 테스트 시나리오

### 시나리오 1: 이상적 조건
- 맑은 낮, 정면 카메라, 단색 배경
- 목표: mAP > 90%, Recall > 95%

### 시나리오 2: 까다로운 조건
- 역광, 흰색 유니폼, 관중석 배경
- 목표: mAP > 75%, Recall > 85%

### 시나리오 3: 실사용 조건
- 야외 훈련장, 다양한 투수
- 목표: 판정 정확도 > 85%, 지연 < 200ms

### 시나리오 4: 스트레스 테스트
- 연속 100구 처리
- 목표: 크래시 없음, 메모리 누수 없음
```

### 10.3 결과 기록 템플릿

```markdown
## 실험 결과 기록

### 실험 ID: EXP-001
- 날짜: 2024-XX-XX
- 목적: YOLOv8n INT8 성능 검증

### 설정
- 모델: YOLOv8n
- 양자화: INT8
- 입력: 416x416
- 디바이스: Pixel 6

### 결과
| 지표 | 값 | 목표 | 달성 |
|------|-----|------|------|
| mAP@0.5 | 0.XX | 0.85 | ✅/❌ |
| Recall | 0.XX | 0.90 | ✅/❌ |
| FPS | XX | 25 | ✅/❌ |
| Latency | XXms | 35ms | ✅/❌ |

### 관찰
- ...

### 다음 단계
- ...
```

---

## 마무리

이 가이드를 따라 단계별로 구현하면:

1. **Phase 0-1 (3-4주)**: 데이터 + 모델 → 작동하는 검출기
2. **Phase 2 (3-4주)**: 안드로이드 앱 → 현장 테스트 가능
3. **Phase 3 (3-4주)**: 서버 + 웹 → 완전한 시스템
4. **Phase 4-5 (4주)**: 물리 기반 + 증강 → 논문급 성능

**총 예상 기간: 약 3-4개월**

질문이나 막히는 부분이 있으면 언제든 물어봐! 🚀

