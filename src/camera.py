import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
import imutils
import sys
import logging


class CameraManager:
    """카메라 관리 및 설정을 담당하는 클래스"""
    
    def __init__(self, shutdown_event):
        """
        Args:
            shutdown_event: 프로그램 종료 이벤트
        """
        self.shutdown_event = shutdown_event
        self.selected_camera_index = None
        self.calibration_data = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.capture = None
    
    def get_camera_list(self):
        """
        사용 가능한 카메라 목록 검색
        
        Returns:
            카메라 목록 (예: ['Camera 0', 'Camera 1', ...])
        """
        camera_list = []
        for index in range(10):
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if cap.isOpened():
                camera_list.append(f"Camera {index}")
                cap.release()
        return camera_list
    
    def create_camera_selection_gui(self):
        """
        카메라 선택 GUI 생성
        
        Returns:
            선택된 카메라 인덱스
        """
        
        try:
            window = tk.Tk()
            window.title("카메라 선택")
            window.geometry("1280x720")
            
            cameras = self.get_camera_list()
            if not cameras:
                print("사용 가능한 카메라가 없습니다")
                return None
            
            n_cameras = len(cameras)
            cols = min(3, n_cameras)
            rows = (n_cameras + cols - 1) // cols
            
            previews = []
            for i, camera in enumerate(cameras):
                idx = int(camera.split()[1])
                preview = CameraPreview(window, idx, self)
                preview.frame.grid(row=i//cols, column=i%cols, padx=10, pady=10)
                previews.append(preview)
            
            def on_closing():
                # 모든 카메라 미리보기 스레드 중지
                for preview in previews:
                    preview.stop()
                self.shutdown_event.set()  # 전체 종료 신호 전파
                window.destroy()
                # cv2 창 종료 및 프로세스 종료
                cv2.destroyAllWindows()
                logging.shutdown()
                sys.exit(0)
            
            window.protocol("WM_DELETE_WINDOW", on_closing)
            window.mainloop()
            
            for preview in previews:
                preview.stop()
            
            return self.selected_camera_index
    
        except Exception as e:
            print(f"ERROR: 카메라 선택 GUI 에러: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def set_selected_camera(self, index):
        """
        선택된 카메라 인덱스 설정
        
        Args:
            index: 선택된 카메라 인덱스
        """
        self.selected_camera_index = index
    
    def load_calibration(self, calibration_path):
        """
        카메라 캘리브레이션 데이터 로드
        
        Args:
            calibration_path: 캘리브레이션 파일 경로
        
        Returns:
            성공 여부
        """
        try:
            self.calibration_data = np.load(calibration_path)
            self.camera_matrix = self.calibration_data["camera_matrix"]
            self.dist_coeffs = self.calibration_data["dist_coeffs"]
            return True
        except Exception as e:
            print(f"캘리브레이션 데이터 로드 실패: {e}")
            return False
    
    def open_camera(self, camera_index=None):
        """
        카메라 열기
        
        Args:
            camera_index: 열 카메라 인덱스 (None이면 선택된 카메라 사용)
        
        Returns:
            성공 여부
        """
        if camera_index is None:
            camera_index = self.selected_camera_index
        
        if camera_index is None:
            print("카메라가 선택되지 않았습니다")
            return False
        
        # Scrcpy 전용 INDEX
        # self.capture = cv2.VideoCapture(5, cv2.CAP_DSHOW)
        
        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
        self.capture.set(cv2.CAP_PROP_FPS, 90)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        print(f"FPS 설정: {fps}")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not self.capture.isOpened():
            print(f"카메라 {camera_index} 열기 실패")
            return False
        
        return True
    
    def read_frame(self):
        """
        카메라에서 프레임 읽기
        
        Returns:
            (ret, frame) 튜플 또는 None
        """
        if self.capture is None or not self.capture.isOpened():
            return None
        
        return self.capture.read()
    
    def release(self):
        """카메라 자원 해제"""
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class CameraPreview:
    """카메라 미리보기 클래스 (카메라 선택 GUI에서 사용)"""
    
    def __init__(self, parent, camera_index, camera_manager):
        """
        Args:
            parent: 부모 tkinter 위젯
            camera_index: 카메라 인덱스
            camera_manager: CameraManager 인스턴스
        """
        self.frame = ttk.Frame(parent)
        self.camera_index = camera_index
        self.camera_manager = camera_manager
        
        self.label = ttk.Label(self.frame, text=f"카메라 {camera_index}")
        self.preview = ttk.Label(self.frame)
        self.select_button = ttk.Button(
            self.frame,
            text="선택",
            command=self.on_select
        )
        
        self.label.pack(pady=5)
        self.preview.pack(pady=5)
        self.select_button.pack(pady=5)
        
        self.stop_event = threading.Event()
        self.image = None  # 이미지 참조 유지
        self.start_preview()
    
    def start_preview(self):
        """미리보기 스레드 시작"""
        def preview_thread():
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
            if not cap.isOpened():
                self.show_error()
                return
            
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.show_error()
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = imutils.resize(frame, width=320)
                image_data = cv2.imencode('.png', frame)[1].tobytes()
                try:
                    self.preview.after(0, lambda: self.update_preview(image_data))
                except RuntimeError:
                    break
            cap.release()
        
        self.preview_thread = threading.Thread(target=preview_thread, daemon=True)
        self.preview_thread.start()
    
    def update_preview(self, image_data):
        """
        미리보기 이미지 업데이트
        
        Args:
            image_data: 이미지 데이터 바이트
        """
        try:
            self.image = PhotoImage(data=image_data)  # 이미지 참조 유지
            self.preview.configure(image=self.image)
        except Exception:
            pass
    
    def show_error(self):
        """에러 표시 (검은 화면)"""
        black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        black_frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        image_data = cv2.imencode('.png', black_frame_rgb)[1].tobytes()
        try:
            self.preview.after(0, lambda: self.update_preview(image_data))
        except RuntimeError:
            pass
    
    def stop(self):
        """미리보기 중지"""
        self.stop_event.set()
        if hasattr(self, 'preview_thread'):
            self.preview_thread.join(timeout=1.0)
    
    def on_select(self):
        """카메라 선택 버튼 클릭 시 처리"""
        self.camera_manager.set_selected_camera(self.camera_index)
        self.frame.master.quit()  # 창 닫기