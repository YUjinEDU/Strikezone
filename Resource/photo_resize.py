import cv2
import numpy as np
import os

# 체스보드 크기 설정 (내부 코너 수)
frame_size = (640, 480)  # 리사이징 후의 해상도

# 이미지 파일 경로
images = [
    'OpenCV_Python_2ndEdition_Examples/data/photo_(1).jpg',
    'OpenCV_Python_2ndEdition_Examples/data/photo_(2).jpg',
    'OpenCV_Python_2ndEdition_Examples/data/photo_(3).jpg',
    'OpenCV_Python_2ndEdition_Examples/data/photo_(4).jpg',
    'OpenCV_Python_2ndEdition_Examples/data/photo_(5).jpg'
]

# 리사이즈된 이미지를 저장할 디렉토리
save_dir = 'OpenCV_Python_2ndEdition_Examples/data/resized_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for image_file in images:
    img = cv2.imread(image_file)
    img_resized = cv2.resize(img, frame_size)  # 이미지 리사이징
    
    # 고유한 파일 이름 생성
    base_name = os.path.basename(image_file)
    save_path = os.path.join(save_dir, base_name)
    
    # 리사이즈된 이미지 저장
    success = cv2.imwrite(save_path, img_resized)
    if success:
        print(f"Resized image saved at {save_path}")
    else:
        print(f"Failed to save resized image at {save_path}")


