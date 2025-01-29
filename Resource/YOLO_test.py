from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # COCO 사전학습된 모델

results = model.predict(source=1,  # 웹캠
                        conf=0., 
                        show=True,  # 예측결과 영상 바로 보여주기
                        classes=[32])  # COCO에서 'sports ball' 클래스 ID가 32번
print(results.pred[0])
# 'sports ball'만 검출
