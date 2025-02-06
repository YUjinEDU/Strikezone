import cv2
import numpy as np
import onnxruntime  # pip install onnxruntime
import time

# ------------------------------
# (1) 딥러닝 검출(ONNX) 관련 초기화
# ------------------------------
class YoloOnnxDetector:
    def __init__(self, onnx_path="yolov5n.onnx", input_size=(640,640)):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_size = input_size
        # 모델에 따라 input/output name, 전처리 다를 수 있음
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        """
        frame: BGR np.array
        return: [(x1,y1,x2,y2, conf), ...] 형태의 bounding box 리스트 (공만 필터링했다고 가정)
        """
        # 1) 전처리 (단순 resize, normalize 등)
        img = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))  # (3,H,W)
        img = np.expand_dims(img, 0)      # (1,3,H,W)

        # 2) 추론
        outputs = self.session.run(None, {self.input_name: img})

        # 예) yolov5 ONNX 결과가 [batch, #boxes, 6=(x1,y1,x2,y2,conf,class)] 라고 가정
        # 세부사항은 모델 구조마다 달라질 수 있음
        preds = outputs[0][0]  # (n, 6)
        #print("Detected:", preds)
        # 3) 결과 중 '공' 클래스를 찾거나, confidence 높은 바운딩박스만 리턴
        # 실제론 class id로 'ball' 필터링 or score threshold
        preds[..., 4] = 1 / (1 + np.exp(-preds[..., 4]))  # conf 확률값 보정
        preds[..., 5] = 1 / (1 + np.exp(-preds[..., 5]))  # cls_id 확률값 보정


        bboxes = []
        conf_thresh = 0.0001
        for det in preds:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            print("conf, cls_id", conf, cls_id)
            if conf < conf_thresh:
                continue
  
            if int(cls_id) == 32:  # <- '공' 클래스 ID
                bboxes.append((x1, y1, x2, y2, conf))

        # 원본 프레임 크기로 복원(640->원본)
        # NOTE: 실제론 scale factor 계산 필요
        H, W = frame.shape[:2]
        sx = W / self.input_size[0]
        sy = H / self.input_size[1]

        final_boxes = []
        for (x1,y1,x2,y2,conf) in bboxes:
            # scale up
            fx1 = int(x1*sx); fy1 = int(y1*sy)
            fx2 = int(x2*sx); fy2 = int(y2*sy)
            final_boxes.append((fx1, fy1, fx2, fy2, conf))

        return final_boxes

# ------------------------------
# (2) 하이브리드 구조
# ------------------------------
def main():
    # 1) 카메라/영상
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Camera not opened!")
        return

    # 2) 딥러닝 초기화
    detector = YoloOnnxDetector("yolov5n.onnx", (640,640))

    # 3) 추적기(KCF 등)
    tracker = None
    tracking = False
    detect_interval = 30  # N프레임마다 재검출
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # (A) N프레임마다 딥러닝 검출
        if (frame_count % detect_interval == 1) or (not tracking):
            bboxes = detector.detect(frame)
            # 가장 conf 높은 box 1개만 사용한다고 가정
            if len(bboxes)>0:
                best = max(bboxes, key=lambda b: b[4]) # conf 기준
                x1,y1,x2,y2,conf = best
                w = x2-x1; h=y2-y1
                if w>0 and h>0:
                    # 새 추적기 초기화
                    tracker = cv2.legacy.TrackerKCF_create()  # 혹은 CSRT, MOSSE 등
                    tracker.init(frame, (x1,y1,w,h))
                    tracking = True
                else:
                    tracking = False
            else:
                tracking = False

        # (B) 추적 단계
        if tracking and tracker is not None:
            ok, bbox = tracker.update(frame)
            if ok:
                (x,y,w,h) = bbox
                x2 = x+w; y2=y+h
                cv2.rectangle(frame, (int(x),int(y)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.putText(frame, "Tracking", (int(x),int(y)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                tracking = False
                cv2.putText(frame, "Tracking Lost", (30,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        # 표시
        cv2.imshow("Hybrid Demo", frame)
        key = cv2.waitKey(1)
        if key==27 or key==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
