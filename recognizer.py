import cv2
import face_recognition
import pickle
import sys
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import cv2, numpy as np, pathlib, pickle, os
import matplotlib.pyplot as plt

# ===== 1. 基本參數 =====
VIDEO_IN   = ""
VIDEO_OUT  = ""
DB_PKL     = "face_db.pkl"
CONF       = 0.30
TOLERANCE  = 0.4  # 餘弦相似度越接近1越相似
MARGIN     = 0.4
# 直接從 Hugging Face 取權重
weight = hf_hub_download("arnabdhar/YOLOv8-Face-Detection", "model.pt")

# ===== 2. 載入模型 & 資料庫 =====
print("▶  載入 YOLOv8-Face 與 InsightFace …")
face_yolo = YOLO(weight)
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

db_vecs, db_names = None, None

# ===== 4. 取得向量（ArcFace） =====
def get_embedding(img_bgr, bbox, margin=MARGIN):
    x1,y1,x2,y2 = bbox
    h, w = img_bgr.shape[:2]
    dx, dy = int((x2-x1)*margin), int((y2-y1)*margin)
    l = max(x1-dx, 0); t = max(y1-dy, 0)
    r = min(x2+dx, w); b = min(y2+dy, h)
    face_crop = img_bgr[t:b, l:r]
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    faces = face_app.get(face_rgb)
    if faces:
        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)
        return emb
    return None


if __name__ == "__main__":
    VIDEO_IN = sys.argv[1]
    VIDEO_OUT = sys.argv[2]
    if len(sys.argv) != 3:
        print("用法：python recognizer.py 輸入影片.mp4 輸出影片.mp4")
    else:

        if os.path.exists(DB_PKL):
            db = pickle.load(open(DB_PKL, "rb"))
            db_vecs  = np.stack([d["vec"] for d in db])
            db_names = [d["name"] for d in db]
            print(f"▶  已載入人臉資料庫：{len(db_names)} 張向量")
        else:
            print("⚠️  找不到 face_db.pkl，將只標示 Unknown")

        # ===== 3. 開啟影片 =====
        cap = cv2.VideoCapture(VIDEO_IN)
        if not cap.isOpened():
            raise FileNotFoundError(f"無法開啟影片：{VIDEO_IN}")

        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourc = cv2.VideoWriter_fourcc(*"mp4v")
        out   = cv2.VideoWriter(VIDEO_OUT, fourc, fps, (w, h))


        # ===== 5. 主迴圈 =====
        print("🚀  開始辨識 … (ESC 離開)")
        plt.ion()  # 開啟即時互動模式
        fig, ax = plt.subplots()
        img_plot = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 5-1 YOLO 偵測
            res = face_yolo(frame, conf=CONF, verbose=False)[0]

            # 5-2 對每個 bbox 做編碼 & 比對
            for box in res.boxes.xyxy.cpu().numpy():
                x1,y1,x2,y2 = map(int, box)
                emb = get_embedding(frame, (x1, y1, x2, y2))
                if emb is None:
                    continue

                # 比對資料庫（餘弦相似度）
                name, score_txt = "Unknown", ""
                if db_vecs is not None:
                    sims = db_vecs @ emb  # 餘弦相似度
                    idx = np.argmax(sims)
                    if sims[idx] > TOLERANCE:
                        name = db_names[idx]
                    score_txt = f"{sims[idx]:.2f}"

                # 畫框 + 標籤
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{name} {score_txt}",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2, cv2.LINE_AA)

            # 5-3 寫入畫面
            out.write(frame)

            # 5-4 使用 matplotlib 顯示畫面
            cv2.imshow("YOLO-Face-Match", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 離開
                break

        # ===== 6. 收尾 =====
        cap.release(); out.release(); plt.ioff(); plt.close()
        print(f"✅  輸出完成 → {pathlib.Path(VIDEO_OUT).resolve()}")