from ultralytics import YOLO
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2
from insightface.app import FaceAnalysis
import cv2, numpy as np, pathlib, pickle, os

# 直接從 Hugging Face 取權重
weight = hf_hub_download("arnabdhar/YOLOv8-Face-Detection", "model.pt")


model = YOLO(weight)      # 只會輸出一個類別：face

# --- 參數 ---
DB_DIR = "faces_db"          # 已知人員照片根目錄（子資料夾 = 人名）
OUT_PKL = "face_db.pkl"       # 輸出檔
# YOLO_WEIGHT = "yolov8n-face.pt"  # YOLO 模型名稱或路徑
CONF = 0.3                   # YOLO 置信度
MARGIN = 0.4                 # 外擴比例（避免裁太緊）

def encode_face(img_bgr, bbox, margin=0.25):
    x1, y1, x2, y2 = bbox
    H, W = img_bgr.shape[:2]

    # 1) 加 margin，並裁到畫面範圍內
    dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
    l = max(x1 - dx, 0)
    t = max(y1 - dy, 0)
    r = min(x2 + dx, W)
    b = min(y2 + dy, H)

    # ---------- 空框保護 ----------
    if r - l < 10 or b - t < 10:        # 幅度太小直接跳過
        return None

    face_crop = img_bgr[t:b, l:r]

    # 2) BGR → RGB，並確保連續 + uint8
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    # 3) 告訴 face_recognition 臉在整張 crop 內
    loc = [(0, rgb.shape[1], rgb.shape[0], 0)]  # (top, right, bottom, left)
    vecs = fr.face_encodings(rgb, known_face_locations=loc)

    return vecs[0] if vecs else None

# --- 載入模型 ---
print("▶ 載入 YOLOv8-Face 與 InsightFace 模型…")
face_yolo = YOLO(weight)
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

# --- 功能：YOLO 偵測 + InsightFace 向量 ---
def get_aligned_embedding(img_bgr, conf=CONF, margin=MARGIN):
    embeddings = []
    if img_bgr is None or img_bgr.size == 0:
        return embeddings

    h, w = img_bgr.shape[:2]
    results = face_yolo(img_bgr, conf=conf, verbose=False)[0]

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        dx, dy = int((x2 - x1) * margin), int((y2 - y1) * margin)
        l, t = max(x1 - dx, 0), max(y1 - dy, 0)
        r, b = min(x2 + dx, w), min(y2 + dy, h)

        if r - l < 10 or b - t < 10:
            continue

        face_crop = img_bgr[t:b, l:r]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        faces = face_app.get(face_rgb)
        if faces:
            emb = faces[0].embedding
            emb = emb / np.linalg.norm(emb)  # 單位化
            embeddings.append(emb)
    return embeddings
if __name__ == "__main__":
    print(f"▶ 開始處理資料夾 '{DB_DIR}' …")
    encodings = []
    for person_dir in pathlib.Path(DB_DIR).iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        print(f"--- 處理 {name} 的照片 ---")
        vectors = []

        for img_path in person_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[!] 無法讀取：{img_path}")
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim != 3:
                continue

            vecs = get_aligned_embedding(img)
            vectors.extend(vecs)

        if vectors:
            avg_vec = np.mean(vectors, axis=0)
            encodings.append({"name": name, "vec": avg_vec})

    print(f"✔ 收集完成，共 {len(encodings)} 個人員向量")
    try:
        with open(OUT_PKL, "wb") as f:
            pickle.dump(encodings, f)
        print(f"📦 向量庫輸出成功 → {OUT_PKL}")
    except Exception as e:
        print(f"[!] 寫入失敗：{e}")