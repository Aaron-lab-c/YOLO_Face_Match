# 🧠 YOLO-Face-Match

> 使用 YOLOv8 與 InsightFace 構建的影片人臉辨識系統


\\

---

## ⚡ 專案概述

本專案提供一套簡潔的流程，可將人臉資料建立為向量資料庫，並對影片中出現的人臉進行即時比對與辨識
整合 YOLOv8-Face 作為人臉偵測器，搭配 InsightFace (ArcFace) 提取特徵向量
適用於監控影片、人員出入紀錄、自動標記等場景

---

## ✨ 主要特色

* 使用 Hugging Face 模型庫自動載入 YOLOv8-Face 權重
* 支援 ArcFace 特徵提取與比對（支援單張照片向量平均）
* 自動顯示辨識結果並可輸出新影片
* 命令列操作簡單明瞭

---

## 🚀 快速開始

### 📦 安裝方式

```bash
git clone https://github.com/yourname/YOLO-Face-Match.git
cd YOLO-Face-Match

pip install -r requirements.txt
# 或手動安裝必要套件：
pip install opencv-python insightface ultralytics face_recognition huggingface_hub matplotlib
```

### ⚡ 基本使用

1️⃣ 建立人臉向量資料庫（將人臉照片放入 `faces_db/人名/*.jpg`）：

```bash
python face_db.py
```

2️⃣ 辨識影片中人臉並輸出標記後影片：

```bash
python recognizer.py input_video.mp4 output_video.mp4
```

---

## 📋 功能特色

| 功能名稱      | 描述                      |
| --------- | ----------------------- |
| 人臉向量提取    | 使用 InsightFace ArcFace  |
| YOLOv8 偵測 | 取自 Hugging Face YOLO 模型 |
| 人臉比對      | 使用餘弦相似度進行比對             |
| 視覺化標記     | 使用 OpenCV 即時顯示與畫框       |
| 輸出影片      | 可自動輸出辨識結果之新影片           |

---

## 📚 完整文件

請參閱原始碼中註解

> TODO：補上 Wiki 連結與參數說明表格

---

## 💻 系統需求

| 項目     | 最低需求                    |
| ------ | ----------------------- |
| 作業系統   | Windows / macOS / Linux |
| Python | 3.8 或以上                 |
| 記憶體    | 建議至少3GB                 |
| 顯示卡    | 選用，支援 CPU 運算            |

---

## 🔧 建立與開發

### 👢 專案檔案結構

```text
YOLO-Face-Match/
├── face_db.py           # 建立人臉向量資料庫
├── recognizer.py        # 影片人臉辨識主程式
├── faces_db/            # 已知人員照片，子資料夾為人名
│   ├── alice/
│   └── bob/
├── face_db.pkl          # 儲存的人臉向量資料庫
├── input_video.mp4      # 測試用輸入影片
└── output_video.mp4     # 輸出結果影片
```

### 🚰 技術

| 類別    | 使用技術                                |
| ----- | ----------------------------------- |
| 偵測模型  | YOLOv8 Face Detection (HuggingFace) |
| 向量模型  | InsightFace / ArcFace               |
| 特徵比對  | 餘弦相似度                               |
| 顯示與輸出 | OpenCV, Matplotlib                  |
| 語言    | Python 3.8+                         |

---

## 📈 品質指標

* 單元測試：❌ TODO
* 自動化部署：❌ TODO
* 型別檢查：❌ TODO
* PEP8 相容：✅

---

## 🤝 貢獻指南

歡迎提問、回報 bug、提交 PR
請遵守以下規範：

### 🐛 問題回報

1. 清楚描述步驟與錯誤訊息
2. 附上重現步驟與環境資訊

---

## 📄 授權

📄 授權
本專案原始碼採用 MIT License
請注意：本專案引用的第三方模型（如 YOLOv8-Face 與 InsightFace）各自遵循其原始授權條款，使用時請依來源遵守

---

## 🙏 致謝

* [ultralytics/YOLOv8](https://github.com/ultralytics/ultralytics)
* [InsightFace](https://github.com/deepinsight/insightface)
* [face\_recognition](https://github.com/ageitgey/face_recognition)
* [Hugging Face Hub](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)

---

## 🔗 相關連結

* YOLOv8-Face 模型權重：[https://huggingface.co/arnabdhar/YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
* InsightFace 官方：[https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)
* 臉部辨識參考資料：[https://github.com/ageitgey/face\_recognition](https://github.com/ageitgey/face_recognition)

---

## ⚠️ 注意事項

* YOLOv8 模型下載可能需等幾秒
* `face_db.py` 預設會遍檢所有 `faces_db` 子資料夾
* 每位人物建議提供至少3張清晰照片（正臉）
* 若無 `face_db.pkl` 檔案，`recognizer.py` 僅會標記為 Unknown
* 請避免輸入解析度過大的影片，避免記憶體不足
# YOLO_Face_Match