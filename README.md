# 🧠 YOLO-Face-Match
> 以 **YOLOv8-Face** 結合 **InsightFace ArcFace** 打造的影片人臉辨識系統  
> Author • Aaron-lab-c

<p align="center">
  <a href="https://github.com/Aaron-lab-c/YOLO-Face-Match">
    <img src="https://img.shields.io/github/v/tag/Aaron-lab-c/YOLO-Face-Match?label=version" alt="Version Badge" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License" />
  </a>
  <a href="https://github.com/Aaron-lab-c/YOLO-Face-Match/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/Aaron-lab-c/YOLO-Face-Match/ci.yml?label=build" alt="Build Status" />
  </a>
  <a href="https://github.com/Aaron-lab-c/YOLO-Face-Match/stargazers">
    <img src="https://img.shields.io/github/stars/Aaron-lab-c/YOLO-Face-Match?style=social" alt="GitHub Stars" />
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+" />
  </a>
</p>

---

## ⚡ 專案概述
YOLO-Face-Match 提供一條龍流程  
先將已知人臉照片轉換為向量資料庫  
再於影片中即時比對並標註人臉  
適用於監控、門禁、影片標記等情境  

核心組件  
* **YOLOv8-Face** 偵測器（Hugging Face Model Hub 自動載入）  
* **InsightFace ArcFace** 特徵向量擷取與比對  

---

## ✨ 主要特色
| 特色 | 說明 |
| --- | --- |
| 📦 **零配置權重** | 自動從 Hugging Face 下載 YOLOv8-Face 權重 |
| 🔍 **精準辨識** | ArcFace 128-D 向量 支援多張平均化 |
| 🎬 **即時標註** | OpenCV 畫框與標籤 可另存新影片 |
| 🖥️ **CLI 友善** | face_db.py 建庫 recognizer.py 比對 一行命令完成 |
| 🧩 **模組化** | 偵測、特徵、比對、視覺化 各自封裝 方便擴充 |

---

## 🚀 快速開始

### 📦 安裝
```bash
git clone https://github.com/Aaron-lab-c/YOLO-Face-Match.git
cd YOLO-Face-Match
pip install -r requirements.txt        # 建議
# 或手動：
pip install ultralytics insightface opencv-python huggingface_hub matplotlib

⚡ 基本流程

1️⃣ 建立人臉向量資料庫
把人臉照片放入 faces_db/<person_name>/*.jpg

python face_db.py

2️⃣ 辨識影片並輸出結果

python recognizer.py input.mp4 output.mp4

預設輸出含框與姓名的 output.mp4 並於螢幕即時顯示

⸻

📋 功能一覽

功能 描述
人臉偵測 YOLOv8-Face
特徵擷取 InsightFace ArcFace
向量比對 餘弦相似度
視覺化 OpenCV 實時顯示與寫檔
影片輸出 ffmpeg-backend 重新編碼 MP4


⸻

📚 文件
 • 原始碼內 Docstring 與註解
 • Wiki （TODO 詳細參數說明）

⸻

💻 系統需求

項目 最低需求
OS Windows macOS Linux
Python 3.8+
記憶體 ≥ 3 GB
GPU 非必需 但可加速


⸻

🔧 開發與建置

專案結構

YOLO-Face-Match
├── face_db.py          # 建立向量資料庫
├── recognizer.py       # 影片辨識主程式
├── faces_db/           # 已知人臉照片
│   ├── alice/
│   └── bob/
├── face_db.pkl         # 向量資料庫輸出
├── tests/              # 單元測試（TODO）
└── requirements.txt

主要技術

類別 技術
偵測 YOLOv8-Face (Ultralytics)
特徵 InsightFace ArcFace
比對 numpy · cosine similarity
影像 OpenCV 4.X
視覺 Matplotlib
語言 Python 3.8+


⸻

📈 品質指標
 • PEP 8 相容
 • 型別註解完整 (mypy clean) TODO
 • pytest 覆蓋率 > 80 % TODO
 • GitHub Actions 自動測試與包裝 TODO

⸻

🤝 貢獻指南

歡迎 Issue 與 PR
請先於本地跑 face_db.py recognizer.py 確認無誤
提交前執行 pre-commit run --all-files（PEP8 與型別檢查）

回報問題
 1. 操作步驟
 2. 預期結果與實際結果
 3. 完整錯誤訊息 截圖或日誌

⸻

📄 授權

此專案採用 MIT License
YOLOv8-Face 與 InsightFace 另有各自授權 請依原專案規範使用

⸻

🙏 致謝
 • ultralytics/ultralytics – YOLOv8
 • deepinsight/InsightFace
 • ageitgey/face_recognition
 • Hugging Face Hub

⸻

🔗 相關連結
 • YOLOv8-Face 權重
https://huggingface.co/arnabdhar/YOLOv8-Face-Detection
 • InsightFace 官方
https://github.com/deepinsight/insightface
 • ArcFace 論文
https://arxiv.org/abs/1801.07698

⸻

⚠️ 注意事項
 • 第一次執行會自動下載 YOLO 權重 需網路
 • 建議每位人物至少三張正臉照片提升辨識度
 • 若缺少 face_db.pkl 系統將全部標記為 Unknown
 • 請避免超高解析度影片 以免記憶體不足
