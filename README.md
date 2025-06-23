# 🏃‍♂️ Player Tracking with YOLOv11 and Deep SORT

This project implements **player and referee tracking** in sports videos using a custom-trained **YOLOv11** model combined with **Deep SORT** for real-time object re-identification.

---

## 🔧 Features

- ⚽ Detects and tracks players (`class 2`) and referees (`class 3`)
- 🟦 Maintains **consistent IDs** across frames using Deep SORT
- 🟥 Assigns **unique colors** and ID labels to each tracked object
- 📺 Displays a live tracking window (no video saved)
- 🧠 Built using `Ultralytics YOLO`, `Deep SORT`, and `OpenCV`

---

## 🧩 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
