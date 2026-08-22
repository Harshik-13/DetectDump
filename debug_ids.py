"""Check what track IDs are assigned to bags."""
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video = r"test_videos\WhatsApp Video 2026-08-22 at 2.57.14 PM.mp4"
cap = cv2.VideoCapture(video)

bag_ids_per_frame = {}
for target in range(0, 240, 6):
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if not ret:
        continue

    results = model.track(frame, persist=True, conf=0.1, verbose=False)
    r = results[0]
    bags = []
    if r.boxes is not None and r.boxes.id is not None:
        for i in range(len(r.boxes)):
            cls = r.names[int(r.boxes.cls[i])]
            if "bag" in cls or cls == "backpack":
                tid = int(r.boxes.id[i])
                conf = float(r.boxes.conf[i])
                bags.append(f"ID:{tid}({conf:.2f})")
    bag_ids_per_frame[target] = bags
    sec = target // 24
    print(f"s{sec}: {bags if bags else 'none'}")

cap.release()

# Analyze ID consistency
print("\nID Analysis:")
all_ids = set()
for frame, bags in bag_ids_per_frame.items():
    for b in bags:
        tid = b.split("(")[0]
        all_ids.add(tid)
print(f"Total unique IDs seen: {len(all_ids)}")
print(f"IDs: {sorted(all_ids)}")
