"""
Minimal CV Pipeline Test Runner
VIDEO → YOLO DETECTION → OBJECT TRACKING → ANNOTATED OUTPUT
"""

import cv2
import time
import sys
from ultralytics import YOLO


def run_pipeline(video_path: str, output_path: str = "output_annotated.mp4"):
    """Run detection + tracking on video, save annotated output."""

    print(f"Loading model: yolov8n.pt")
    model = YOLO("yolov8n.pt")

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Video: {width}x{height}, {total_frames} frames, {fps:.1f} FPS")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    total_detections = 0
    unique_tracks = set()
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        results = model.track(frame, persist=True, verbose=False)
        r = results[0]

        annotated = r.plot()

        if r.boxes is not None and len(r.boxes) > 0:
            total_detections += len(r.boxes)
            if r.boxes.id is not None:
                for tid in r.boxes.id.cpu().numpy():
                    unique_tracks.add(int(tid))

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = r.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                cv2.putText(
                    annotated,
                    f"{name} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                if r.boxes.id is not None:
                    tid = int(box.id[0])
                    cv2.putText(
                        annotated,
                        f"ID:{tid}",
                        (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

        writer.write(annotated)

        if frame_num % 30 == 0 or frame_num == 1:
            elapsed = time.time() - start_time
            current_fps = frame_num / elapsed if elapsed > 0 else 0
            print(
                f"  Frame {frame_num}/{total_frames} | "
                f"FPS: {current_fps:.1f} | "
                f"Detections so far: {total_detections}"
            )

    elapsed = time.time() - start_time
    avg_fps = frame_num / elapsed if elapsed > 0 else 0

    cap.release()
    writer.release()

    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*50}")
    print(f"Frames processed:  {frame_num}")
    print(f"Elapsed time:      {elapsed:.2f}s")
    print(f"Average FPS:       {avg_fps:.1f}")
    print(f"Total detections:  {total_detections}")
    print(f"Unique track IDs:  {len(unique_tracks)}")
    print(f"Output saved to:   {output_path}")
    print(f"{'='*50}")

    return {
        "frames": frame_num,
        "elapsed": elapsed,
        "fps": avg_fps,
        "detections": total_detections,
        "tracks": len(unique_tracks),
    }


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "test_synthetic.mp4"
    output = sys.argv[2] if len(sys.argv) > 2 else "output_annotated.mp4"
    run_pipeline(video, output)
