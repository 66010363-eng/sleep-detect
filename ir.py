import cv2
import time

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def pick_two_eyes(eyes, face_h):
    """
    เลือกกรอบดวงตาไม่เกิน 2 กล่อง:
    - เน้นที่อยู่ครึ่งบนของหน้า
    - เลือกกล่องใหญ่/ชัดก่อน
    - กันกล่องซ้อนกันด้วย IoU
    """
    if eyes is None or len(eyes) == 0:
        return []

    # แปลงเป็น list ของ tuple
    eyes = [tuple(map(int, e)) for e in eyes]

    # 1) กรองให้อยู่ครึ่งบน/ช่วงบนของหน้า (ตาอยู่ประมาณนี้)
    upper = [e for e in eyes if e[1] < int(face_h * 0.65)]
    candidates = upper if len(upper) >= 1 else eyes

    # 2) เรียงจาก "พื้นที่ใหญ่" ก่อน
    candidates = sorted(candidates, key=lambda e: e[2] * e[3], reverse=True)

    # 3) เลือกแบบไม่ให้ซ้อนกันมาก
    picked = []
    for e in candidates:
        if all(iou(e, p) < 0.30 for p in picked):
            picked.append(e)
        if len(picked) == 2:
            break

    # 4) ถ้าได้ 2 กล่องแล้ว จัดเรียงซ้าย->ขวา ให้ดูเป็นธรรมชาติ
    picked = sorted(picked, key=lambda e: e[0])

    return picked


def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("ไม่พบกล้อง USB หรือเปิดกล้องไม่ได้ (ลองเปลี่ยน index เป็น 0,1,2)")

    # Face detector
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise RuntimeError("โหลด Haar Cascade Face ไม่ได้: " + face_cascade_path)

    # Eye detectors
    eye_default_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    eye_glasses_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    eye_default = cv2.CascadeClassifier(eye_default_path)
    eye_glasses = cv2.CascadeClassifier(eye_glasses_path)
    if eye_default.empty():
        raise RuntimeError("โหลด Haar Cascade Eye(default) ไม่ได้: " + eye_default_path)
    if eye_glasses.empty():
        raise RuntimeError("โหลด Haar Cascade Eye(glasses) ไม่ได้: " + eye_glasses_path)

    # ===================== โหมดการตรวจตา =====================
    MODES = {
        1: {  # ตาตี่
            "name": "SMALL EYES",
            "cascade": eye_default,
            "scaleFactor": 1.08,
            "minNeighbors": 6,
            "minSize": (14, 14),
            "closed_thresh": 14,
            "min_eyes_required": 1
        },
        2: {  # ทั่วไป
            "name": "NORMAL",
            "cascade": eye_default,
            "scaleFactor": 1.10,
            "minNeighbors": 8,
            "minSize": (20, 20),
            "closed_thresh": 10,
            "min_eyes_required": 1
        },
        3: {  # ใส่แว่น
            "name": "GLASSES",
            "cascade": eye_glasses,
            "scaleFactor": 1.10,
            "minNeighbors": 8,
            "minSize": (18, 18),
            "closed_thresh": 12,
            "min_eyes_required": 1
        },
    }

    mode_id = 2
    closed_frames = 0
    last_beep_time = 0.0
    BEEP_COOLDOWN_SEC = 1.0

    roi_color = (255, 255, 255)
    roi_thickness = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("อ่านภาพจากกล้องไม่ได้")
            break

        h, w = frame.shape[:2]

        # ====== ROI กำหนดเอง ======
        roi_w = 300
        roi_h = 300
        x1 = (w - roi_w) // 2 - 170
        y1 = (h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        roi_gray = gray_eq[y1:y2, x1:x2]

        faces = face_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        out = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(out, (x1, y1), (x2, y2), roi_color, roi_thickness)

        cfg = MODES[mode_id]
        eye_cascade = cfg["cascade"]
        eye_scale = cfg["scaleFactor"]
        eye_neighbors = cfg["minNeighbors"]
        eye_minsize = cfg["minSize"]
        CLOSED_FRAMES_THRESHOLD = cfg["closed_thresh"]
        MIN_EYES_REQUIRED = cfg["min_eyes_required"]

        eyes_count_for_warning = 0

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            abs_fx = x1 + fx
            abs_fy = y1 + fy

            cv2.rectangle(out, (abs_fx, abs_fy), (abs_fx + fw, abs_fy + fh), (0, 255, 0), 2)

            face_roi_gray = roi_gray[fy:fy+fh, fx:fx+fw]

            eyes_raw = eye_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=eye_scale,
                minNeighbors=eye_neighbors,
                minSize=eye_minsize
            )

            # ✅ เลือกให้เหลือแค่ 2 กล่อง
            eyes_picked = pick_two_eyes(eyes_raw, face_h=fh)
            eyes_count_for_warning = len(eyes_picked)

            # วาดเฉพาะ 2 กล่องที่เลือก
            for (ex, ey, ew, eh) in eyes_picked:
                cv2.rectangle(
                    out,
                    (abs_fx + ex, abs_fy + ey),
                    (abs_fx + ex + ew, abs_fy + ey + eh),
                    (255, 0, 0), 2
                )

            if eyes_count_for_warning < MIN_EYES_REQUIRED:
                closed_frames += 1
            else:
                closed_frames = 0
        else:
            closed_frames = 0

        if closed_frames >= CLOSED_FRAMES_THRESHOLD:
            cv2.putText(out, "WARNING: EYES CLOSED!", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            now = time.time()
            if now - last_beep_time >= BEEP_COOLDOWN_SEC:
                try:
                    import winsound
                    winsound.Beep(1200, 200)
                except Exception:
                    pass
                last_beep_time = now

        cv2.putText(out, f"MODE: {mode_id} - {cfg['name']} (1=Small 2=Normal 3=Glasses)",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 50), 2)

        # DEBUG ขวาล่าง: แค่ eyes และ closed_frames
        debug_text = f"eyes:{eyes_count_for_warning}  closed_frames:{closed_frames}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(debug_text, font, font_scale, thickness)
        margin = 10
        x_text = max(margin, w - tw - margin)
        y_text = max(th + margin, h - margin)
        cv2.putText(out, debug_text, (x_text, y_text),
                    font, font_scale, (255, 0, 50), thickness)

        cv2.imshow("USB Camera - ROI + Face + Eye Detect", out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('1'):
            mode_id = 1
            closed_frames = 0
        elif key == ord('2'):
            mode_id = 2
            closed_frames = 0
        elif key == ord('3'):
            mode_id = 3
            closed_frames = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
