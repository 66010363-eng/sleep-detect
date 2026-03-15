import cv2
import time
import json
import uuid
import paho.mqtt.client as mqtt

# ===================== MQTT CONFIG =====================
MQTT_ENABLE = True
MQTT_HOST = "broker.hivemq.com"
MQTT_PORT = 1883  # TCP (non-TLS)  :contentReference[oaicite:2]{index=2}

# แนะนำให้ตั้ง topic ให้ "เฉพาะของคุณ" กันชนกับคนอื่นบน public broker
BASE_TOPIC = "kllc/drowsy/demo001"   # <- เปลี่ยนได้ตามต้องการ
TOPIC_ALERT = f"{BASE_TOPIC}/alert"  # payload: "1"=ง่วง/หลับใน, "0"=ปกติ
TOPIC_STATE = f"{BASE_TOPIC}/state"  # payload: JSON status
# =======================================================

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


def pick_two_eyes(eyes):
    if eyes is None or len(eyes) == 0:
        return []
    eyes = [tuple(map(int, e)) for e in eyes]

    # เรียงตามขนาดจากใหญ่ไปเล็ก (โฟกัสที่ดวงตาคู่ที่ใหญ่ที่สุดก่อน)
    candidates = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)

    picked = []
    for e in candidates:
        if all(iou(e, p) < 0.30 for p in picked):
            picked.append(e)
        if len(picked) == 2:
            break

    # เรียงตำแหน่งซ้ายไปขวา
    picked = sorted(picked, key=lambda e: e[0])
    return picked


def draw_help_right_center(
    img, lines,
    color=(255, 0, 50),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    thickness=2,
    margin=10,
    line_gap=6,
    bg_color=(0, 0, 0),
    bg_pad=8,
    bg_alpha=0.35
):
    if not lines:
        return
    sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t in lines]
    widths = [s[0][0] for s in sizes]
    heights = [s[0][1] for s in sizes]
    baselines = [s[1] for s in sizes]
    max_w = max(widths)
    total_h = sum(heights) + (len(lines) - 1) * line_gap

    h, w = img.shape[:2]
    x = max(margin, w - max_w - margin)
    y_start = int(h * 0.5 - total_h * 0.5)
    y0 = max(margin + heights[0], y_start + heights[0])

    top = y0 - heights[0] - bg_pad
    bottom = y0 + bg_pad
    y = y0
    for i in range(len(lines)):
        bottom = y + baselines[i] + bg_pad
        if i < len(lines) - 1:
            y += heights[i] + line_gap

    left = x - bg_pad
    right = x + max_w + bg_pad
    top = max(0, top); left = max(0, left)
    bottom = min(h - 1, bottom); right = min(w - 1, right)

    if bg_alpha <= 0:
        cv2.rectangle(img, (left, top), (right, bottom), bg_color, -1)
    else:
        overlay = img.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), bg_color, -1)
        cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, dst=img)

    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 255), 1)

    y = y0
    for i, text in enumerate(lines):
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
        if i < len(lines) - 1:
            y += heights[i] + line_gap


def draw_label_on_roi(img, text, x1, y1, color=(0, 255, 255),
                      font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, thickness=2, pad=6):
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y_text = y1 - 6
    if y_text - th - pad < 0:
        y_text = y1 + th + pad
    x_text = x1 + pad

    x_bg1 = x_text - pad
    y_bg1 = y_text - th - pad
    x_bg2 = x_text + tw + pad
    y_bg2 = y_text + pad

    h, w = img.shape[:2]
    x_bg1 = max(0, x_bg1); y_bg1 = max(0, y_bg1)
    x_bg2 = min(w - 1, x_bg2); y_bg2 = min(h - 1, y_bg2)

    cv2.rectangle(img, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)
    cv2.putText(img, text, (x_text, y_text), font, font_scale, color, thickness)


# ===================== MQTT HELPERS =====================
def mqtt_setup():
    if not MQTT_ENABLE:
        return None, {"connected": False}

    state = {"connected": False}

    client_id = f"drowsy-cam-{uuid.uuid4().hex[:8]}"
    c = mqtt.Client(client_id=client_id, clean_session=True)

    def on_connect(client, userdata, flags, rc):
        state["connected"] = (rc == 0)

    def on_disconnect(client, userdata, rc):
        state["connected"] = False

    c.on_connect = on_connect
    c.on_disconnect = on_disconnect

    # public broker: no username/password
    c.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    c.loop_start()
    return c, state


def mqtt_publish(client, state, topic, payload, retain=False):
    if (not MQTT_ENABLE) or (client is None) or (not state.get("connected", False)):
        return
    try:
        client.publish(topic, payload, qos=0, retain=retain)
    except Exception:
        pass
# ========================================================


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("ไม่พบกล้อง USB หรือเปิดกล้องไม่ได้ (ลองเปลี่ยน index เป็น 0,1,2)")

    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise RuntimeError("โหลด Haar Cascade Face ไม่ได้: " + face_cascade_path)

    eye_default_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    eye_glasses_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    eye_default = cv2.CascadeClassifier(eye_default_path)
    eye_glasses = cv2.CascadeClassifier(eye_glasses_path)
    if eye_default.empty():
        raise RuntimeError("โหลด Haar Cascade Eye(default) ไม่ได้: " + eye_default_path)
    if eye_glasses.empty():
        raise RuntimeError("โหลด Haar Cascade Eye(glasses) ไม่ได้: " + eye_glasses_path)

    MODES = {
        1: {"name": "SMALL EYES", "cascade": eye_default, "scaleFactor": 1.08, "minNeighbors": 6,
            "minSize": (14, 14), "closed_thresh": 14, "min_eyes_required": 1},
        2: {"name": "NORMAL", "cascade": eye_default, "scaleFactor": 1.10, "minNeighbors": 8,
            "minSize": (20, 20), "closed_thresh": 10, "min_eyes_required": 1},
        3: {"name": "GLASSES", "cascade": eye_glasses, "scaleFactor": 1.10, "minNeighbors": 8,
            "minSize": (18, 18), "closed_thresh": 12, "min_eyes_required": 1},
    }

    # ===== MQTT init =====
    mqtt_client, mqtt_state = mqtt_setup()
    last_state_pub = 0.0
    pub_interval_sec = 1.0
    drowsy_prev = None  # ใช้ตรวจ “เปลี่ยนสถานะ” แล้วค่อย publish
    # ====================

    mode_id = 2
    closed_frames = 0
    last_beep_time = 0.0
    BEEP_COOLDOWN_SEC = 1.0

    eye_min_w, eye_min_h = MODES[mode_id]["minSize"]
    STEP = 2
    MIN_LIMIT = 6
    MAX_LIMIT = 200

    upper_ratio = 0.48
    U_STEP = 0.02
    U_MIN = 0.40
    U_MAX = 0.90
    U_DEFAULT = 0.48

    HILIGHT_ALPHA = 0.25

    roi_color = (255, 255, 255)
    roi_thickness = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("อ่านภาพจากกล้องไม่ได้")
            break

        h, w = frame.shape[:2]

        roi_w = 700
        roi_h = 700
        x1 = (w - roi_w) // 2 - 400
        y1 = (h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        roi_gray = gray_eq[y1:y2, x1:x2]

        faces = face_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)
        )

        out = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(out, (x1, y1), (x2, y2), roi_color, roi_thickness)

        cfg = MODES[mode_id]
        mode_text = f"MODE: {mode_id} - {cfg['name']}"
        draw_label_on_roi(out, mode_text, x1, y1, color=(0, 255, 255), font_scale=0.6, thickness=2)

        eye_cascade = cfg["cascade"]
        eye_scale = cfg["scaleFactor"]
        eye_neighbors = cfg["minNeighbors"]
        CLOSED_FRAMES_THRESHOLD = cfg["closed_thresh"]
        MIN_EYES_REQUIRED = cfg["min_eyes_required"]
        eye_minsize = (eye_min_w, eye_min_h)

        eyes_count_for_warning = 0
        face_found = False

        if len(faces) > 0:
            face_found = True
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            abs_fx = x1 + fx
            abs_fy = y1 + fy

            cv2.rectangle(out, (abs_fx, abs_fy), (abs_fx + fw, abs_fy + fh), (0, 255, 0), 2)

            y_limit = int(fh * upper_ratio)
            abs_y_limit = abs_fy + y_limit

            overlay = out.copy()
            cv2.rectangle(overlay, (abs_fx, abs_fy), (abs_fx + fw, abs_y_limit), (0, 255, 255), -1)
            out = cv2.addWeighted(overlay, HILIGHT_ALPHA, out, 1 - HILIGHT_ALPHA, 0)
            cv2.line(out, (abs_fx, abs_y_limit), (abs_fx + fw, abs_y_limit), (0, 255, 255), 2)

            # ==========================================
            # [ส่วนที่แก้ไข]
            # ตัดภาพเฉพาะพื้นที่ "ด้านบน" (ความสูงจาก 0 ถึง y_limit)
            upper_face_roi_gray = roi_gray[fy:fy + y_limit, fx:fx + fw]

            # โยนภาพที่มีแค่ครึ่งบนให้ Cascade ตรวจหาตา
            eyes_raw = eye_cascade.detectMultiScale(
                upper_face_roi_gray,
                scaleFactor=eye_scale,
                minNeighbors=eye_neighbors,
                minSize=eye_minsize
            )

            # เรียกฟังก์ชันที่ตัดพารามิเตอร์เก่าออกแล้ว
            eyes_picked = pick_two_eyes(eyes_raw)
            # ==========================================

            eyes_count_for_warning = len(eyes_picked)

            # วาดกรอบดวงตา (พิกัด ex, ey ที่คืนค่ามายังสอดคล้องกับ abs_fx, abs_fy อยู่แล้ว)
            for (ex, ey, ew, eh) in eyes_picked:
                cv2.rectangle(out,
                              (abs_fx + ex, abs_fy + ey),
                              (abs_fx + ex + ew, abs_fy + ey + eh),
                              (255, 0, 0), 2)

            if eyes_count_for_warning < MIN_EYES_REQUIRED:
                closed_frames += 1
            else:
                closed_frames = 0

        # ===== สถานะง่วง/หลับใน =====
        drowsy = (closed_frames >= CLOSED_FRAMES_THRESHOLD)

        # ===== MQTT publish (เมื่อสถานะเปลี่ยน) =====
        if drowsy_prev is None or drowsy != drowsy_prev:
            mqtt_publish(mqtt_client, mqtt_state, TOPIC_ALERT, "1" if drowsy else "0", retain=False)
            drowsy_prev = drowsy

        # ===== MQTT publish state เป็นระยะ =====
        now_t = time.time()
        if now_t - last_state_pub >= pub_interval_sec:
            payload = {
                "ts": int(now_t),
                "drowsy": bool(drowsy),
                "eyes": int(eyes_count_for_warning),
                "closed_frames": int(closed_frames),
                "threshold": int(CLOSED_FRAMES_THRESHOLD),
                "mode": int(mode_id),
                "mode_name": cfg["name"],
                "upper_ratio": float(upper_ratio),
                "eye_minSize": [int(eye_min_w), int(eye_min_h)],
                "face_found": bool(face_found),
            }
            mqtt_publish(mqtt_client, mqtt_state, TOPIC_STATE, json.dumps(payload), retain=False)
            last_state_pub = now_t
        # =====================================

        if drowsy:
            cv2.putText(out, "WARNING: EYES CLOSED!", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if now_t - last_beep_time >= BEEP_COOLDOWN_SEC:
                try:
                    import winsound
                    winsound.Beep(1200, 200)
                except Exception:
                    pass
                last_beep_time = now_t

        help_lines = [
            f"eye_minSize: {eye_min_w} x {eye_min_h}",
            f"upper_ratio: {upper_ratio:.2f}",
            "1,2,3 mode",
            "-,+ minSize",
            "r reset minSize",
            "/,* upper_ratio",
            "u reset upper_ratio",
        ]
        draw_help_right_center(out, help_lines, color=(0, 255, 255), font_scale=0.6, thickness=2)

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
            eye_min_w, eye_min_h = MODES[mode_id]["minSize"]
        elif key == ord('2'):
            mode_id = 2
            closed_frames = 0
            eye_min_w, eye_min_h = MODES[mode_id]["minSize"]
        elif key == ord('3'):
            mode_id = 3
            closed_frames = 0
            eye_min_w, eye_min_h = MODES[mode_id]["minSize"]

        elif key == ord('-'):
            eye_min_w = max(MIN_LIMIT, eye_min_w - STEP)
            eye_min_h = max(MIN_LIMIT, eye_min_h - STEP)
        elif key == ord('+') or key == ord('='):
            eye_min_w = min(MAX_LIMIT, eye_min_w + STEP)
            eye_min_h = min(MAX_LIMIT, eye_min_h + STEP)
        elif key == ord('r'):
            eye_min_w, eye_min_h = MODES[mode_id]["minSize"]

        elif key == ord('/'):
            upper_ratio = max(U_MIN, upper_ratio - U_STEP)
        elif key == ord('*'):
            upper_ratio = min(U_MAX, upper_ratio + U_STEP)
        elif key == ord('u'):
            upper_ratio = U_DEFAULT

    cap.release()
    cv2.destroyAllWindows()

    # ปิด MQTT สวย ๆ
    if MQTT_ENABLE and mqtt_client is not None:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
