import cv2
import mediapipe as mp
import numpy as np
import time
import math
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class HandDetector:
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.2,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.start_time = time.time()

    def detect(self, frame):
        """Returns list of (center, radius, handedness) for each open hand."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)

        open_hands = []
        if not results.hand_landmarks:
            return open_hands

        h, w = frame.shape[:2]

        for landmarks, handedness in zip(results.hand_landmarks, results.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            if self._is_open(landmarks, label):
                center = self._palm_center(landmarks, w, h)
                radius = self._circle_radius(landmarks, w, h)
                open_hands.append((center, radius, label))

        return open_hands

    def _is_open(self, lm, label):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        # Count extended fingers instead of requiring ALL 4
        extended = 0
        for tip_id, pip_id in zip(tips, pips):
            if lm[tip_id].y < lm[pip_id].y + 0.03:  # small tolerance
                extended += 1

        # Require at least 3 out of 4 fingers extended (not all 4)
        if extended < 3:
            return False

        return True

    def _palm_center(self, lm, w, h):
        ids = [0, 5, 9, 13, 17]
        cx = int(sum(lm[i].x for i in ids) / len(ids) * w)
        cy = int(sum(lm[i].y for i in ids) / len(ids) * h)
        return (cx, cy)

    def _circle_radius(self, lm, w, h):
        wrist = np.array([lm[0].x * w, lm[0].y * h])
        mcp = np.array([lm[9].x * w, lm[9].y * h])
        return int(np.linalg.norm(wrist - mcp) * 1.4)


class MagicCircleRenderer:
    # BGR colors
    PRIMARY = (255, 0, 200)      # bright magenta
    SECONDARY = (180, 0, 255)    # deep purple
    ACCENT = (255, 100, 255)     # light pink
    GLOW = (200, 50, 255)        # glow purple

    def __init__(self):
        self.outer_angle = 0.0
        self.inner_angle = 0.0
        self.star_angle = 0.0
        self.prev_time = time.time()

    def _update_angles(self):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        self.outer_angle = (self.outer_angle + 60 * dt) % 360
        self.inner_angle = (self.inner_angle - 90 * dt) % 360
        self.star_angle = (self.star_angle + 30 * dt) % 360

    def draw(self, frame, center, radius):
        self._update_angles()
        overlay = frame.copy().astype(np.float32)

        self._draw_glow(overlay, center, radius)
        self._draw_dashed_circle(overlay, center, radius, self.PRIMARY, 3, 24, self.outer_angle)
        self._draw_dashed_circle(overlay, center, int(radius * 0.65), self.SECONDARY, 2, 16, self.inner_angle)
        self._draw_hexagram(overlay, center, int(radius * 0.45), self.star_angle)
        self._draw_dot_markers(overlay, center, radius, 8, self.outer_angle)

        frame[:] = cv2.addWeighted(overlay, 1.0, frame.astype(np.float32), 0.0, 0).astype(np.uint8)

    def _draw_glow(self, img, center, radius):
        for i in range(1, 5):
            r = radius + i * 4
            alpha = 0.15
            glow_layer = np.zeros_like(img, dtype=np.float32)
            cv2.circle(glow_layer, center, r, self.GLOW, 2)
            img[:] = cv2.addWeighted(img, 1.0, glow_layer, alpha, 0)

    def _draw_dashed_circle(self, img, center, radius, color, thickness, n_segments, angle_offset):
        """Draw N evenly-spaced arc segments around a circle."""
        gap_ratio = 0.4
        seg_angle = 360.0 / n_segments
        arc_span = seg_angle * (1.0 - gap_ratio)

        for i in range(n_segments):
            start_deg = angle_offset + i * seg_angle
            end_deg = start_deg + arc_span
            start_rad = math.radians(start_deg)
            end_rad = math.radians(end_deg)

            pts = []
            steps = max(4, int(arc_span))
            for s in range(steps + 1):
                a = start_rad + (end_rad - start_rad) * s / steps
                x = int(center[0] + radius * math.cos(a))
                y = int(center[1] + radius * math.sin(a))
                pts.append((x, y))

            for j in range(len(pts) - 1):
                cv2.line(img, pts[j], pts[j + 1], color, thickness, cv2.LINE_AA)

    def _draw_hexagram(self, img, center, radius, angle_offset):
        """Draw a 6-pointed star (two overlapping triangles)."""
        self._draw_polygon(img, center, radius, 3, angle_offset, self.ACCENT, 2)
        self._draw_polygon(img, center, radius, 3, angle_offset + 60, self.SECONDARY, 2)

    def _draw_polygon(self, img, center, radius, sides, angle_offset_deg, color, thickness):
        pts = []
        for i in range(sides):
            a = math.radians(angle_offset_deg + i * 360.0 / sides)
            x = int(center[0] + radius * math.cos(a))
            y = int(center[1] + radius * math.sin(a))
            pts.append((x, y))
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def _draw_dot_markers(self, img, center, radius, n_dots, angle_offset):
        for i in range(n_dots):
            a = math.radians(angle_offset + i * 360.0 / n_dots)
            x = int(center[0] + radius * math.cos(a))
            y = int(center[1] + radius * math.sin(a))
            cv2.circle(img, (x, y), 4, self.ACCENT, -1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    detector = HandDetector()
    renderer = MagicCircleRenderer()
    prev_time = time.time()

    print("Magic Circle running — hold open palm toward camera.")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        hands = detector.detect(frame)

        for center, radius, handedness in hands:
            renderer.draw(frame, center, radius)

        # FPS counter
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Magic Circle", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()