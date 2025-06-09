import cv2
import mediapipe as mp
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
draw_utils = mp.solutions.drawing_utils

canvas_current = np.ones((480, 640, 3), np.uint8) * 255
current_color = (0, 0, 0)
prev_x, prev_y = None, None
last_pointer = None
show_pointer = False
sketch_enabled = False
sketch_mode_displayed = False
sketch_mode_activated_time = None
show_sketch_only = False
adjust_mode = False

line_thickness = 5
erase_radius = 30

color_ranges = [
    ((0, 70, 50), (10, 255, 255)),
    ((35, 70, 50), (85, 255, 255)),
    ((90, 70, 50), (130, 255, 255))
]

print("[INFO] 검지만 펴면 그리기 시작")
print("[INFO] 손가락 다 접으면 그리기 중지 (포인터 표시)")
print("[INFO] 검지+중지 펴면 지우기")
print("[INFO] 새끼손가락만 펴면 색상 선택")
print("[INFO] 손 전체 펴면 스케치 모드 ON")
print("[INFO] 'q' → 채도 조절 모드 진입")

while True:
    if not show_sketch_only and not adjust_mode:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = hands.process(image_rgb)

        sketch_mode = False
        erase_mode = False
        fingers_up = []

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                draw_utils.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                landmarks = handLms.landmark
                h, w = frame.shape[:2]

                fingers_up = []
                for tip_id in [4, 8, 12, 16, 20]:
                    if tip_id == 4:
                        fingers_up.append(landmarks[tip_id].x < landmarks[tip_id - 1].x)
                    else:
                        fingers_up.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)

                if all(fingers_up) and not sketch_mode_displayed:
                    sketch_enabled = True
                    sketch_mode_displayed = True
                    sketch_mode_activated_time = cv2.getTickCount()

                only_index_up = fingers_up[1] and not any(fingers_up[2:])
                index_middle_up = fingers_up[1] and fingers_up[2] and not (fingers_up[0] or fingers_up[3] or fingers_up[4])
                all_fingers_down = not any(fingers_up)

                x = int(landmarks[8].x * w)
                y = int(landmarks[8].y * h)

                if sketch_enabled and only_index_up:
                    sketch_mode = True
                    show_pointer = False
                    last_pointer = None

                    cv2.putText(frame, "Drawing...", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

                    if prev_x is not None:
                        cv2.line(canvas_current, (prev_x, prev_y), (x, y), current_color, line_thickness)

                    prev_x, prev_y = x, y
                elif sketch_enabled and all_fingers_down:
                    sketch_mode = False
                    if prev_x is not None:
                        last_pointer = (prev_x, prev_y)
                        show_pointer = True
                    prev_x, prev_y = None, None
                else:
                    prev_x, prev_y = x, y

                if index_middle_up:
                    erase_mode = True
                    cx = int(landmarks[8].x * w)
                    cy = int(landmarks[8].y * h)
                    cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 2)
                    cv2.circle(canvas_current, (cx, cy), erase_radius, (255, 255, 255), -1)

        else:
            prev_x, prev_y = None, None

        if sketch_enabled:
            roi = frame[0:100, 0:100]
            roi_hsv = hsv[0:100, 0:100]
            cv2.rectangle(frame, (0, 0), (100, 100), (200, 200, 200), 2)

            if fingers_up == [False, False, False, False, True]:
                cv2.putText(frame, "Color Selecting...", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                for lower, upper in color_ranges:
                    mask = cv2.inRange(roi_hsv, lower, upper)
                    count = cv2.countNonZero(mask)
                    if count > 500:
                        mean_val = cv2.mean(roi, mask=mask)
                        new_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
                        if current_color != new_color:
                            print(f"[INFO] 색상 변경됨 → {new_color}")
                            current_color = new_color
                        break

            cv2.rectangle(frame, (0, 100), (50, 150), current_color, -1)

        r, g, b = current_color[2], current_color[1], current_color[0]
        rgb_text = f"RGB: ({r}, {g}, {b})"
        cv2.putText(frame, rgb_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if sketch_mode_displayed and sketch_mode_activated_time:
            time_passed = (cv2.getTickCount() - sketch_mode_activated_time) / cv2.getTickFrequency()
            if time_passed < 2.0:
                cv2.putText(frame, "Sketch Mode ON", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

        canvas_show = canvas_current.copy()

        if result.multi_hand_landmarks and prev_x is not None:
            drawing_dot_radius = line_thickness // 2 + 3
            cv2.circle(canvas_show, (prev_x, prev_y), drawing_dot_radius, (0, 255, 255), -1)

        if sketch_enabled and index_middle_up:
            cv2.circle(canvas_show, (x, y), erase_radius, (0, 255, 255), 2)

        if show_pointer and last_pointer:
            tip_x, tip_y = last_pointer
            size = 30
            triangle = np.array([
                [tip_x, tip_y],
                [tip_x - size // 2, tip_y + size],
                [tip_x + size // 2, tip_y + size]
            ])
            cv2.drawContours(canvas_show, [triangle], 0, current_color, -1)

            coord_text = f"X: {tip_x}, Y: {tip_y}"
            cv2.putText(canvas_show, coord_text, (canvas_show.shape[1] - 200, canvas_show.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Camera", frame)
        if sketch_enabled:
            cv2.imshow("Sketch", canvas_show)

    elif adjust_mode:
        canvas_hsv = cv2.cvtColor(adjusted_canvas, cv2.COLOR_BGR2HSV)
        sat_val = cv2.getTrackbarPos('Saturation', 'Adjust Colors')
        sat_scale = sat_val / 100.0
        canvas_hsv[:, :, 1] = np.clip(canvas_hsv[:, :, 1].astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
        adjusted_show = cv2.cvtColor(canvas_hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('Adjust Colors', adjusted_show)

    else:
        adjust_mode = True
        adjusted_canvas = canvas_current.copy()
        cv2.namedWindow('Adjust Colors')
        cv2.createTrackbar('Saturation', 'Adjust Colors', 100, 200, nothing)
        print("[INFO] 색상 조절 모드 진입: 트랙바로 채도 조절, ESC로 종료")

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        if not show_sketch_only and not adjust_mode:
            cv2.destroyWindow("Camera")
            cv2.destroyWindow("Sketch")
            show_sketch_only = True
        elif adjust_mode:
            break
    elif key == 27:
        if adjust_mode:
            break

cap.release()
cv2.destroyAllWindows()
