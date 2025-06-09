import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
draw_utils = mp.solutions.drawing_utils

canvas_current = np.ones((600, 800, 3), np.uint8) * 255
current_color = (0, 0, 0)
prev_x, prev_y = None, None
last_pointer = None
sketch_enabled = False
sketch_mode_displayed = False
sketch_mode_activated_time = None

print("[INFO] 손을 모두 펴면 스케치 모드 활성화")
print("[INFO] 검지만 펴면 그림을 그릴 수 있음")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

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
            x = int(landmarks[8].x * w)
            y = int(landmarks[8].y * h)

            if sketch_enabled and only_index_up:
                cv2.putText(frame, "Drawing...", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)
                if prev_x is not None:
                    cv2.line(canvas_current, (prev_x, prev_y), (x, y), current_color, 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    if sketch_mode_displayed and sketch_mode_activated_time:
        time_passed = (cv2.getTickCount() - sketch_mode_activated_time) / cv2.getTickFrequency()
        if time_passed < 2.0:
            cv2.putText(frame, "Sketch Mode ON", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

    cv2.imshow("Camera", frame)
    if sketch_enabled:
        cv2.imshow("Sketch", canvas_current)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
