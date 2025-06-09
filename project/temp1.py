import cv2
import mediapipe as mp
import numpy as np

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
sketch_mode_activated_time = None  # 추가됨
show_sketch_only = False

color_ranges = [
    ((0, 70, 50), (10, 255, 255), (0, 0, 255)),     # 빨강
    ((35, 70, 50), (85, 255, 255), (0, 255, 0)),    # 초록
    ((90, 70, 50), (130, 255, 255), (255, 0, 0))    # 파랑
]

print("[INFO] 검지만 펴면 스케치 모드")
print("[INFO] 손바닥을 카메라 왼쪽 위에 대면 색상 변경")
print("[INFO] 엄지를 위로 들면 해당 위치를 지움")
print("[INFO] 손을 모두 펴면 스케치 모드 활성화")

while True:
    if not show_sketch_only:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = hands.process(image_rgb)

        sketch_mode = False
        erase_mode = False

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

                # 손가락 모두 펴졌을 때 스케치 모드 활성화 + 시작 시간 기록
                if all(fingers_up) and not sketch_mode_displayed:
                    sketch_enabled = True
                    sketch_mode_displayed = True
                    sketch_mode_activated_time = cv2.getTickCount()  # 시간 기록

                only_index_up = fingers_up[1] and not any(fingers_up[2:])
                thumb_up = fingers_up[0] and not any(fingers_up[1:])

                x = int(landmarks[8].x * w)
                y = int(landmarks[8].y * h)

                if sketch_enabled and only_index_up:
                    sketch_mode = True
                    show_pointer = False

                    cv2.putText(frame, "Drawing...", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

                    if prev_x is not None:
                        cv2.line(canvas_current, (prev_x, prev_y), (x, y), current_color, 5)

                    prev_x, prev_y = x, y
                    last_pointer = (x, y)
                else:
                    sketch_mode = False
                    if prev_x is not None:
                        show_pointer = True
                    prev_x, prev_y = None, None

                if thumb_up:
                    erase_mode = True
                    cx = int(landmarks[4].x * w)
                    cy = int(landmarks[4].y * h)
                    cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 2)
                    cv2.circle(canvas_current, (cx, cy), 30, (255, 255, 255), -1)

        else:
            prev_x, prev_y = None, None

        # 색상 변경 영역
        if sketch_enabled:
            roi = hsv[0:100, 0:100]
            cv2.rectangle(frame, (0, 0), (100, 100), (200, 200, 200), 2)
            for lower, upper, new_color in color_ranges:
                mask = cv2.inRange(roi, lower, upper)
                if cv2.countNonZero(mask) > 500:
                    if current_color != new_color:
                        print(f"[INFO] 색상 변경됨 → {new_color}")
                        current_color = new_color
                    break
            cv2.rectangle(frame, (0, 100), (50, 150), current_color, -1)

        # RGB 표시
        r, g, b = current_color[2], current_color[1], current_color[0]
        rgb_text = f"RGB: ({r}, {g}, {b})"
        cv2.putText(frame, rgb_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 포인터 표시
        if show_pointer and last_pointer:
            tip_x, tip_y = last_pointer
            cv2.circle(frame, (tip_x, tip_y), 5, current_color, -1)
            cv2.line(frame, (tip_x, tip_y), (tip_x, tip_y + 30), (50, 50, 50), 3)

        # Sketch Mode ON 텍스트 2초 동안 표시
        if sketch_mode_displayed and sketch_mode_activated_time:
            time_passed = (cv2.getTickCount() - sketch_mode_activated_time) / cv2.getTickFrequency()
            if time_passed < 2.0:
                cv2.putText(frame, "Sketch Mode ON", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

        cv2.imshow("Camera", frame)
        if sketch_enabled:
            cv2.imshow("Sketch", canvas_current)

    else:
        cv2.imshow("Sketch Only", canvas_current)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        if not show_sketch_only:
            cv2.destroyWindow("Camera")
            show_sketch_only = True
        else:
            break

cap.release()
cv2.destroyAllWindows()
