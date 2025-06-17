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
sketch_enabled = False
sketch_mode_displayed = False
sketch_mode_activated_time = None
show_sketch_only = False

color_ranges = [
    ((0, 70, 50), (10, 255, 255)),     # 빨강
    ((35, 70, 50), (85, 255, 255)),    # 초록
    ((90, 70, 50), (130, 255, 255))    # 파랑
]

print("[INFO] 검지만 펴면 그리기 시작")
print("[INFO] 손가락 다 접으면 그리기 중지")
print("[INFO] 검지와 중지 두 손가락 펴면 해당 위치를 지움")
print("[INFO] 손을 모두 펴면 스케치 모드 활성화")
print("[INFO] 검지+새끼손가락만 펴면 프로그램 종료")

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

                # 종료 조건: 검지(8)와 새끼손가락(20)만 펴졌을 때
                if fingers_up == [False, True, False, False, True]:
                    print("[INFO] 종료 제스처 감지됨: 프로그램 종료")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                if all(fingers_up) and not sketch_mode_displayed:
                    sketch_enabled = True
                    sketch_mode_displayed = True
                    sketch_mode_activated_time = cv2.getTickCount()

                only_index_up = fingers_up[1] and not any(fingers_up[2:])  # index만
                index_middle_up = fingers_up[1] and fingers_up[2] and not (fingers_up[0] or fingers_up[3] or fingers_up[4])
                all_fingers_down = not any(fingers_up)

                x = int(landmarks[8].x * w)
                y = int(landmarks[8].y * h)

                if sketch_enabled and only_index_up:
                    sketch_mode = True
                    cv2.putText(frame, "Drawing...", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)

                    if prev_x is not None:
                        cv2.line(canvas_current, (prev_x, prev_y), (x, y), current_color, 5)

                    prev_x, prev_y = x, y
                elif sketch_enabled and all_fingers_down:
                    sketch_mode = False
                    prev_x, prev_y = None, None
                else:
                    prev_x, prev_y = None, None

                if index_middle_up:
                    erase_mode = True
                    cx = int(landmarks[8].x * w)
                    cy = int(landmarks[8].y * h)
                    cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 2)
                    cv2.circle(canvas_current, (cx, cy), 30, (255, 255, 255), -1)

        else:
            prev_x, prev_y = None, None

        if sketch_enabled:
            roi = frame[0:100, 0:100]
            roi_hsv = hsv[0:100, 0:100]
            cv2.rectangle(frame, (0, 0), (100, 100), (200, 200, 200), 2)

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
