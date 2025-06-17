import cv2
import mediapipe as mp
import numpy as np

# --- 트랙바 관련 전역 변수 및 함수 ---
unique_colors = []
trackbars_created = False
canvas_original = None

def on_trackbar_change(val):
    """트랙바 값이 변경될 때마다 호출되어 'Filtered Image' 창만 업데이트하는 함수"""
    global canvas_original, unique_colors

    if canvas_original is None:
        return

    # 필터링을 적용할 이미지를 원본에서 매번 새로 복사
    filtered_image = canvas_original.copy()

    # 각 트랙바의 상태를 확인하여 색상 필터링
    for i, color in enumerate(unique_colors):
        trackbar_name = f'Color {i+1} {color.tolist()}'
        status = cv2.getTrackbarPos(trackbar_name, 'Color Controls')

        if status == 0:  # 트랙바가 OFF 상태이면
            mask = np.all(filtered_image == color, axis=2)
            filtered_image[mask] = [255, 255, 255]
    
    # --- 수정된 부분: 필터링된 이미지를 'Filtered Image' 창에 표시 ---
    cv2.imshow("Filtered Image", filtered_image)


def create_color_trackbars():
    """스케치에서 사용된 색상을 찾아 트랙바를 생성하는 함수 (canvas 인자 제거)"""
    global unique_colors, canvas_original

    pixels = canvas_original.reshape(-1, 3)
    unique_colors_bgr = np.unique(pixels, axis=0)

    for color in unique_colors_bgr:
        if not np.array_equal(color, [255, 255, 255]):
            unique_colors.append(color)

    if not unique_colors:
        print("[INFO] No colors drawn to create trackbars.")
        return

    cv2.namedWindow('Color Controls')
    cv2.resizeWindow('Color Controls', 400, len(unique_colors) * 50)

    for i, color in enumerate(unique_colors):
        trackbar_name = f'Color {i+1} {color.tolist()}'
        cv2.createTrackbar(trackbar_name, 'Color Controls', 1, 1, on_trackbar_change)
        print(f"[INFO] Created trackbar for color (BGR): {color}")

    # 초기 필터링 화면 업데이트를 위해 콜백 함수 호출
    on_trackbar_change(0)

# --- 기본 설정 ---
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
# ... (이하 print문들은 이전과 동일)

while True:
    if not show_sketch_only:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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

                if fingers_up == [False, True, False, False, True]:
                    print("[INFO] 종료 제스처 감지됨: 프로그램 종료")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                if all(fingers_up) and not sketch_mode_displayed:
                    sketch_enabled = True
                    sketch_mode_displayed = True
                    sketch_mode_activated_time = cv2.getTickCount()

                only_index_up = fingers_up[1] and not any(fingers_up[2:])
                index_middle_up = fingers_up[1] and fingers_up[2] and not (fingers_up[0] or fingers_up[3] or fingers_up[4])

                x = int(landmarks[8].x * w)
                y = int(landmarks[8].y * h)

                if sketch_enabled and only_index_up:
                    if prev_x is not None:
                        cv2.line(canvas_current, (prev_x, prev_y), (x, y), current_color, 5)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None

                if index_middle_up:
                    cx, cy = int(landmarks[8].x * w), int(landmarks[8].y * h)
                    cv2.circle(canvas_current, (cx, cy), 30, (255, 255, 255), -1)
        else:
            prev_x, prev_y = None, None

        if sketch_enabled:
            roi_frame = frame[0:100, 0:100]
            roi_hsv = hsv[0:100, 0:100]
            cv2.rectangle(frame, (0, 0), (100, 100), (200, 200, 200), 2)
            for lower, upper in color_ranges:
                mask = cv2.inRange(roi_hsv, lower, upper)
                if cv2.countNonZero(mask) > 500:
                    mean_val = cv2.mean(roi_frame, mask=mask)
                    new_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))
                    if current_color != new_color:
                        print(f"[INFO] 색상 변경됨 → {new_color}")
                        current_color = new_color
                    break
        
        # === 이 부분이 복원되었습니다 (1/2) ===
        # 현재 선택된 색상을 보여주는 사각형
        cv2.rectangle(frame, (0, 100), (50, 150), current_color, -1)
        
        # === 이 부분이 복원되었습니다 (2/2) ===
        # 현재 색상의 RGB 값을 텍스트로 표시
        r, g, b = current_color[2], current_color[1], current_color[0]
        rgb_text = f"RGB: ({r}, {g}, {b})"
        cv2.putText(frame, rgb_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 스케치 모드 활성화 메시지 표시
        if sketch_mode_displayed and sketch_mode_activated_time:
            time_passed = (cv2.getTickCount() - sketch_mode_activated_time) / cv2.getTickFrequency()
            if time_passed < 2.0:
                cv2.putText(frame, "Sketch Mode ON", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)
        if sketch_enabled:
            cv2.imshow("Sketch", canvas_current)

    else:  # show_sketch_only == True
        if not trackbars_created:
            cv2.imshow("Original Image", canvas_original)
            cv2.imshow("Filtered Image", canvas_original)
            create_color_trackbars()
            trackbars_created = True

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        if not show_sketch_only:
            canvas_original = canvas_current.copy()
            if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Camera")
            if cv2.getWindowProperty("Sketch", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Sketch")
            show_sketch_only = True
        else:
            break

cap.release()
cv2.destroyAllWindows()