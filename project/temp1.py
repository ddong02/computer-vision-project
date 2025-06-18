import cv2
import mediapipe as mp
import numpy as np

color_layers = {}
background_canvas = np.ones((480, 640, 3), np.uint8) * 255 # 최종 배경

# --- 트랙바 관련 전역 변수 및 함수 ---
unique_colors_tuple = [] # 색상 튜플을 저장할 리스트
trackbars_created = False
canvas_original_layers = {} # 'q' 또는 제스처 인식 시점의 레이어 상태를 저장

# --- 기본 설정 ---
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
draw_utils = mp.solutions.drawing_utils

current_color = (0, 0, 0) # BGR
prev_x, prev_y = None, None
sketch_enabled = False
sketch_mode_displayed = False
sketch_mode_activated_time = None
show_sketch_only = False
trackbars_initialized = False
drawing_paused = False
drawing_activated_time = None
SHOW_MESSAGE_DURATION = 1.0
CANVAS_WIDTH, CANVAS_HEIGHT = 640, 480
ERASER_SIZE = 15

color_ranges = [
    ('red', (0, 70, 50), (10, 255, 255)),
    ('red', (170, 70, 50), (180, 255, 255)),
    ('green', (35, 70, 50), (85, 255, 255)),
    ('blue', (90, 70, 50), (130, 255, 255)),
    ('black', (0, 0, 0), (179, 255, 50))
]

def on_trackbar_change(val):
    """트랙바 값에 따라 투명도를 조절하여 'Filtered Image'를 업데이트하는 함수"""
    global canvas_original_layers, unique_colors_tuple, trackbars_initialized

    if not trackbars_initialized or not canvas_original_layers:
        return

    filtered_image = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), np.uint8) * 255

    for i, color_tuple in enumerate(unique_colors_tuple):
        trackbar_name = f'Color {i+1}'
        opacity_percent = cv2.getTrackbarPos(trackbar_name, 'Color Controls')
        
        if opacity_percent == -1:
            continue
        
        alpha = opacity_percent / 100.0
        beta = 1.0 - alpha
        
        layer = canvas_original_layers[color_tuple]
        mask = np.any(layer != [255, 255, 255], axis=-1)

        if mask.any():
            background_roi = filtered_image[mask]
            layer_roi = layer[mask]
            blended_roi = cv2.addWeighted(layer_roi, alpha, background_roi, beta, 0)
            filtered_image[mask] = blended_roi

    cv2.imshow("Filtered Image", filtered_image)


def create_color_trackbars():
    """스케치에서 사용된 색상을 찾아 트랙바를 생성하고, 콘솔에 색상 정보를 출력하는 함수"""
    global unique_colors_tuple, canvas_original_layers, trackbars_initialized

    unique_colors_tuple = list(canvas_original_layers.keys())
    
    if not unique_colors_tuple:
        print("[INFO] No colors drawn to create trackbars.")
        return

    cv2.namedWindow('Color Controls')
    cv2.resizeWindow('Color Controls', 400, len(unique_colors_tuple) * 50)
    
    print("\n--- [Color-Trackbar] ---")
    for i, color_tuple in enumerate(unique_colors_tuple):
        trackbar_name = f'Color {i+1}'
        cv2.createTrackbar(trackbar_name, 'Color Controls', 100, 100, on_trackbar_change)
        
        r, g, b = color_tuple[2], color_tuple[1], color_tuple[0]
        print(f"{trackbar_name} -> RGB: ({r}, {g}, {b})")

    print("--------------------------------\n")

    trackbars_initialized = True
    cv2.waitKey(1)
    on_trackbar_change(0)

def combine_layers():
    """모든 색상 레이어를 합쳐서 하나의 이미지로 만드는 함수"""
    composite_image = background_canvas.copy()
    for color, layer in color_layers.items():
        mask = np.any(layer != [255, 255, 255], axis=-1)
        composite_image[mask] = layer[mask]
    return composite_image

print("[INFO] 검지만 펴면 그리기 시작")

# --- 메인 루프 시작 ---
while True:
    trigger_mode_switch = False

    if not show_sketch_only:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = hands.process(image_rgb)
        
        # canvas_display는 항상 최신 레이어 상태를 반영
        canvas_display = combine_layers()

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                # (손 인식 및 제스처 판별 로직은 동일)
                draw_utils.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                landmarks = handLms.landmark
                h, w = frame.shape[:2]

                fingers_up = []
                for tip_id in [4, 8, 12, 16, 20]:
                    if tip_id == 4:
                        fingers_up.append(landmarks[tip_id].x < landmarks[tip_id - 1].x)
                    else:
                        fingers_up.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)

                only_index_up = fingers_up == [False, True, False, False, False]
                index_middle_up = fingers_up == [False, True, True, False, False]
                all_fingers_up = all(fingers_up)
                is_fist = not any(fingers_up)
                thumb_and_pinky_up = fingers_up == [True, False, False, False, True]

                if thumb_and_pinky_up:
                    trigger_mode_switch = True

                x = int(landmarks[8].x * w)
                y = int(landmarks[8].y * h)

                if sketch_enabled:
                    if is_fist:
                        if not drawing_paused:
                            print("[INFO] 그리기 비활성화 (주먹 감지)")
                            drawing_paused = True
                            drawing_activated_time = None
                        prev_x, prev_y = None, None

                    elif all_fingers_up:
                        if drawing_paused:
                            print("[INFO] 그리기 활성화 (모든 손가락 폄)")
                            drawing_paused = False
                            drawing_activated_time = cv2.getTickCount()
                        
                        cv2.circle(canvas_display, (x, y), 15, (255, 100, 100), -1)
                        cv2.circle(canvas_display, (x, y), 15, (255, 255, 255), 2)
                        prev_x, prev_y = None, None

                    elif not drawing_paused:
                        if only_index_up:
                            current_color_tuple = tuple(current_color)
                            if current_color_tuple not in color_layers:
                                color_layers[current_color_tuple] = np.ones((480, 640, 3), np.uint8) * 255
                            
                            if prev_x is not None:
                                cv2.line(color_layers[current_color_tuple], (prev_x, prev_y), (x, y), current_color, 5)
                            prev_x, prev_y = x, y
                            
                            cv2.circle(canvas_display, (x, y), 10, (0, 0, 255), 2)

                        elif index_middle_up:
                            cx, cy = int(landmarks[8].x * w), int(landmarks[8].y * h)
                            for layer in color_layers.values():
                                cv2.circle(layer, (cx, cy), ERASER_SIZE, (255, 255, 255), -1)
                            
                            cv2.circle(canvas_display, (cx, cy), ERASER_SIZE, (192, 192, 192), 2)
                            prev_x, prev_y = None, None
                        
                        else: prev_x, prev_y = None, None
                    else: prev_x, prev_y = None, None
                else:
                    if all_fingers_up and not sketch_mode_displayed:
                        sketch_enabled = True
                        sketch_mode_displayed = True
                        sketch_mode_activated_time = cv2.getTickCount()
                    prev_x, prev_y = None, None

                pos_text = f"(x, y) : ({x}, {y})"
                cv2.putText(frame, pos_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                r, g, b = current_color[2], current_color[1], current_color[0]
                rgb_text = f"RGB: ({r}, {g}, {b})"
                cv2.putText(frame, rgb_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # (색상 변경 로직은 동일)
        if sketch_enabled:
            can_change_color = drawing_paused
            roi_box_color = (0, 255, 0) if can_change_color else (200, 200, 200)
            cv2.rectangle(frame, (0, 0), (50, 50), roi_box_color, 2)
            cv2.rectangle(frame, (0, 60), (30, 90), current_color, -1)
            
            if can_change_color:
                roi_hsv = hsv[0:50, 0:50]
                for name, lower, upper in color_ranges:
                    mask = cv2.inRange(roi_hsv, lower, upper)
                    if cv2.countNonZero(mask) > 250:
                        new_color = (0,0,0) if name == 'black' else tuple(map(int, cv2.mean(frame[0:50, 0:50], mask=mask)[:3]))
                        if current_color != new_color:
                            print(f"[INFO] 색상 변경됨 ({name}) → {new_color}")
                            current_color = new_color
                        break
        
        if sketch_mode_displayed and sketch_mode_activated_time:
            if (cv2.getTickCount() - sketch_mode_activated_time) / cv2.getTickFrequency() < SHOW_MESSAGE_DURATION:
                cv2.putText(frame, "Sketch Mode ON", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)

        if sketch_enabled:
            if drawing_paused:
                text = "Drawing Paused"
                font_scale = 1.0
                text_color = (0, 0, 255)
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_x = (canvas_display.shape[1] - text_size[0]) // 2
                text_y = 50
                cv2.putText(canvas_display, text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 100), 2, cv2.LINE_AA)
                cv2.putText(canvas_display, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

            elif drawing_activated_time:
                time_since_activation = (cv2.getTickCount() - drawing_activated_time) / cv2.getTickFrequency()
                if time_since_activation < SHOW_MESSAGE_DURATION:
                    text = "Drawing Activated"
                    font_scale = 1.0
                    text_color = (0, 255, 0)
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                    text_x = (canvas_display.shape[1] - text_size[0]) // 2
                    text_y = 50
                    cv2.putText(canvas_display, text, (text_x + 2, text_y + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 100), 2, cv2.LINE_AA)
                    cv2.putText(canvas_display, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
                else:
                    drawing_activated_time = None
            
            cv2.imshow("Sketch", canvas_display)

    else: # if show_sketch_only is True
        if not trackbars_created:
            final_sketch = combine_layers()
            cv2.imshow("Original Image", final_sketch)
            cv2.imshow("Filtered Image", final_sketch)
            create_color_trackbars()
            trackbars_created = True

    key = cv2.waitKey(1)
    q_pressed = key & 0xFF == ord('q')

    if (q_pressed or trigger_mode_switch) and not show_sketch_only:
        if trigger_mode_switch:
            print("[INFO] 트랙바 모드 활성화 (엄지+새끼 제스처 감지)")
        else:
            print("[INFO] 트랙바 모드 활성화 (Q 키 입력)")

        canvas_original_layers = {color: layer.copy() for color, layer in color_layers.items()}
        
        if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Camera")
        if cv2.getWindowProperty("Sketch", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Sketch")
        
        show_sketch_only = True

    elif q_pressed and show_sketch_only:
        break

cap.release()
cv2.destroyAllWindows()
