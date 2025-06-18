import cv2
import mediapipe as mp
import numpy as np

# --- 데이터 구조 변경: 단일 켄버스에서 색상별 레이어 딕셔너리로 ---
# 각 색상별로 독립된 켄버스(레이어)를 저장합니다.
# key: 색상(BGR 튜플), value: 해당 색상의 그림이 그려진 numpy 배열
color_layers = {} 
background_canvas = np.ones((480, 640, 3), np.uint8) * 255 # 최종 배경

# --- 트랙바 관련 전역 변수 및 함수 ---
unique_colors_tuple = [] # 색상 튜플을 저장할 리스트
trackbars_created = False
canvas_original_layers = {} # 'q'를 눌렀을 때의 레이어 상태를 저장

def on_trackbar_change(val):
    """트랙바 값이 변경될 때마다 호출되어 레이어를 합쳐 'Filtered Image'를 업데이트하는 함수"""
    global canvas_original_layers

    if not canvas_original_layers:
        return

    # 시작은 깨끗한 흰색 켄버스로
    filtered_image = np.ones((480, 640, 3), np.uint8) * 255

    # 각 색상 레이어에 대해 트랙바 상태 확인 후 합성
    for i, color_tuple in enumerate(unique_colors_tuple):
        trackbar_name = f'Color {i+1} {color_tuple}'
        status = cv2.getTrackbarPos(trackbar_name, 'Color Controls')

        if status == 1:  # 트랙바가 ON 상태이면
            # 해당 색상 레이어를 가져옴
            layer = canvas_original_layers[color_tuple]
            # 해당 레이어에서 그림이 그려진 부분(흰색이 아닌 부분)에 대한 마스크 생성
            mask = np.any(layer != [255, 255, 255], axis=-1)
            # 마스크를 이용해 현재 이미지에 해당 레이어의 그림을 덮어씀
            filtered_image[mask] = layer[mask]

    cv2.imshow("Filtered Image", filtered_image)


def create_color_trackbars():
    """스케치에서 사용된 색상을 찾아 트랙바를 생성하는 함수"""
    global unique_colors_tuple, canvas_original_layers

    # 원본 레이어 딕셔너리의 키(색상 튜플)들을 가져옴
    unique_colors_tuple = list(canvas_original_layers.keys())
    
    if not unique_colors_tuple:
        print("[INFO] No colors drawn to create trackbars.")
        return

    cv2.namedWindow('Color Controls')
    cv2.resizeWindow('Color Controls', 400, len(unique_colors_tuple) * 50)

    for i, color_tuple in enumerate(unique_colors_tuple):
        # 트랙바 이름에 BGR 값을 사용
        trackbar_name = f'Color {i+1} {color_tuple}'
        cv2.createTrackbar(trackbar_name, 'Color Controls', 1, 1, on_trackbar_change)
        print(f"[INFO] Created trackbar for color (BGR): {color_tuple}")

    # 초기 필터링 화면 업데이트를 위해 콜백 함수 호출
    on_trackbar_change(0)

def combine_layers():
    """모든 색상 레이어를 합쳐서 하나의 이미지로 만드는 함수"""
    # 시작은 깨끗한 흰색 켄버스로
    composite_image = background_canvas.copy()
    
    # 각 색상 레이어를 순서대로 덮어씀
    for color, layer in color_layers.items():
        # 그림이 그려진 부분(흰색이 아닌 부분)에 대한 마스크 생성
        mask = np.any(layer != [255, 255, 255], axis=-1)
        # 마스크를 이용해 현재 이미지에 해당 레이어의 그림을 덮어씀
        composite_image[mask] = layer[mask]
        
    return composite_image


# --- 기본 설정 ---
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
draw_utils = mp.solutions.drawing_utils

# canvas_current 는 이제 레이어를 합친 최종본을 보여주는 용도
# canvas_current = np.ones((480, 640, 3), np.uint8) * 255
current_color = (0, 0, 0) # BGR
prev_x, prev_y = None, None
sketch_enabled = False
sketch_mode_displayed = False
sketch_mode_activated_time = None
show_sketch_only = False

color_ranges = [
    ((0, 70, 50), (10, 255, 255)),   # 빨강
    ((35, 70, 50), (85, 255, 255)),  # 초록
    ((90, 70, 50), (130, 255, 255))   # 파랑
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
        
        # --- 레이어 합성 로직 추가 ---
        # 매 프레임마다 모든 레이어를 합쳐서 최신 스케치 상태를 만듦
        canvas_display = combine_layers()

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
                    # --- 그리기 로직 수정 ---
                    # 현재 선택된 색상의 레이어가 없으면 새로 생성
                    current_color_tuple = tuple(current_color)
                    if current_color_tuple not in color_layers:
                        color_layers[current_color_tuple] = np.ones((480, 640, 3), np.uint8) * 255
                    
                    if prev_x is not None:
                        # 해당 색상의 레이어에만 그림
                        cv2.line(color_layers[current_color_tuple], (prev_x, prev_y), (x, y), current_color, 5)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = None, None

                if index_middle_up:
                    # --- 지우개 로직 수정 ---
                    # 모든 레이어에서 해당 부분을 지워야 함 (흰색으로 덮어쓰기)
                    cx, cy = int(landmarks[8].x * w), int(landmarks[8].y * h)
                    for layer in color_layers.values():
                        cv2.circle(layer, (cx, cy), 30, (255, 255, 255), -1)

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
        
        cv2.rectangle(frame, (0, 100), (50, 150), current_color, -1)
        r, g, b = current_color[2], current_color[1], current_color[0]
        rgb_text = f"RGB: ({r}, {g}, {b})"
        cv2.putText(frame, rgb_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if sketch_mode_displayed and sketch_mode_activated_time:
            time_passed = (cv2.getTickCount() - sketch_mode_activated_time) / cv2.getTickFrequency()
            if time_passed < 2.0:
                cv2.putText(frame, "Sketch Mode ON", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", frame)
        if sketch_enabled:
            # 합성된 최종 이미지를 보여줌
            cv2.imshow("Sketch", canvas_display)

    else:  # show_sketch_only == True
        if not trackbars_created:
            # 필터링을 위해 최종 합본과 분리된 레이어를 모두 보여줌
            final_sketch = combine_layers()
            cv2.imshow("Original Image", final_sketch)
            cv2.imshow("Filtered Image", final_sketch) # 초기 필터링 이미지는 원본과 동일
            create_color_trackbars()
            trackbars_created = True

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        if not show_sketch_only:
            # --- 'q'를 눌렀을 때의 상태 저장 방식 변경 ---
            # 현재 레이어들의 상태를 깊은 복사하여 저장
            canvas_original_layers = {color: layer.copy() for color, layer in color_layers.items()}
            
            if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Camera")
            if cv2.getWindowProperty("Sketch", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Sketch")
            show_sketch_only = True
        else:
            break

cap.release()
cv2.destroyAllWindows()
