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
    """트랙바 값에 따라 투명도를 조절하여 'Filtered Image'를 업데이트하는 함수"""
    global canvas_original_layers, unique_colors_tuple, trackbars_initialized

    # 트랙바가 완전히 준비되기 전에는 아무 작업도 하지 않음
    if not trackbars_initialized:
        return

    # 원본 데이터가 없으면 실행하지 않음
    if not canvas_original_layers:
        return

    # 매번 깨끗한 흰색 배경에서 이미지를 다시 생성 시작
    filtered_image = np.ones((480, 640, 3), np.uint8) * 255

    # 저장된 각 색상 레이어에 대해 루프 실행
    for i, color_tuple in enumerate(unique_colors_tuple):
        trackbar_name = f'Color {i+1}'
        
        # 트랙바에서 현재 투명도 값을 읽어옴
        opacity_percent = cv2.getTrackbarPos(trackbar_name, 'Color Controls')
        
        # 트랙바를 읽는 데 실패하면 해당 레이어는 건너뜀
        if opacity_percent == -1:
            continue
        
        # 알파 블렌딩을 위한 가중치 계산
        alpha = opacity_percent / 100.0
        beta = 1.0 - alpha
        
        # 해당 색상의 원본 레이어와 마스크를 가져옴
        layer = canvas_original_layers[color_tuple]
        mask = np.any(layer != [255, 255, 255], axis=-1)

        # 마스크에 그려진 영역이 있을 경우에만 블렌딩 수행
        if mask.any():
            background_roi = filtered_image[mask]
            layer_roi = layer[mask]
            
            # addWeighted 함수로 두 이미지를 합성
            blended_roi = cv2.addWeighted(layer_roi, alpha, background_roi, beta, 0)
            
            # 합성된 부분을 최종 이미지에 적용
            filtered_image[mask] = blended_roi

    # 완성된 이미지를 창에 표시
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
    
    print("\n--- [Color-Trackbar Mapping] ---") # 콘솔 출력 헤더
    for i, color_tuple in enumerate(unique_colors_tuple):
        # === 1. 트랙바 이름 수정 ===
        trackbar_name = f'Color {i+1}'
        cv2.createTrackbar(trackbar_name, 'Color Controls', 100, 100, on_trackbar_change)
        
        # === 2. 콘솔에 RGB 값 출력 ===
        # color_tuple은 (B, G, R) 순서이므로, 출력은 (R, G, B) 순서로 바꿔줍니다.
        r, g, b = color_tuple[2], color_tuple[1], color_tuple[0]
        print(f"{trackbar_name} -> RGB: ({r}, {g}, {b})")

    print("--------------------------------\n")

    trackbars_initialized = True
    # === 오류 수정: UI가 준비될 시간을 1ms 확보 ===
    cv2.waitKey(1)
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
trackbars_initialized = False

# 기존 color_ranges 리스트를 아래와 같이 수정합니다.
color_ranges = [
    # 기존 색상들
    ('red', (0, 70, 50), (10, 255, 255)),     # 빨강
    ('red', (170, 70, 50), (180, 255, 255)),  # 빨강 (H값이 0 주변에 걸쳐있어 두 범위로 나눔)
    ('green', (35, 70, 50), (85, 255, 255)),   # 초록
    ('blue', (90, 70, 50), (130, 255, 255)),  # 파랑

    # === 새로 추가할 무채색 범위 ===
    # H(색상)는 전체 범위(0-179)로, S(채도)와 V(명도)로 구분
    ('black', (0, 0, 0), (179, 255, 50))         # 검은색
]

print("[INFO] 검지만 펴면 그리기 시작")
# ... (이하 print문들은 이전과 동일)

# (이전 코드 생략... 모든 함수와 기본 설정은 그대로입니다)

# --- 메인 루프 시작 ---
while True:
    if not show_sketch_only:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        result = hands.process(image_rgb)
        
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

                # if fingers_up == [False, True, False, False, True]:
                #     print("[INFO] 종료 제스처 감지됨: 프로그램 종료")
                #     cap.release()
                #     cv2.destroyAllWindows()
                #     exit()
                
                # === 스케치 모드 활성화 로직 수정 ===
                # 스케치 모드 활성화는 처음 한 번만 실행되도록 하고,
                # 이후 all(fingers_up)은 포인터 모드로 사용됩니다.
                if all(fingers_up) and not sketch_mode_displayed:
                    sketch_enabled = True
                    sketch_mode_displayed = True
                    sketch_mode_activated_time = cv2.getTickCount()

                # === 제스처 정의 로직 수정 ===
                only_index_up = fingers_up == [False, True, False, False, False]
                index_middle_up = fingers_up == [False, True, True, False, False]
                # '모든 손가락을 편' 제스처를 명확히 정의합니다.
                all_fingers_up = all(fingers_up)

                x = int(landmarks[8].x * w)
                y = int(landmarks[8].y * h)

                # === if/elif/else 구조로 제스처 우선순위 제어 ===
                if sketch_enabled and only_index_up:
                    # 1. 그리기 모드
                    current_color_tuple = tuple(current_color)
                    if current_color_tuple not in color_layers:
                        color_layers[current_color_tuple] = np.ones((480, 640, 3), np.uint8) * 255
                    
                    if prev_x is not None:
                        cv2.line(color_layers[current_color_tuple], (prev_x, prev_y), (x, y), current_color, 5)
                    prev_x, prev_y = x, y
                    
                    cv2.circle(canvas_display, (x, y), 10, (0, 0, 255), 2)

                elif sketch_enabled and index_middle_up:
                    # 2. 지우기 모드
                    cx, cy = int(landmarks[8].x * w), int(landmarks[8].y * h)
                    for layer in color_layers.values():
                        cv2.circle(layer, (cx, cy), 30, (255, 255, 255), -1)
                    
                    cv2.circle(canvas_display, (cx, cy), 30, (192, 192, 192), 2)
                    prev_x, prev_y = None, None # 지우개 모드에서는 선이 이어지면 안됨

                elif sketch_enabled and all_fingers_up:
                    # === 3. 포인터 모드 (새로 추가된 부분) ===
                    # 포인터만 표시하고, 실제 그림을 그리지는 않습니다.
                    # 선이 이어지지 않도록 prev_x, prev_y를 초기화하는 것이 중요합니다.
                    cv2.circle(canvas_display, (x, y), 15, (255, 100, 100), -1) # 파란색 채운 원
                    cv2.circle(canvas_display, (x, y), 15, (255, 255, 255), 2)  # 흰색 테두리
                    prev_x, prev_y = None, None

                else:
                    # 4. 그 외의 모든 경우 (주먹을 쥐거나 다른 제스처)
                    # 선 그리기를 멈추기 위해 좌표를 초기화합니다.
                    prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None
        
# while 루프 안의 '# === 색상 변경 기능 복원 ===' 블록을 아래 코드로 대체하세요.

        # === 색상 변경 기능 수정 (흰색, 검은색 처리 추가) ===
        if sketch_enabled:
            roi_frame = frame[0:50, 0:50]
            roi_hsv = hsv[0:50, 0:50]
            cv2.rectangle(frame, (0, 0), (50, 50), (200, 200, 200), 2)

            # 이름과 함께 색상 범위를 순회
            for name, lower, upper in color_ranges:
                mask = cv2.inRange(roi_hsv, lower, upper)
                
                if cv2.countNonZero(mask) > 125:
                    new_color = None
                    # === 이 부분이 핵심 수정사항 ===
                    if name == 'black':
                        new_color = (0, 0, 0)       # 순수한 검은색으로 지정
                    else: # 그 외 R,G,B 색상일 경우
                        mean_val = cv2.mean(roi_frame, mask=mask)
                        new_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))

                    if current_color != new_color:
                        print(f"[INFO] 색상 변경됨 ({name}) → {new_color}")
                        current_color = new_color
                    break
        # === 색상 변경 기능 수정 끝 ===
        
        # 현재 선택된 색상을 보여주는 사각형
        cv2.rectangle(frame, (0, 60), (30, 90), current_color, -1)
        
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
            cv2.imshow("Sketch", canvas_display)

    else:
        # (필터링 모드 로직은 이전과 동일)
        if not trackbars_created:
            final_sketch = combine_layers()
            cv2.imshow("Original Image", final_sketch)
            cv2.imshow("Filtered Image", final_sketch)
            create_color_trackbars()
            trackbars_created = True

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        if not show_sketch_only:
            canvas_original_layers = {color: layer.copy() for color, layer in color_layers.items()}
            
            if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Camera")
            if cv2.getWindowProperty("Sketch", cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow("Sketch")
            show_sketch_only = True
        else:
            break

cap.release()
cv2.destroyAllWindows()
