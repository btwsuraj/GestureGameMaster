import cv2
import mediapipe as mp
import time
import numpy as np
import keyboard  # Import the keyboard library for key input

# Define keycodes for control
left_key_pressed = 'left'
right_key_pressed = 'right'
up_key_pressed = 'up'
down_key_pressed = 'down'

# Sleep for 2 seconds to allow time to focus on the game window
time.sleep(2.0)

# Set up MediaPipe
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define finger tip IDs
tip_ids = [4, 8, 12, 16, 20]

# Open the webcam
video = cv2.VideoCapture(0)

def get_hand_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
                [640, 480]).astype(int))
            output = text, coords
    return output

# Initialize MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        keyPressed = False
        break_pressed = False
        jump_pressed = False
        dunk_pressed = False
        accelerator_pressed = False
        key_count = 0
        key_pressed = None  # Initialize key_pressed as None
        ret, image = video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lm_list = []
        text = ''
        if results.multi_hand_landmarks:
            for idx, classification in enumerate(results.multi_handedness):
                if classification.classification[0].index == idx:
                    label = classification.classification[0].label
                    text = '{}'.format(label)
                else:
                    label = classification.classification[0].label
                    text = '{}'.format(label)
            for hand_landmark in results.multi_hand_landmarks:
                my_hand = results.multi_hand_landmarks[0]
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)
        fingers = []

        if len(lm_list) != 0:
            if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            total = fingers.count(1)
            if total == 4 and text == "Right":
                cv2.rectangle(image, (400, 300), (600, 425), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, "LEFT", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                keyboard.press(left_key_pressed)
                break_pressed = True
                key_pressed = left_key_pressed
                keyPressed = True
                key_count += 1
            elif total == 5 and text == "Left":
                cv2.rectangle(image, (400, 300), (600, 425), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, " RIGHT", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                keyboard.press(right_key_pressed)
                key_pressed = right_key_pressed
                accelerator_pressed = True
                keyPressed = True
                key_count += 1
            elif total == 1:
                cv2.rectangle(image, (400, 300), (600, 425), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, "UP", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                keyboard.press(up_key_pressed)
                key_pressed = up_key_pressed
                jump_pressed = True
                keyPressed = True
                key_count += 1
            elif total == 0:
                cv2.rectangle(image, (400, 300), (600, 425), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, "Down", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                keyboard.press(down_key_pressed)
                key_pressed = down_key_pressed
                down_pressed = True
                keyPressed = True
                key_count += 1

        if not keyPressed and key_pressed is not None:
            keyboard.release(key_pressed)
            key_pressed = None
        elif key_count == 1 and key_pressed is not None:
            for key in [left_key_pressed, right_key_pressed, up_key_pressed, down_key_pressed]:
                if key_pressed != key:
                    keyboard.release(key)
        cv2.imshow("Frame", image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
