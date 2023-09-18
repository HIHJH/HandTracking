import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture("exercise1_l.mp4")
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('left1.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        bg = np.zeros_like(img)
        drawing_spec_lines = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(
                bg,
                handLms,
                mpHands.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=5),
                mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        img = bg

        out.write(img)

    cv2.imshow("Finger Counter", img)
    cv2.waitKey(5)

out.release()
cv2.destroyAllWindows()