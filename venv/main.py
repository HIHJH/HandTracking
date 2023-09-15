import cv2
import mediapipe as mp

cap = cv2.VideoCapture("fingers.mp4")
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinates = (4,2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks
    #print(multiLandMarks)

    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))
        for point in handPoints:
            cv2.circle(img, point, 5, (0,0,255), cv2.FILLED)

        cv2.line(img, handPoints[8], handPoints[6], (51, 255, 51), 2)
        cv2.line(img, handPoints[12], handPoints[10], (51, 255, 51), 2)
        cv2.line(img, handPoints[16], handPoints[14], (51, 255, 51), 2)
        cv2.line(img, handPoints[20], handPoints[18], (51, 255, 51), 2)
        # upCount = 0
        # for coordinate in fingerCoordinates:
        #     if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]:
        #         upCount += 1
        #
        #     cv2.putText(img, str(upCount), (100,100), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12)
        out.write(img)

    cv2.imshow("Finger Counter", img)
    cv2.waitKey(5)

out.release()
cv2.destroyAllWindows()

