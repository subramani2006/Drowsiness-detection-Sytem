import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam not available")
    exit()

print("Press 'q' to quit the camera window.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break

    cv2.imshow("Camera Test", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
  