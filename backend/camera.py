import cv2

# Open the default webcam
camera = cv2.VideoCapture(0)

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Read one frame
    success, frame = camera.read()

    if not success:
        print("Failed to capture frame.")
        break

    # Display video
    cv2.imshow("ClassPulse Camera", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()