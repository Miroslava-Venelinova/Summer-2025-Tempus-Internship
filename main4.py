from ugot import ugot
import cv2
import numpy as np
import time

# Create UGOT object
got = ugot.UGOT()

# Initialize connection (replace with your UGOT IP)
got.initialize('192.168.15.107')

# Load word recognition model
got.load_models(['word_recognition'])

# Open the camera stream
got.open_camera()

try:
    while True:
        # Get the current frame from UGOT camera
        frame = got.read_camera_data()

        if frame is not None:
            # Decode JPEG bytes to OpenCV image
            nparr = np.frombuffer(frame, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Get recognized text
            text_result = got.get_words_result()

            # Draw recognized text on the image
            if text_result:
                print("Recognized text:", text_result)
                cv2.putText(image, text_result, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                print("No text detected.")

            # Show the image
            cv2.imshow("UGOT Camera with Text Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Waiting for camera frame...")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    cv2.destroyAllWindows()
