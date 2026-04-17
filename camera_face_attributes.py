from ugot import ugot
import cv2
import numpy as np
import time

# Create UGOT object
got = ugot.UGOT()

# Initialize connection (replace with your UGOT IP)
got.initialize('192.168.15.107')

# Load face attribute model
got.load_models(['face_attribute'])

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

            # Get face attributes
            faces = got.get_face_characteristic_total_info()

            # Draw detection info on image
            if faces:
                for idx, face in enumerate(faces):
                    gender, mask_info, emotion, center_x, center_y, height, width, area = face

                    # Draw rectangle (approximate)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    w = int(width)
                    h = int(height)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Put text
                    text = f"{gender}, {mask_info}, {emotion}"
                    cv2.putText(image, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the image
            cv2.imshow("UGOT Camera with Face Attributes", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Waiting for camera frame...")

        # You can add a small sleep if you want
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    cv2.destroyAllWindows()
