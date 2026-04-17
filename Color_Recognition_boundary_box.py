import time
import cv2
import numpy as np
from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.147.92")
got.load_models(['color_recognition'])
got.open_camera()

try:
    while True:
        frame = got.read_camera_data()

        if frame is not None:

            nparr = np.frombuffer(frame, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            info = got.get_color_total_info()
            if info:
                color, shape, center_x, center_y, height, width, area = info


                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                w = int(width)
                h = int(height)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


                label = f"{color} {shape}"
                cv2.putText(image, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                print(f"Detected {label} at ({center_x:.2f},{center_y:.2f}) "
                      f"size: {width:.2f}x{height:.2f}, area: {area:.2f}")
            else:
                print("Nothing detected.")


            cv2.imshow("UGOT Camera with Color Recognition", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Waiting for camera frame...")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    cv2.destroyAllWindows()
