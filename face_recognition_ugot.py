from ugot import ugot
import time

# Create UGOT object
got = ugot.UGOT()

# Initialize connection (replace with your UGOT IP)
got.initialize('192.168.15.107')

# Load the color recognition model on the UGOT
#got.load_models(['color_recognition'])
got.load_models(['face_attribute'])
# Run in a loop, print detected color objects
try:
    #while True:
    #    info = got.get_face_characteristic_total_info()
    #    if info:
    #        color, shape, center_x, center_y, height, width, area = info
    #        print(f"Detected {color} {shape} at ({center_x:.2f},{center_y:.2f}) "
    #              f"size: {width:.2f}x{height:.2f}, area: {area:.2f}")
    #    else:
    #        print("Nothing detected.")
    #    time.sleep(0.5)
    while True:
        faces = got.get_face_characteristic_total_info()

        if faces:
            for idx, face in enumerate(faces):
                gender, mask_info, emotion, center_x, center_y, height, width, area = face
                print(f"Face {idx + 1}:")
                print(f"  Gender: {gender}")
                print(f"  Mask: {mask_info}")
                print(f"  Emotion: {emotion}")
                print(f"  Center: ({center_x:.2f}, {center_y:.2f})")
                print(f"  Size: {width:.2f} x {height:.2f}")
                print(f"  Area: {area:.2f}")
                print("-" * 30)
        else:
            print("No face detected.")

        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nStopping...")

