from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import time


def initialize_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 640)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow camera to warm up
    return picam2


def process_frame(frame, model):
    # Get image width for middle point calculation
    image_width = frame.shape[1]
    middle_x = image_width // 2

    # Run YOLO inference
    results = model.predict(frame, imgsz=640, conf=0.79)

    # Initialize variables for position detection
    t_x1, t_y1, t_x2, t_y2 = None, None, None, None  # topTether coordinates
    l_x1, l_y1, l_x2, l_y2 = None, None, None, None  # logo coordinates

    # Process detection results
    annotated_frame = frame.copy()

    # Draw middle line for reference
    cv2.line(annotated_frame, (middle_x, 0), (middle_x, frame.shape[0]), (255, 0, 0), 1)

    for result in results[0].boxes.data:
        x1, y1, x2, y2 = map(int, result[:4])
        conf, cls = result[4:6]
        cls = int(cls)

        # Get the class name from the YOLO model
        class_name = model.names[cls]

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw coordinates
        label = f"{class_name} ({conf:.2f})"
        cv2.putText(annotated_frame, f"({x1}, {y1})", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"({x2}, {y2})", (x2, y2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate and mark middle point
        mid_x = (x1 + x2) // 2
        cv2.circle(annotated_frame, (mid_x, y1), 3, (0, 0, 255), -1)
        cv2.putText(annotated_frame, f"Mid: ({mid_x}, {y1})", (mid_x, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Store coordinates based on class
        if class_name == "topTether":
            t_x1, t_y1, t_x2, t_y2 = x1, y1, x2, y2
        elif class_name == "logo":
            l_x1, l_y1, l_x2, l_y2 = x1, y1, x2, y2
        if(class_name == "inverted_topTether" or class_name == "inverted_logo"):
            logo_status = "DEFECTIVE LOGO"
            logo_color = (0, 0, 255)  # red
            cv2.putText(annotated_frame, logo_status, (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, logo_color, 3)
        else:
            # Insert the requested logic here
            if t_x1 is not None and l_x1 is not None:
                t_mid_x = (t_x1 + t_x2) // 2
                l_mid_x = (l_x1 + l_x2) // 2
                width_t = t_x2 - t_x1

                diff_x = l_mid_x - t_mid_x
                percentage = (diff_x / width_t) * 100

                # Determine horizontal position status
                if percentage < 1:
                    h_status = "under Normal Position"
                    h_color = (0, 255, 255)  # Yellow
                elif percentage > 10:
                    h_status = "above of Normal Position"
                    h_color = (0, 0, 255)  # Red
                else:
                    h_status = "Normal Position"
                    h_color = (0, 255, 0)  # Green

                # Determine up/down position based on x2
                v_status = "Right" if x2 > middle_x else "Left"

                # Determine if logo is normal or defective
                if h_status == "Normal Position" and v_status == "Right":
                    logo_status = "NORMAL LOGO"
                    logo_color = (0, 255, 0)  # Green
                else:
                    logo_status = "DEFECTIVE LOGO"
                    logo_color = (0, 0, 255)  # Red

                # Display measurements and status
                cv2.putText(annotated_frame, f"t_mid_x: {t_mid_x}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"l_mid_x: {l_mid_x}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"width_t: {width_t}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"Pct: {percentage:.2f}%", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"H-Status: {h_status}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, h_color, 2)
                cv2.putText(annotated_frame, f"V-Position: {v_status}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                # Display logo status in large text at the bottom
                cv2.putText(annotated_frame, logo_status, (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, logo_color, 3)

                print(f"Percentage Difference: {percentage:.2f}% - {h_status} - {v_status} - {logo_status}")
            
            

    

    return annotated_frame


def main():
    try:
        # Initialize camera
        picam2 = initialize_camera()

        # Load YOLO model
        model = YOLO("bestV2.pt")
        print("Model loaded successfully")
        print("Starting position detection. Press 'q' to exit.")

        while True:
            # Capture frame
            frame = picam2.capture_array()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame and perform detection
            annotated_frame = process_frame(rgb_frame, model)

            # Display the frame
            cv2.imshow("Position Detection", annotated_frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        picam2.stop()
        cv2.destroyAllWindows()


if _name_ == "_main_":
    main()
