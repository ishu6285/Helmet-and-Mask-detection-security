import cv2
import time
from ultralytics import YOLO
import numpy as np
import threading


def main():
    # Load the YOLOv11 model
    print("Loading YOLOv11 model...")

    # Specify your custom model path here
    model_path = 'best.pt'  # Update this to your custom model path
    model = YOLO(model_path)  # Using the custom model with helmet and mask classes

    # Move model to GPU if available
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    print(f"Using device: {device}")
    model.to(device)

    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set video properties for smooth 60fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60fps from camera if supported

    # Variables for detection timing
    detection_interval = 1.0  # Detection interval in seconds
    last_detection_time = time.time()

    # Variables for tracking continuous detections
    helmet_first_detected_time = None
    mask_first_detected_time = None
    continuous_detection_threshold = 3.0  # 3 seconds

    # Alert status
    helmet_alert_active = False
    mask_alert_active = False

    # Variables for smooth display
    latest_results = None
    latest_boxes = []
    results_lock = threading.Lock()
    detection_running = False
    stop_threads = False

    # Function to run detection in a separate thread
    def run_detection(frame):
        nonlocal latest_results, latest_boxes, detection_running
        nonlocal helmet_first_detected_time, mask_first_detected_time
        nonlocal helmet_alert_active, mask_alert_active

        try:
            # Run prediction with the model - use a lower base threshold
            # We'll filter specific classes later
            results = model.predict(
                source=frame.copy(),  # Make a copy to avoid race conditions
                conf=0.25,  # Base confidence threshold
                iou=0.45,  # NMS IOU threshold
                max_det=20,  # Maximum detections per image
                verbose=False
            )

            # Process and store results
            if results and len(results) > 0:
                detected_boxes = []

                # Check for specific classes
                helmet_detected = False
                mask_detected = False

                # Print detection information
                print(f"\n--- Detection at {time.strftime('%H:%M:%S')} ---")

                for r in results:
                    boxes = r.boxes  # Boxes object for bbox outputs

                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        cls_name = model.names[cls_id]
                        conf = round(box.conf[0].item(), 2)

                        # Apply class-specific confidence thresholds

                        class_threshold = 0.25  # Default threshold

                        if cls_name.lower() == 'hel':
                            class_threshold = 0.8
                            if conf >= class_threshold:
                                helmet_detected = True
                        elif cls_name.lower() == 'mask':
                            class_threshold = 0.85
                            if conf >= class_threshold:
                                mask_detected = True

                        # Skip this detection if confidence is below the class-specific threshold
                        if conf < class_threshold:
                            continue

                        # Get coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Store detection data
                        detected_boxes.append({
                            'class': cls_name,
                            'confidence': conf,
                            'box': (x1, y1, x2, y2)
                        })

                        print(f"Detected: {cls_name}, Confidence: {conf}, Bounding Box: ({x1}, {y1}, {x2}, {y2})")

                # Update continuous detection tracking
                current_time = time.time()

                # Helmet tracking
                if helmet_detected:
                    if helmet_first_detected_time is None:
                        helmet_first_detected_time = current_time
                    elif current_time - helmet_first_detected_time >= continuous_detection_threshold and not helmet_alert_active:
                        print("\n!!! ALERT: REMOVE YOUR HELMET !!!")
                        helmet_alert_active = True
                else:
                    helmet_first_detected_time = None
                    helmet_alert_active = False

                # Mask tracking
                if mask_detected:
                    if mask_first_detected_time is None:
                        mask_first_detected_time = current_time
                    elif current_time - mask_first_detected_time >= continuous_detection_threshold and not mask_alert_active:
                        print("\n!!! ALERT: REMOVE YOUR MASK !!!")
                        mask_alert_active = True
                else:
                    mask_first_detected_time = None
                    mask_alert_active = False

                # Both helmet and mask detected together
                if helmet_detected and mask_detected:
                    if (helmet_first_detected_time is not None and
                            mask_first_detected_time is not None and
                            current_time - helmet_first_detected_time >= continuous_detection_threshold and
                            current_time - mask_first_detected_time >= continuous_detection_threshold):
                        print("\n!!! ALERT: REMOVE BOTH HELMET AND MASK !!!")

                # Update the global detection results (thread-safe)
                with results_lock:
                    latest_results = results[0]
                    latest_boxes = detected_boxes

        finally:
            detection_running = False

    print("Starting object detection. Press 'q' to quit.")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        current_time = time.time()

        # Check if it's time for a new detection and no detection is currently running
        if current_time - last_detection_time >= detection_interval and not detection_running:
            last_detection_time = current_time
            detection_running = True

            # Start detection in a separate thread
            detection_thread = threading.Thread(target=run_detection, args=(frame,))
            detection_thread.daemon = True
            detection_thread.start()

        # Create a copy of the frame for annotation
        display_frame = frame.copy()

        # Add latest detection results to the frame (thread-safe)
        with results_lock:
            if latest_boxes:
                # Draw boxes and labels manually
                for detection in latest_boxes:
                    x1, y1, x2, y2 = detection['box']
                    label = f"{detection['class']} {detection['confidence']:.2f}"

                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(display_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)

                    # Draw label text
                    cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Add alert messages if active
            if helmet_alert_active and mask_alert_active:
                # Both detected - combined alert
                alert_text = "REMOVE BOTH HELMET AND MASK!"
                cv2.putText(display_frame, alert_text, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                # Individual alerts
                if helmet_alert_active:
                    alert_text = "REMOVE YOUR HELMET!"
                    cv2.putText(display_frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if mask_alert_active:
                    alert_text = "REMOVE YOUR MASK!"
                    cv2.putText(display_frame, alert_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Display the frame with annotations
        cv2.imshow('YOLOv11 Detection (60 FPS)', display_frame)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Object detection stopped")


if __name__ == "__main__":
    main()