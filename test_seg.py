import cv2
import numpy as np
import time
from ultralytics import YOLO


def main():
    # Hardcoded model path
    model_path = 'best_seg1.pt'
    print(f"Loading YOLOv8 model from: {model_path}...")

    # Load the model
    model = YOLO(model_path)

    # Use GPU if available
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    print(f"Using device: {device}")
    model.to(device)

    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # Prepare colors for visualization
    colors = {
        'helmet': (0, 255, 0),  # Green for helmet
        'mask': (255, 0, 0)  # Blue for mask
    }

    # Class-specific thresholds
    thresholds = {
        'helmet': 0.35,  # Threshold for helmet
        'mask': 0.75  # Threshold for mask
    }

    # Variables for tracking continuous detections
    helmet_first_detected_time = None
    mask_first_detected_time = None
    continuous_detection_threshold = 2.0  # 2 seconds

    # Alert status
    helmet_alert_active = False
    mask_alert_active = False
    dual_alert_active = False

    print("Starting YOLOv8 instance segmentation...")
    print("Press 'q' to quit")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image from webcam")
            break

        # Ensure frame is 640x640 (resize if needed)
        if frame.shape[0] != 640 or frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 640))

        # Keep a copy of the original frame
        display_frame = frame.copy()

        # Run YOLOv8 inference on the frame
        results = model.predict(
            source=frame,
            conf=0.2,  # Low base confidence, we'll filter by class later
            iou=0.45,
            verbose=False,
            device=device
        )

        # Process the results
        result = results[0]

        # Variables to track if both classes are detected in the current frame
        helmet_detected = False
        mask_detected = False
        helmet_confidence = 0.0
        mask_confidence = 0.0

        # Extract masks and draw them
        if result.masks is not None:
            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                # Extract class and confidence
                cls_id = int(box.cls.item())
                conf = box.conf.item()

                try:
                    cls_name = model.names[cls_id].lower()
                except (IndexError, KeyError):
                    cls_name = f"class_{cls_id}"

                # Apply class-specific thresholds
                if cls_name in thresholds:
                    if conf < thresholds[cls_name]:
                        continue  # Skip this detection if below threshold

                    # Track which classes are detected with confidence
                    if cls_name == 'helmet':
                        helmet_detected = True
                        helmet_confidence = conf
                    elif cls_name == 'mask':
                        mask_detected = True
                        mask_confidence = conf

                    # Get instance segmentation mask
                    seg_mask = mask.data.cpu().numpy()

                    # Ensure mask has correct dimensions
                    if seg_mask.shape[1:] != (frame.shape[0], frame.shape[1]):
                        resized_mask = np.zeros((1, frame.shape[0], frame.shape[1]), dtype=np.float32)
                        if seg_mask.shape[1] > 0 and seg_mask.shape[2] > 0:
                            temp_mask = cv2.resize(
                                seg_mask[0], (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                            resized_mask[0] = temp_mask
                        seg_mask = resized_mask

                    # Create binary mask
                    bool_mask = seg_mask[0] > 0.5

                    # Get color for this class
                    color = colors.get(cls_name, (0, 255, 255))  # Default to yellow

                    # Create colored mask for overlay
                    colored_mask = np.zeros_like(display_frame)
                    colored_mask[:, :, 0] = color[0]  # B
                    colored_mask[:, :, 1] = color[1]  # G
                    colored_mask[:, :, 2] = color[2]  # R

                    # Apply the mask overlay on the frame
                    for c in range(3):
                        display_frame[:, :, c] = np.where(
                            bool_mask,
                            0.7 * display_frame[:, :, c] + 0.3 * colored_mask[:, :, c],
                            display_frame[:, :, c]
                        )

                    # Get mask contours for drawing outline
                    mask_uint8 = bool_mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        mask_uint8,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Draw contour
                    cv2.drawContours(display_frame, contours, -1, color, 2)

                    # Get bounding box
                    if box.xyxy.numel() > 0:
                        x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())

                        # Draw rectangle
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                        # Draw label
                        label = f"{cls_name} {conf:.2f}"
                        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(
                            display_frame,
                            (x1, y1 - text_size[1] - 5),
                            (x1 + text_size[0], y1),
                            color,
                            -1
                        )
                        cv2.putText(
                            display_frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )

        # Update continuous detection tracking
        current_time = time.time()

        # Helmet tracking
        if helmet_detected:
            if helmet_first_detected_time is None:
                helmet_first_detected_time = current_time
                helmet_alert_active = False
            elif current_time - helmet_first_detected_time >= continuous_detection_threshold and not helmet_alert_active:
                print("\n!!! ALERT: HELMET DETECTED !!!")
                helmet_alert_active = True
        else:
            helmet_first_detected_time = None
            helmet_alert_active = False

        # Mask tracking
        if mask_detected:
            if mask_first_detected_time is None:
                mask_first_detected_time = current_time
                mask_alert_active = False
            elif current_time - mask_first_detected_time >= continuous_detection_threshold and not mask_alert_active:
                print("\n!!! ALERT: MASK DETECTED !!!")
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
                if not dual_alert_active:
                    print("\n!!! ALERT: BOTH HELMET AND MASK DETECTED !!!")
                    dual_alert_active = True
        else:
            dual_alert_active = False

        # Add alert messages to the display frame
        if dual_alert_active:
            alert_text = "BOTH HELMET AND MASK DETECTED!"
            cv2.putText(display_frame, alert_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            if helmet_alert_active:
                alert_text = "HELMET DETECTED!"
                cv2.putText(display_frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if mask_alert_active:
                alert_text = "MASK DETECTED!"
                cv2.putText(display_frame, alert_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Display thresholds and current detection statuses
        info_text = f"Helmet: {helmet_confidence:.2f}/{thresholds['helmet']:.2f} | Mask: {mask_confidence:.2f}/{thresholds['mask']:.2f}"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("YOLOv8 Instance Segmentation", display_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("YOLOv8 instance segmentation stopped")


if __name__ == "__main__":
    main()