import cv2
import numpy as np
import time
from ultralytics import YOLO
import pyttsx3
import threading
import pygame
import requests
import json
from datetime import datetime
import pytz


# Initialize text-to-speech engine
def initialize_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    return engine


# Function to speak text in a separate thread to avoid blocking the main loop
def speak_async(engine, text):
    def speak_thread():
        engine.say(text)
        engine.runAndWait()

    t = threading.Thread(target=speak_thread)
    t.daemon = True
    t.start()
    return t


# Function to generate siren sound
def generate_siren_sound():
    """Generate a siren sound using pygame that can loop"""
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

        # Create shorter siren sound data for looping (2 seconds instead of 5)
        duration = 2.0  # 2 seconds for seamless looping
        sample_rate = 22050

        # Generate siren waveform (alternating frequencies)
        samples = []
        for i in range(int(sample_rate * duration)):
            t = i / sample_rate
            # Create alternating frequency pattern (siren effect)
            freq = 800 + 400 * np.sin(2 * np.pi * 2 * t)  # 2 Hz oscillation between 400-1200 Hz
            amplitude = 0.3 * np.sin(2 * np.pi * freq * t)
            samples.append(int(amplitude * 32767))

        # Convert to stereo
        stereo_samples = [(sample, sample) for sample in samples]

        # Create sound object
        siren_sound = pygame.sndarray.make_sound(np.array(stereo_samples, dtype=np.int16))

        return siren_sound
    except Exception as e:
        print(f"Error generating siren sound: {e}")
        return None


# Function to play continuous siren until stop event is set
def play_continuous_siren(siren_sound, stop_event):
    """Play siren continuously until stop_event is set"""

    def continuous_siren():
        try:
            if siren_sound:
                # Play siren in a loop until stop_event is set
                while not stop_event.is_set():
                    siren_sound.play()
                    # Wait for the sound to finish (2 seconds) or until stop is requested
                    for _ in range(20):  # Check every 0.1 seconds for 2 seconds total
                        if stop_event.is_set():
                            siren_sound.stop()
                            return
                        time.sleep(0.1)
                siren_sound.stop()
        except Exception as e:
            print(f"Error playing continuous siren: {e}")

    t = threading.Thread(target=continuous_siren)
    t.daemon = True
    t.start()
    return t


# Enhanced SMS notification function with Sri Lankan time
def send_sms_notification(detected_items, location="Security Camera 1"):
    """Send SMS notification via notify.lk API with Sri Lankan time"""
    try:
        # Get current time in Sri Lankan timezone
        sri_lanka_tz = pytz.timezone('Asia/Colombo')
        current_time = datetime.now(sri_lanka_tz)
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')

        # Prepare the message based on detected items
        if "helmet" in detected_items and "mask" in detected_items:
            item_type = "HELMET and MASK"
            action_text = "Remove helmet and mask"
        elif "helmet" in detected_items:
            item_type = "HELMET"
            action_text = "Remove helmet"
        elif "mask" in detected_items:
            item_type = "MASK"
            action_text = "Remove mask"
        else:
            item_type = "HELMET or MASK"
            action_text = "Remove detected item"

        # Create the message in the exact format requested
        message = f"""🚨 SECURITY ALERT 🚨
{item_type} detected!
Time: {formatted_time}
Location: {location}
Action Required: {action_text}"""

        # API parameters
        url = "https://app.notify.lk/api/v1/send"
        params = {
            'user_id': '28905',
            'api_key': '2jEPCqTL3eZsMaz7Hxxt',
            'sender_id': 'NotifyDEMO',
            'to': '94712654439',
            'message': message
        }

        headers = {
            'Cookie': 'XSRF-TOKEN=eyJpdiI6IlRwb1FwUGdKT2dvMUR2MHZJV0NKUXc9PSIsInZhbHVlIjoiaFJKRFU2VURhQTZRL083a1grVTIwcGJCY2Q5Um1SWHFCVnFmQnlqSlhrbmp4UVhBM1F5enpTMmdYaHorRFliMElCNmMzRmlxaGpDWHcvemdueTF6UEJ1dldVZGlXcFdDSWlnMWhVNlpoTjJqRGk2MDlrYUwzSUhjKzVvVzI0b1IiLCJtYWMiOiJiNDJiYmVjZDYyZjczYTQ4OGYwNjVhYWZkNGNjYTVlMTNjMGYxN2UyZTQ3OTZlZGI3ZWQxZTVkODZhODc5ZDg5IiwidGFnIjoiIn0%3D; notifylk_session=eyJpdiI6InE0ZXBoT01iU3c0Y3UzbHVCeFZHcWc9PSIsInZhbHVlIjoib0l6S3ZrSFRLV3dSbE0yQ3JZbmMvd3p2Z0R0RUZPRTNma2ZUZlJCTkwrY3FzV2xCWlBqdXRIcndOQVR3ampUWldNQnpGQjRNb1hXNUMxdXY2cUlhYlFvaGtYTEFnM2IvbmlNWmZzUzNVSVFQOUlRNDNXMDcweUFRcjQ2ajRjZ04iLCJtYWMiOiJlZDQ3YjBkNzNkYTc2M2IyYmExYWRkMGFiYzY5ZWNiZTNjMzQxNjhmZDQ0ZGU5ZjBlN2QyZDc4OTVkMjY2NDIyIiwidGFnIjoiIn0%3D'
        }

        # Send the request
        response = requests.post(url, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            print(f"SMS notification sent successfully at {formatted_time}")
            print(f"Message: {message}")
            return True
        else:
            print(f"Failed to send SMS: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"Error sending SMS notification: {e}")
        return False


# Function to send SMS in a separate thread
def send_sms_async(detected_items, location="Security Camera 1"):
    def send_sms():
        send_sms_notification(detected_items, location)

    t = threading.Thread(target=send_sms)
    t.daemon = True
    t.start()
    return t


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
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    # Initialize text-to-speech engine
    tts_engine = initialize_tts()
    print("Text-to-speech engine initialized")

    # Initialize siren sound
    print("Initializing siren sound...")
    siren_sound = generate_siren_sound()
    if siren_sound:
        print("Siren sound initialized successfully")
    else:
        print("Warning: Siren sound initialization failed")

    # Prepare colors for visualization
    colors = {
        'helmet': (0, 255, 0),  # Green for helmet
        'mask': (255, 0, 0)  # Blue for mask
    }

    # Class-specific thresholds
    thresholds = {
        'helmet': 0.70,  # Threshold for helmet
        'mask': 0.80  # Threshold for mask
    }

    # Variables for tracking continuous detections
    helmet_first_detected_time = None
    mask_first_detected_time = None
    continuous_detection_threshold = 2.0  # 2 seconds for initial alert

    # Alert status
    helmet_alert_active = False
    mask_alert_active = False
    dual_alert_active = False

    # Voice alert cooldown
    last_voice_alert_time = 0
    voice_alert_cooldown = 3.0  # seconds between voice alerts

    # Enhanced alert system variables
    helmet_voice_alert_start_time = None
    mask_voice_alert_start_time = None
    dual_voice_alert_start_time = None

    helmet_siren_triggered = False
    mask_siren_triggered = False
    dual_siren_triggered = False

    helmet_siren_start_time = None
    mask_siren_start_time = None
    dual_siren_start_time = None

    helmet_sms_sent = False
    mask_sms_sent = False
    dual_sms_sent = False

    # Speech and siren threads
    helmet_speech_thread = None
    mask_speech_thread = None
    dual_speech_thread = None

    # Continuous siren control
    helmet_siren_stop_event = threading.Event()
    mask_siren_stop_event = threading.Event()
    dual_siren_stop_event = threading.Event()

    helmet_siren_thread = None
    mask_siren_thread = None
    dual_siren_thread = None

    print("Starting YOLOv8 instance segmentation with enhanced alert system...")
    print("Alert sequence: Voice (5s) -> Continuous Siren -> SMS notification with Sri Lankan time")
    print("Siren continues until helmet/mask is removed")
    print("Press 'q' to quit")

    # Initial voice notification
    speak_async(tts_engine, "Enhanced detection system activated")

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
            conf=0.2,
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
        voice_alert_ready = current_time - last_voice_alert_time >= voice_alert_cooldown

        # Helmet tracking with siren control
        prev_helmet_alert = helmet_alert_active
        if helmet_detected:
            if helmet_first_detected_time is None:
                helmet_first_detected_time = current_time
                helmet_alert_active = False
            elif current_time - helmet_first_detected_time >= continuous_detection_threshold and not helmet_alert_active:
                print("\n!!! ALERT: PLEASE REMOVE YOUR HELMET !!!")
                helmet_alert_active = True
        else:
            helmet_first_detected_time = None
            if helmet_alert_active:  # If alert was active and now helmet is removed
                helmet_alert_active = False
                # Stop helmet siren
                helmet_siren_stop_event.set()
                if helmet_siren_thread and helmet_siren_thread.is_alive():
                    helmet_siren_thread.join(timeout=1.0)
                print("Helmet removed - siren stopped")

            # Reset all helmet alert stages when not detected
            helmet_voice_alert_start_time = None
            helmet_siren_triggered = False
            helmet_siren_start_time = None
            helmet_sms_sent = False
            helmet_siren_stop_event.clear()  # Reset the stop event

        # Mask tracking with siren control
        prev_mask_alert = mask_alert_active
        if mask_detected:
            if mask_first_detected_time is None:
                mask_first_detected_time = current_time
                mask_alert_active = False
            elif current_time - mask_first_detected_time >= continuous_detection_threshold and not mask_alert_active:
                print("\n!!! ALERT: PLEASE REMOVE YOUR MASK !!!")
                mask_alert_active = True
        else:
            mask_first_detected_time = None
            if mask_alert_active:  # If alert was active and now mask is removed
                mask_alert_active = False
                # Stop mask siren
                mask_siren_stop_event.set()
                if mask_siren_thread and mask_siren_thread.is_alive():
                    mask_siren_thread.join(timeout=1.0)
                print("Mask removed - siren stopped")

            # Reset all mask alert stages when not detected
            mask_voice_alert_start_time = None
            mask_siren_triggered = False
            mask_siren_start_time = None
            mask_sms_sent = False
            mask_siren_stop_event.clear()  # Reset the stop event

        # Both helmet and mask detected together with siren control
        prev_dual_alert = dual_alert_active
        if helmet_detected and mask_detected:
            if (helmet_first_detected_time is not None and
                    mask_first_detected_time is not None and
                    current_time - helmet_first_detected_time >= continuous_detection_threshold and
                    current_time - mask_first_detected_time >= continuous_detection_threshold):
                if not dual_alert_active:
                    print("\n!!! ALERT: PLEASE REMOVE BOTH HELMET AND MASK !!!")
                    dual_alert_active = True
        else:
            if dual_alert_active:  # If dual alert was active and now items are removed
                dual_alert_active = False
                # Stop dual siren
                dual_siren_stop_event.set()
                if dual_siren_thread and dual_siren_thread.is_alive():
                    dual_siren_thread.join(timeout=1.0)
                print("Items removed - dual siren stopped")

            # Reset all dual alert stages when not detected
            dual_voice_alert_start_time = None
            dual_siren_triggered = False
            dual_siren_start_time = None
            dual_sms_sent = False
            dual_siren_stop_event.clear()  # Reset the stop event

        # ENHANCED ALERT SYSTEM - Handle voice alerts, siren, and SMS notifications
        if voice_alert_ready:
            # Handle dual detection (both helmet and mask)
            if dual_alert_active and not prev_dual_alert:
                dual_speech_thread = speak_async(tts_engine, "Please remove both helmet and mask")
                dual_voice_alert_start_time = current_time
                last_voice_alert_time = current_time
                print("Stage 1: Voice alert for both items started")

            # Handle individual detections
            elif not dual_alert_active:
                if helmet_alert_active and not prev_helmet_alert:
                    helmet_speech_thread = speak_async(tts_engine, "Please remove your helmet")
                    helmet_voice_alert_start_time = current_time
                    last_voice_alert_time = current_time
                    print("Stage 1: Voice alert for helmet started")

                elif mask_alert_active and not prev_mask_alert:
                    mask_speech_thread = speak_async(tts_engine, "Please remove your mask")
                    mask_voice_alert_start_time = current_time
                    last_voice_alert_time = current_time
                    print("Stage 1: Voice alert for mask started")

        # STAGE 2: Trigger continuous siren after 5 seconds of voice alert

        # Dual detection siren
        if (dual_alert_active and dual_voice_alert_start_time is not None and
                not dual_siren_triggered and
                current_time - dual_voice_alert_start_time >= 5.0):
            if siren_sound:
                dual_siren_thread = play_continuous_siren(siren_sound, dual_siren_stop_event)
                dual_siren_triggered = True
                dual_siren_start_time = current_time
                print("Stage 2: Continuous siren activated for both items")
            else:
                print("Stage 2: Siren not available, skipping to SMS")
                dual_siren_triggered = True
                dual_siren_start_time = current_time

        # Helmet siren
        if (helmet_alert_active and not dual_alert_active and
                helmet_voice_alert_start_time is not None and
                not helmet_siren_triggered and
                current_time - helmet_voice_alert_start_time >= 5.0):
            if siren_sound:
                helmet_siren_thread = play_continuous_siren(siren_sound, helmet_siren_stop_event)
                helmet_siren_triggered = True
                helmet_siren_start_time = current_time
                print("Stage 2: Continuous siren activated for helmet")
            else:
                print("Stage 2: Siren not available, skipping to SMS")
                helmet_siren_triggered = True
                helmet_siren_start_time = current_time

        # Mask siren
        if (mask_alert_active and not dual_alert_active and
                mask_voice_alert_start_time is not None and
                not mask_siren_triggered and
                current_time - mask_voice_alert_start_time >= 5.0):
            if siren_sound:
                mask_siren_thread = play_continuous_siren(siren_sound, mask_siren_stop_event)
                mask_siren_triggered = True
                mask_siren_start_time = current_time
                print("Stage 2: Continuous siren activated for mask")
            else:
                print("Stage 2: Siren not available, skipping to SMS")
                mask_siren_triggered = True
                mask_siren_start_time = current_time

        # STAGE 3: Send SMS after 5 seconds of siren
        # Dual detection SMS
        if (dual_alert_active and dual_siren_triggered and
                dual_siren_start_time is not None and
                not dual_sms_sent and
                current_time - dual_siren_start_time >= 5.0):
            send_sms_async(["helmet", "mask"], "Security Camera")
            dual_sms_sent = True
            print("Stage 3: SMS notification sent for both items")

        # Helmet SMS
        if (helmet_alert_active and not dual_alert_active and
                helmet_siren_triggered and helmet_siren_start_time is not None and
                not helmet_sms_sent and
                current_time - helmet_siren_start_time >= 5.0):
            send_sms_async(["helmet"], "Security Camera")
            helmet_sms_sent = True
            print("Stage 3: SMS notification sent for helmet")

        # Mask SMS
        if (mask_alert_active and not dual_alert_active and
                mask_siren_triggered and mask_siren_start_time is not None and
                not mask_sms_sent and
                current_time - mask_siren_start_time >= 5.0):
            send_sms_async(["mask"], "Security Camera")
            mask_sms_sent = True
            print("Stage 3: SMS notification sent for mask")

        # COUNTDOWN DISPLAY AND ALERT MESSAGES
        alert_y_pos = 90

        # Calculate and display countdown timers
        if dual_alert_active:
            alert_text = "PLEASE REMOVE BOTH HELMET AND MASK!"
            cv2.putText(display_frame, alert_text, (50, alert_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show countdown and current stage
            if dual_sms_sent:
                stage_text = "STAGE 3: SMS SENT - SIREN CONTINUES"
                cv2.putText(display_frame, stage_text, (50, alert_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
            elif dual_siren_triggered:
                # Show SMS countdown
                sms_countdown = max(0, 5 - (current_time - dual_siren_start_time))
                stage_text = f"STAGE 2: SIREN ACTIVE - SMS IN {sms_countdown:.1f}s"
                cv2.putText(display_frame, stage_text, (50, alert_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 165, 255), 2)
            elif dual_voice_alert_start_time:
                # Show siren countdown
                siren_countdown = max(0, 5 - (current_time - dual_voice_alert_start_time))
                stage_text = f"STAGE 1: VOICE ALERT - SIREN IN {siren_countdown:.1f}s"
                cv2.putText(display_frame, stage_text, (50, alert_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)

        else:
            if helmet_alert_active:
                alert_text = "PLEASE REMOVE YOUR HELMET!"
                cv2.putText(display_frame, alert_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show countdown and current stage for helmet
                if helmet_sms_sent:
                    stage_text = "HELMET - STAGE 3: SMS SENT - SIREN CONTINUES"
                    cv2.putText(display_frame, stage_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif helmet_siren_triggered:
                    # Show SMS countdown
                    sms_countdown = max(0, 5 - (current_time - helmet_siren_start_time))
                    stage_text = f"HELMET - STAGE 2: SIREN - SMS IN {sms_countdown:.1f}s"
                    cv2.putText(display_frame, stage_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif helmet_voice_alert_start_time:
                    # Show siren countdown
                    siren_countdown = max(0, 5 - (current_time - helmet_voice_alert_start_time))
                    stage_text = f"HELMET - STAGE 1: VOICE - SIREN IN {siren_countdown:.1f}s"
                    cv2.putText(display_frame, stage_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if mask_alert_active:
                alert_text = "PLEASE REMOVE YOUR MASK!"
                cv2.putText(display_frame, alert_text, (50, alert_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Show countdown and current stage for mask
                if mask_sms_sent:
                    stage_text = "MASK - STAGE 3: SMS SENT - SIREN CONTINUES"
                    cv2.putText(display_frame, stage_text, (50, alert_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)
                elif mask_siren_triggered:
                    # Show SMS countdown
                    sms_countdown = max(0, 5 - (current_time - mask_siren_start_time))
                    stage_text = f"MASK - STAGE 2: SIREN - SMS IN {sms_countdown:.1f}s"
                    cv2.putText(display_frame, stage_text, (50, alert_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)
                elif mask_voice_alert_start_time:
                    # Show siren countdown
                    siren_countdown = max(0, 5 - (current_time - mask_voice_alert_start_time))
                    stage_text = f"MASK - STAGE 1: VOICE - SIREN IN {siren_countdown:.1f}s"
                    cv2.putText(display_frame, stage_text, (50, alert_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)

        # Display thresholds and current detection statuses
        info_text = f"Helmet: {helmet_confidence:.2f}/{thresholds['helmet']:.2f} | Mask: {mask_confidence:.2f}/{thresholds['mask']:.2f}"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add continuous detection progress for initial 3-second threshold
        if helmet_detected and helmet_first_detected_time is not None and not helmet_alert_active:
            progress = min(current_time - helmet_first_detected_time,
                           continuous_detection_threshold) / continuous_detection_threshold * 100
            progress_text = f"Helmet Detection: {progress:.0f}%"
            if progress < 100:
                countdown_remaining = continuous_detection_threshold - (current_time - helmet_first_detected_time)
                progress_text += f" ({countdown_remaining:.1f}s to alert)"
            cv2.putText(display_frame, progress_text, (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if mask_detected and mask_first_detected_time is not None and not mask_alert_active:
            progress = min(current_time - mask_first_detected_time,
                           continuous_detection_threshold) / continuous_detection_threshold * 100
            progress_text = f"Mask Detection: {progress:.0f}%"
            if progress < 100:
                countdown_remaining = continuous_detection_threshold - (current_time - mask_first_detected_time)
                progress_text += f" ({countdown_remaining:.1f}s to alert)"
            cv2.putText(display_frame, progress_text, (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Add detection status indicators in top-right corner
        status_x = 400
        if helmet_detected:
            cv2.putText(display_frame, "HELMET DETECTED", (status_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if mask_detected:
            cv2.putText(display_frame, "MASK DETECTED", (status_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Add timestamp
        current_time_str = datetime.now().strftime('%H:%M:%S')
        cv2.putText(display_frame, f"Time: {current_time_str}", (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

        # Show the frame
        cv2.imshow("YOLOv8 Enhanced Detection System with Continuous Siren", display_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup: Stop all sirens before exiting
    print("Stopping all active sirens...")
    helmet_siren_stop_event.set()
    mask_siren_stop_event.set()
    dual_siren_stop_event.set()

    # Wait for siren threads to finish
    if helmet_siren_thread and helmet_siren_thread.is_alive():
        helmet_siren_thread.join(timeout=2.0)
    if mask_siren_thread and mask_siren_thread.is_alive():
        mask_siren_thread.join(timeout=2.0)
    if dual_siren_thread and dual_siren_thread.is_alive():
        dual_siren_thread.join(timeout=2.0)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Clean up pygame
    try:
        pygame.mixer.quit()
    except:
        pass

    print("Enhanced YOLOv8 detection system with continuous siren stopped")


if __name__ == "__main__":
    main()