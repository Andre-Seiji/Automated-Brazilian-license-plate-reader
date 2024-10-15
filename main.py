# Rodar com Python 3.12
import cv2
import numpy as np
import time
from ultralytics import YOLO
from util import read_license_plate_easyocr

  
def get_coordinates(cap, x_start, x_end, y_start, y_end):
    """
    Get coordinates of Region of Interest. Input values are normalized values between 0 and 1
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    x1 = int(x_start * width)
    x2 = int(x_end * width)
    y1 = int(y_start * height)
    y2 = int(y_end * height)
    
    return (y1, x1), (x1, y1, x2, y2)

def get_frame_region(frame, x1, y1, x2, y2):
    """
    Get Region of Interest
    """
    return frame[y1:y2, x1:x2]


def search_for_plate(license_plate_detector, frame, offset_y, offset_x, xA, yA, xB, yB):
    """
    Return value for plates detected
    """
    detected_plates=[]

    # Select area of interest
    selected_frame = get_frame_region(frame, xA, yA, xB, yB)

    # Detect license plates in area
    license_plates = license_plate_detector(selected_frame, verbose=False)[0]

    # If found, license plate is read
    if len(license_plates.boxes.data.tolist()) > 0:
    
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y
            # Select license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # Image processing
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            # Mitigate light effects e.g. shadow
            rgb_planes = cv2.split(license_plate_crop_gray)
            result_norm_planes = []

            for plane in rgb_planes:
                dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
                bg_img = cv2.medianBlur(dilated_img, 21)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_norm_planes.append(norm_img)
            result_norm = cv2.merge(result_norm_planes)

            # Binary conversion
            _, license_plate_crop_thresh = cv2.threshold(result_norm, 230, 0, cv2.THRESH_TRUNC)
            license_plate_crop_norm = cv2.normalize(license_plate_crop_thresh, license_plate_crop_thresh, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            
            # Read Alphanumeric character
            license_plate_text, _ = read_license_plate_easyocr(license_plate_crop_norm)
            if license_plate_text is not None:
                 detected_plates.append({
                      'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                      'text': license_plate_text,
                      'score': score
                 })

    return detected_plates

def processing_frame(frame, detected_plates):
    """
    Image processing for each frame
    """
    if detected_plates is None:
        return frame

    for plate in detected_plates:
        print(f"Plate: {plate['text']}  Score: {plate['score']:.2f}\n")
        cv2.rectangle(frame, (int(plate['x1']), int(plate['y1'])), (int(plate['x2']), int(plate['y2'])), (0, 255, 0), 2)
        # Display Alphanumeric characters
        cv2.putText(frame, plate['text'], (int(plate['x1']), int(plate['y1']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    return frame


def display(cap, plate_checking_interval, license_plate_detector, reset_frequency, frame_duration, offset_y, offset_x, xA, yA, xB, yB):
    """
    Display model working in real time
    """
    frame_counter = 0
    reset_values_interval_counter=0
    detected_plates=None
    start_total = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        reset_values_interval_counter+=1

        start = time.time()
        
        # Display ROI represented by a rectangle
        cv2.rectangle(frame, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2) 

        # Check if is needed to seach for plates
        if frame_counter % plate_checking_interval == 0:
            reset_values_interval_counter=0
            detected_plates=search_for_plate(license_plate_detector, frame, offset_y, offset_x, xA, yA, xB, yB)
        
        # Check if it is reset period
        if reset_values_interval_counter == reset_frequency:
            detected_plates=None
        
        frame=processing_frame(frame, detected_plates)
        rframe = cv2.resize(frame, (1920, 1080))

        # Display frame
        cv2.imshow('Video', rframe)

        # Control frame rate
        elapsed_time = time.time() - start
        sleep_time=max(0, frame_duration - elapsed_time)
        time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    end_total = time.time()
    print(f"Running time: {(end_total - start_total)} seconds")


def generate_video(cap, plate_checking_interval, license_plate_detector, reset_frequency, offset_y, offset_x, xA, yA, xB, yB):
    """
    Generate output file
    """
    frame_counter = 0
    reset_values_interval_counter=0
    detected_plates=None
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('Example.mp4', cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        reset_values_interval_counter+=1

        # Check if is needed to seach for plates                 
        if frame_counter % plate_checking_interval == 0:
            reset_values_interval_counter=0
            detected_plates=search_for_plate(license_plate_detector, frame, offset_y, offset_x, xA, yA, xB, yB)
        
        # Check if it is reset period
        if reset_values_interval_counter == reset_frequency:
            detected_plates=None

        frame=processing_frame(frame, detected_plates)
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    
    # Load YOLO model
    license_plate_detector = YOLO('')

    # Load video
    video_path = '.mp4'
    cap = cv2.VideoCapture(video_path)

    # Check video
    if not cap.isOpened():
        print("Error while opening file")
        exit()

    # Set frame rate
    fps_original=cap.get(cv2.CAP_PROP_FPS)

    frame_duration=1.0/fps_original

    plate_checking_interval=80

    reset_frequency=20

    # Setting ROI
    x_start, x_end = 0.1, 0.9
    y_start, y_end = 0.5, 1 

    (offset_y, offset_x), (xA, yA, xB, yB) = get_coordinates(cap, x_start, x_end, y_start, y_end)

    display(cap, plate_checking_interval, license_plate_detector, reset_frequency, frame_duration, offset_y, offset_x, xA, yA, xB, yB)
    #generate_video(cap, plate_checking_interval, license_plate_detector, reset_frequency, offset_y, offset_x, xA, yA, xB, yB)


if __name__ == "__main__":
    main()

