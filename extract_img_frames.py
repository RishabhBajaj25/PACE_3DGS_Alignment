import cv2
import os
change
# Path to the video file
video_path = '/media/rishabh/SSD_1/Data/lab_videos_reg/2_20250324_120731.mp4'
# Directory to save the frames
every_nth_frame = 10

save_dir = os.path.join('/media/rishabh/SSD_1/Data/lab_videos_reg', os.path.splitext(os.path.basename(video_path))[0] + '_frames_' + str(every_nth_frame) + '_fps')
os.makedirs(save_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Read the video frame by frame
frame_count = 0
saved_frame_count = 0  # This counts the actual saved frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every nth frame
    if frame_count % every_nth_frame == 0:
        # Rotate frame by 90 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Save the frame as an image file
        frame_path = os.path.join(save_dir, f'frame_{saved_frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        saved_frame_count += 1  # Increment only when a frame is saved

    frame_count += 1

# Release the video capture object
cap.release()
print(f'Extracted {saved_frame_count} frames from the video.')
