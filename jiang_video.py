import cv2
import os
import glob

frame_rate = 30  # Frames per second
look_back = 60
ticker = "BRK-A"

# Get all image file paths sorted by name
image_files = sorted(glob.glob(os.path.join(f"images/{look_back}", f"*{ticker}*.png")))

# Get frame size
frame_size = cv2.imread(image_files[0]).shape[1::-1]  # (width, height)

# Define output video file
output_video = f"{ticker}_{look_back}.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Instantiate video writer
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, frame_size)

# Iterrate over images
for image_file in image_files:
    img = cv2.imread(image_file)
    video_writer.write(img)

# Close resources
video_writer.release()
cv2.destroyAllWindows()

print(f"MP4 video saved as {output_video}")
