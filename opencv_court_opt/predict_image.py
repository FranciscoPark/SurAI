import cv2
from court_detector import CourtDetector
import time
# Path to the input image
input_image_path = '/home/isco/Desktop/ambient/opencv_court_opt/trials.png'
start_time = time.time()
# Initialize CourtDetector
court_detector = CourtDetector()

# Read the input image
frame = cv2.imread(input_image_path)

# Detect the court lines
lines = court_detector.detect(frame)
end_time = time.time()
runtime = end_time - start_time
print("Runtime:", runtime, "seconds")

# Draw the court lines on the frame
for i in range(0, len(lines), 4):
    x1, y1, x2, y2 = lines[i], lines[i+1], lines[i+2], lines[i+3]
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

# Save the result image
output_image_path = '/home/isco/Desktop/ambient/opencv_court_opt/result2.png'
cv2.imwrite(output_image_path, frame)

# # Display the frame with court lines
# cv2.imshow('Court Detection', frame)

# # Wait for a key press and then close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
