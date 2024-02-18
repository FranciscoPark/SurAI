import argparse
import cv2
from court_detector import CourtDetector

# parse parameters
#parser = argparse.ArgumentParser()
#parser.add_argument("--input_video_path", type=str)
#args = parser.parse_args()

#input_video_path = args.input_video_path
input_video_path = '/home/isco/Desktop/ambient/opencv_court/test3.png'
# initialize CourtDetector
court_detector = CourtDetector()

# load video
video = cv2.VideoCapture(input_video_path)

# iterate through each frame of the video
while True:
    ret, frame = video.read()

    if ret:
        # detect the court lines
        lines = court_detector.detect(frame)
        
        # draw the court lines on the frame
        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i+1], lines[i+2], lines[i+3]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

        # display the frame with court lines
        cv2.imshow('Court Detection', frame)
        
        # check if the user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# release the video capture object
video.release()
cv2.destroyAllWindows()
