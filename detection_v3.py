import operator
import os
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from scipy import signal
import imutils

from court_detector import CourtDetector
from sort import Sort
from utils import get_video_properties, get_dtype
import matplotlib.pyplot as plt


class DetectionModel:
    def __init__(self, dtype=torch.FloatTensor):
        self.detection_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        self.detection_model.type(dtype)  # Also moves model to GPU if available
        self.detection_model.eval()
        self.dtype = dtype
        self.PERSON_LABEL = 1
        self.RACKET_LABEL = 43
        self.BALL_LABEL = 37
        self.PERSON_SCORE_MIN = 0.6
        self.PERSON_SECONDARY_SCORE = 0.3
        self.RACKET_SCORE_MIN = 0.6
        self.BALL_SCORE_MIN = 0.6
        self.v_width = 0
        self.v_height = 0
        self.player_1_boxes = []
        self.player_1_2_boxes=[]
        self.player_2_boxes = []
        self.player_2_2_boxes=[]
        self.persons_boxes = {}
        self.persons_dists = {}
        self.persons_first_appearance = {}
        self.counter = 0
        self.num_of_misses = 0
        self.last_frame = None
        self.current_frame = None
        self.next_frame = None
        self.movement_threshold = 200
        self.mot_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.05)

    def detect_player_1(self, image, image_court):
        """
        Detecting bottom player
        """
        boxes = np.zeros_like(image)

        self.v_height, self.v_width = image.shape[:2]
        # Check if player has been detected before
        if len(self.player_1_boxes) == 0:
            # Detect all persons in the ROI
            persons_boxes, _ = self._detect(image_court)
            print('BOXES ', persons_boxes)
            if len(persons_boxes) > 0:
                # Choose person with biggest box
                # biggest_box = sorted(persons_boxes, key=lambda x: area_of_box(x), reverse=True)[0]
                biggest_box = max(persons_boxes, key=lambda x: area_of_box(x)).round()
                print('BIGGEST ', biggest_box)
                self.player_1_boxes.append(biggest_box)
            else:
                return None
        else:
            # Use previous player location to define new ROI
            xt, yt, xb, yb = self.player_1_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xb), int(yb)
            margin = 250
            box_corners = (
            max(xt - margin, 0), max(yt - margin, 0), min(xb + margin, self.v_width), min(yb + margin, self.v_height))
            trimmed_image = image[max(yt - margin, 0): min(yb + margin, self.v_height),
                            max(xt - margin, 0): min(xb + margin, self.v_width), :]

            # Detect all person in ROI
            persons_boxes, _ = self._detect(trimmed_image, self.PERSON_SECONDARY_SCORE)
            if len(persons_boxes) > 0:
                # Find person closest to previous detection
                c1 = center_of_box(self.player_1_boxes[-1])
                closest_box = None
                smallest_dist = np.inf
                for box in persons_boxes:
                    orig_box_location = (
                    box_corners[0] + box[0], box_corners[1] + box[1], box_corners[0] + box[2], box_corners[1] + box[3])
                    c2 = center_of_box(orig_box_location)
                    distance = np.linalg.norm(np.array(c1) - np.array(c2))
                    if distance < smallest_dist:
                        smallest_dist = distance
                        closest_box = orig_box_location
                if smallest_dist < 100:
                    self.counter = 0
                    self.player_1_boxes.append(closest_box)
                else:
                    # Counter is to decide if box has not been found for more than number of frames
                    self.counter += 1
                    self.player_1_boxes.append(self.player_1_boxes[-1])
            else:
                self.player_1_boxes.append(self.player_1_boxes[-1])
                self.num_of_misses += 1
        cv2.rectangle(boxes, (int(self.player_1_boxes[-1][0]), int(self.player_1_boxes[-1][1])),
                      (int(self.player_1_boxes[-1][2]), int(self.player_1_boxes[-1][3])), [0, 0, 255], 2)

        return boxes
    def detect_player_1_2(self, image, image_court): # bottom_1
        """
        Detecting bottom player
        """
        boxes = np.zeros_like(image)

        self.v_height, self.v_width = image.shape[:2]
        # Check if player has been detected before
        if len(self.player_1_2_boxes) == 0:
            # Detect all persons in the ROI
            persons_boxes, _ = self._detect(image_court)
            print('BOXES ', persons_boxes)
            if len(persons_boxes) > 1:
                # Choose person with biggest box
                # biggest_box = sorted(persons_boxes, key=lambda x: area_of_box(x), reverse=True)[0]
                second_box = sorted(persons_boxes, key=lambda x: area_of_box(x), reverse=True)[1].round()
                print('SECOND BIGGEST ', second_box )
                self.player_1_2_boxes.append(second_box )
            else:
                return None
        else:
            # Use previous player location to define new ROI
            xt, yt, xb, yb = self.player_1_2_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xb), int(yb)
            margin = 250
            box_corners = (
            max(xt - margin, 0), max(yt - margin, 0), min(xb + margin, self.v_width), min(yb + margin, self.v_height))
            trimmed_image = image[max(yt - margin, 0): min(yb + margin, self.v_height),
                            max(xt - margin, 0): min(xb + margin, self.v_width), :]

            # Detect all person in ROI
            persons_boxes, _ = self._detect(trimmed_image, self.PERSON_SECONDARY_SCORE)
            if len(persons_boxes) > 0:
                # Find person closest to previous detection
                c1 = center_of_box(self.player_1_2_boxes[-1])
                closest_box = None
                smallest_dist = np.inf
                for box in persons_boxes:
                    orig_box_location = (
                    box_corners[0] + box[0], box_corners[1] + box[1], box_corners[0] + box[2], box_corners[1] + box[3])
                    c2 = center_of_box(orig_box_location)
                    distance = np.linalg.norm(np.array(c1) - np.array(c2))
                    if distance < smallest_dist:
                        smallest_dist = distance
                        closest_box = orig_box_location
                if smallest_dist < 100:
                    self.counter = 0
                    self.player_1_2_boxes.append(closest_box)
                else:
                    # Counter is to decide if box has not been found for more than number of frames
                    self.counter += 1
                    self.player_1_2_boxes.append(self.player_1_2_boxes[-1])
            else:
                self.player_1_2_boxes.append(self.player_1_2_boxes[-1])
                self.num_of_misses += 1
        cv2.rectangle(boxes, (int(self.player_1_2_boxes[-1][0]), int(self.player_1_2_boxes[-1][1])),
                      (int(self.player_1_2_boxes[-1][2]), int(self.player_1_2_boxes[-1][3])), [0, 0, 255], 2)

        return boxes

    def detect_player_2(self, image, image_court):
        """
        Detecting top biggest player
        """
        boxes = np.zeros_like(image)

        self.v_height, self.v_width = image.shape[:2]
        # Check if player has been detected before
        if len(self.player_2_boxes) == 0:

            # Detect all persons in the ROI
            persons_boxes, _ = self._detect(image_court,0.1)
            print('BOXES ', persons_boxes)
            if len(persons_boxes) > 0:
                # Choose person with biggest box
                # biggest_box = sorted(persons_boxes, key=lambda x: area_of_box(x), reverse=True)[0]
                biggest_box = max(persons_boxes, key=lambda x: area_of_box(x)).round()
                print('BIGGEST ', biggest_box)
                self.player_2_boxes.append(biggest_box)
            else:
                return None
        else:
            # Use previous player location to define new ROI
            xt, yt, xb, yb = self.player_2_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xb), int(yb)
            margin = 250
            box_corners = (
            max(xt - margin, 0), max(yt - margin, 0), min(xb + margin, self.v_width), min(yb + margin, self.v_height))
            trimmed_image = image[max(yt - margin, 0): min(yb + margin, self.v_height),
                            max(xt - margin, 0): min(xb + margin, self.v_width), :]

            # Detect all person in ROI
            persons_boxes, _ = self._detect(trimmed_image, 0.1)
            if len(persons_boxes) > 0:
                # Find person closest to previous detection
                c1 = center_of_box(self.player_2_boxes[-1])
                closest_box = None
                smallest_dist = np.inf
                for box in persons_boxes:
                    orig_box_location = (
                    box_corners[0] + box[0], box_corners[1] + box[1], box_corners[0] + box[2], box_corners[1] + box[3])
                    c2 = center_of_box(orig_box_location)
                    distance = np.linalg.norm(np.array(c1) - np.array(c2))
                    if distance < smallest_dist:
                        smallest_dist = distance
                        closest_box = orig_box_location
                if smallest_dist < 100:
                    self.counter = 0
                    self.player_2_boxes.append(closest_box)
                else:
                    # Counter is to decide if box has not been found for more than number of frames
                    self.counter += 1
                    self.player_2_boxes.append(self.player_2_boxes[-1])
            else:
                self.player_2_boxes.append(self.player_2_boxes[-1])
                self.num_of_misses += 1
        cv2.rectangle(boxes, (int(self.player_2_boxes[-1][0]), int(self.player_2_boxes[-1][1])),
                      (int(self.player_2_boxes[-1][2]), int(self.player_2_boxes[-1][3])), [0, 0, 255], 2)

        return boxes

    def detect_player_2_2(self, image, image_court): # bottom_1
        """
        Detecting top player
        """
        boxes = np.zeros_like(image)

        self.v_height, self.v_width = image.shape[:2]
        # Check if player has been detected before
        if len(self.player_2_2_boxes) == 0:
            # Detect all persons in the ROI
            persons_boxes, _ = self._detect(image_court,0.1)
            print('BOXES ', persons_boxes)
            if len(persons_boxes) > 1: 
                # Choose person with biggest box
                # biggest_box = sorted(persons_boxes, key=lambda x: area_of_box(x), reverse=True)[0]
                second_box = sorted(persons_boxes, key=lambda x: area_of_box(x), reverse=True)[1].round()
                print('SECOND BIGGEST ', second_box )
                self.player_2_2_boxes.append(second_box )
            else:
                return None
        else:
            # Use previous player location to define new ROI
            xt, yt, xb, yb = self.player_2_2_boxes[-1]
            xt, yt, xb, yb = int(xt), int(yt), int(xb), int(yb)
            margin = 250
            box_corners = (
            max(xt - margin, 0), max(yt - margin, 0), min(xb + margin, self.v_width), min(yb + margin, self.v_height))
            trimmed_image = image[max(yt - margin, 0): min(yb + margin, self.v_height),
                            max(xt - margin, 0): min(xb + margin, self.v_width), :]

            # Detect all person in ROI
            persons_boxes, _ = self._detect(trimmed_image, 0.1)
            if len(persons_boxes) > 0:
                # Find person closest to previous detection
                c1 = center_of_box(self.player_2_2_boxes[-1])
                closest_box = None
                smallest_dist = np.inf
                for box in persons_boxes:
                    orig_box_location = (
                    box_corners[0] + box[0], box_corners[1] + box[1], box_corners[0] + box[2], box_corners[1] + box[3])
                    c2 = center_of_box(orig_box_location)
                    distance = np.linalg.norm(np.array(c1) - np.array(c2))
                    if distance < smallest_dist:
                        smallest_dist = distance
                        closest_box = orig_box_location
                if smallest_dist < 100:
                    self.counter = 0
                    self.player_2_2_boxes.append(closest_box)
                else:
                    # Counter is to decide if box has not been found for more than number of frames
                    self.counter += 1
                    self.player_2_2_boxes.append(self.player_2_2_boxes[-1])
            else:
                self.player_2_2_boxes.append(self.player_2_2_boxes[-1])
                self.num_of_misses += 1
        cv2.rectangle(boxes, (int(self.player_2_2_boxes[-1][0]), int(self.player_2_2_boxes[-1][1])),
                      (int(self.player_2_2_boxes[-1][2]), int(self.player_2_2_boxes[-1][3])), [0, 0, 255], 2)

        return boxes

  
    def _detect(self, image, person_min_score=None):
        """
        Use deep learning model to detect all person in the image
        """
        if person_min_score is None:
            person_min_score = self.PERSON_SCORE_MIN
        # creating torch.tensor from the image ndarray
        frame_t = image.transpose((2, 0, 1)) / 255
        frame_tensor = torch.from_numpy(frame_t).unsqueeze(0).type(self.dtype)

        # Finding boxes and keypoints
        with torch.no_grad():
            # forward pass
            p = self.detection_model(frame_tensor)

        persons_boxes = []
        probs = []
        for box, label, score in zip(p[0]['boxes'][:], p[0]['labels'], p[0]['scores']):
            if label == self.PERSON_LABEL and score > person_min_score:
                '''cv2.rectangle(boxes, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
                cv2.putText(boxes, 'Person %.3f' % score, (int(box[0]) - 10, int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)'''
                persons_boxes.append(box.detach().cpu().numpy())
                probs.append(score.detach().cpu().numpy())
        return persons_boxes, probs

    def calculate_feet_positions(self, court_detector):
        """
        Calculate the feet position of both players using the inverse transformation of the court and the boxes
        of both players
        """
        inv_mats = court_detector.game_warp_matrix
        positions_1 = []
        positions_2 = []
        # Bottom player feet locations
        for i, box in enumerate(self.player_1_boxes):
            feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2).item(), box[3].item()]).reshape((1, 1, 2))
            feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
            positions_1.append(feet_court_pos)
        mask = []
        # Top player feet locations
        for i, box in enumerate(self.player_2_boxes):
            if box[0] is not None:
                feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2), box[3]]).reshape((1, 1, 2))
                feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
                positions_2.append(feet_court_pos)
                mask.append(True)
            elif len(positions_2) > 0:
                positions_2.append(positions_2[-1])
                mask.append(False)
            else:
                positions_2.append(np.array([0, 0]))
                mask.append(False)

        # Smooth both feet locations
        positions_1 = np.array(positions_1)
        smoothed_1 = np.zeros_like(positions_1)
        smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
        smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)
        positions_2 = np.array(positions_2)
        smoothed_2 = np.zeros_like(positions_2)
        smoothed_2[:, 0] = signal.savgol_filter(positions_2[:, 0], 7, 2)
        smoothed_2[:, 1] = signal.savgol_filter(positions_2[:, 1], 7, 2)

        smoothed_2[not mask, :] = [None, None]
        return smoothed_1, smoothed_2

def mark_player_box(frame, boxes, frame_num):
    if len(boxes)<=frame_num:
        return frame
    box = boxes[frame_num]
    if box[0] is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 0], 2)
    return frame

def center_of_box(box):
    """
    Calculate the center of a box
    """
    if box[0] is None:
        return None, None
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2


def area_of_box(box):
    height = box[3] - box[1]
    width = box[2] - box[0]
    return height * width


def boxes_dist(boxes):
    """
    Calculate the cumulative distance of all the boxes
    """
    total_dist = 0
    for box1, box2 in zip(boxes, boxes[1:]):
        box1_center = np.array(center_of_box(box1))
        box2_center = np.array(center_of_box(box2))
        dist = np.linalg.norm(box2_center - box1_center)
        total_dist += dist
    return total_dist


def sections_intersect(sec1, sec2):
    """
    Check if two sections intersect
    """
    if sec1[0] <= sec2[0] <= sec1[1] or sec2[0] <= sec1[0] <= sec2[1]:
        return True
    return False

def merge(frame, image):
  frame_w = frame.shape[1]
  frame_h = frame.shape[0]

  width = frame_w // 7
  resized = imutils.resize(image, width=width)

  img_w = resized.shape[1]
  img_h = resized.shape[0]

  w = frame_w - img_w

  frame[:img_h, w:] = resized

  return frame 

def draw_ball_position(frame, court_detector, xy, i):
        """
        Calculate the ball position of both players using the inverse transformation of the court and the x, y positions
        """
        inv_mats = court_detector.game_warp_matrix[i]
        coord = xy
        img = frame.copy()
        # Ball locations
        if coord is not None:
          p = np.array(coord,dtype='float64')
          ball_pos = np.array([p[0].item(), p[1].item()]).reshape((1, 1, 2))
          transformed = cv2.perspectiveTransform(ball_pos, inv_mats)[0][0].astype('int64')
          cv2.circle(frame, (transformed[0], transformed[1]), 35, (0,255,255), -1)
        else:
          pass
        return img 

        # # Smooth ball locations
        # positions = np.array(positions)
        # smoothed = np.zeros_like(positions)
        # smoothed[:, 0] = signal.savgol_filter(positions[:, 0], 7, 2)
        # smoothed[:, 1] = signal.savgol_filter(positions[:, 1], 7, 2)

        return positions #smoothed,

def calculate_feet_positions(self, court_detector):
  """
  Calculate the feet position of both players using the inverse transformation of the court and the boxes
  of both players
  """
  inv_mats = court_detector.game_warp_matrix
  positions_1 = []
  positions_2 = []
  # Bottom player feet locations
  for i, box in enumerate(self.player_1_boxes):
      feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2).item(), box[3].item()]).reshape((1, 1, 2))
      feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
      positions_1.append(feet_court_pos)
  mask = []
  # Top player feet locations
  for i, box in enumerate(self.player_2_boxes):
      if box[0] is not None:
          feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2), box[3]]).reshape((1, 1, 2))
          feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
          positions_2.append(feet_court_pos)
          mask.append(True)
      elif len(positions_2) > 0:
          positions_2.append(positions_2[-1])
          mask.append(False)
      else:
          positions_2.append(np.array([0, 0]))
          mask.append(False)

  # Smooth both feet locations
  positions_1 = np.array(positions_1)
  smoothed_1 = np.zeros_like(positions_1)
  smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
  smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)
  positions_2 = np.array(positions_2)
  smoothed_2 = np.zeros_like(positions_2)
  smoothed_2[:, 0] = signal.savgol_filter(positions_2[:, 0], 7, 2)
  smoothed_2[:, 1] = signal.savgol_filter(positions_2[:, 1], 7, 2)

  smoothed_2[not mask, :] = [None, None]
  return smoothed_1, smoothed_2



def create_top_view(court_detector, detection_model, xy, fps):
    """
    Creates top view video of the gameplay
    """
    coords = xy[:]
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('VideoOutput/minimap.mp4',cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (v_width, v_height))
    # players location on court
    smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)
    i = 0 
    for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
        frame = court.copy()
        frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 45, (255, 0, 0), -1)
        if feet_pos_2[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 45, (255, 0, 0), -1)
        draw_ball_position(frame, court_detector, coords[i], i)
        i += 1
        out.write(frame)
    out.release()


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolation(coords):
  coords =coords.copy()
  x, y = [x[0] if x is not None else np.nan for x in coords], [x[1] if x is not None else np.nan for x in coords]

  xxx = np.array(x) # x coords
  yyy = np.array(y) # y coords

  nons, yy = nan_helper(xxx)
  xxx[nons]= np.interp(yy(nons), yy(~nons), xxx[~nons])
  nans, xx = nan_helper(yyy)
  yyy[nans]= np.interp(xx(nans), xx(~nans), yyy[~nans])

  newCoords = [*zip(xxx,yyy)]

  return newCoords

def diff_xy(coords):
  coords = coords.copy()
  diff_list = []
  for i in range(0, len(coords)-1):
    if coords[i] is not None and coords[i+1] is not None:
      point1 = coords[i]
      point2 = coords[i+1]
      diff = [abs(point2[0] - point1[0]), abs(point2[1] - point1[1])]
      diff_list.append(diff)
    else:
      diff_list.append(None)
  
  xx, yy = np.array([x[0] if x is not None else np.nan for x in diff_list]), np.array([x[1] if x is not None else np.nan for x in diff_list])
  
  return xx, yy



def remove_outliers(x, y, coords):
  ids = set(np.where(x > 50)[0]) & set(np.where(y > 50)[0])
  for id in ids:
    left, middle, right = coords[id-1], coords[id], coords[id+1]
    if left is None:
      left = [0]
    if  right is None:
      right = [0]
    if middle is None:
      middle = [0]
    MAX = max(map(list, (left, middle, right)))
    if MAX == [0]:
      pass
    else:
      try:
        coords[coords.index(tuple(MAX))] = None
      except ValueError:
        coords[coords.index(MAX)] = None


