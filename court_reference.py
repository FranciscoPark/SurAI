import cv2
import numpy as np
import matplotlib.pyplot as plt


class CourtReference:
    """
    Court reference model
    """
    def __init__(self):
        # Define lines as a dictionary
        self.lines = {
            #top of baseline
            'baseline_top': np.array(((286, 561), (1379, 561))),
            #bottom of baseline
            'baseline_bottom': np.array(((286, 2935), (1379, 2935))),
            #net
            'net': np.array(((286, 1748), (1379, 1748))),
            #left side furthest line
            'left_court_line': np.array(((286, 561), (286, 2935))),
            #right side furthest line
            'right_court_line': np.array(((1379, 561), (1379, 2935))),
            #left singles line
            'left_inner_line': np.array(((423, 561), (423, 2935))),
            #right singles line
            'right_inner_line': np.array(((1242, 561), (1242, 2935))),
            #middle service line
            'middle_line': np.array(((832, 1110), (832, 2386))),
            #top service line
            'top_inner_line': np.array(((423, 1110), (1242, 1110))),
            #bottom service line
            'bottom_inner_line': np.array(((423, 2386), (1242, 2386))),
            'top_extra_part': np.array((832.5, 580)),
            'bottom_extra_part': np.array((832.5, 2910))
        }

        # Define court configuration using the line names
        self.court_conf = {
            1: np.array([*self.lines['baseline_top'], *self.lines['baseline_bottom']]),
            2: np.array([self.lines['left_inner_line'][0], self.lines['right_inner_line'][0],
                         self.lines['left_inner_line'][1], self.lines['right_inner_line'][1]]),
            3: np.array([self.lines['left_inner_line'][0], self.lines['right_court_line'][0],
                         self.lines['left_inner_line'][1], self.lines['right_court_line'][1]]),
            4: np.array([self.lines['left_court_line'][0], self.lines['right_inner_line'][0],
                         self.lines['left_court_line'][1], self.lines['right_inner_line'][1]]),
            5: np.array([*self.lines['top_inner_line'], *self.lines['bottom_inner_line']]),
            6: np.array([*self.lines['top_inner_line'], self.lines['left_inner_line'][1],
                         self.lines['right_inner_line'][1]]),
            7: np.array([self.lines['left_inner_line'][0], self.lines['right_inner_line'][0],
                         *self.lines['bottom_inner_line']]),
            8: np.array([self.lines['right_inner_line'][0], self.lines['right_court_line'][0],
                         self.lines['right_inner_line'][1], self.lines['right_court_line'][1]]),
            9: np.array([self.lines['left_court_line'][0], self.lines['left_inner_line'][0],
                         self.lines['left_court_line'][1], self.lines['left_inner_line'][1]]),
            10: np.array([self.lines['top_inner_line'][0], self.lines['middle_line'][0],
                          self.lines['bottom_inner_line'][0], self.lines['middle_line'][1]]),
            11: np.array([self.lines['middle_line'][0], self.lines['top_inner_line'][1],
                          self.lines['middle_line'][1], self.lines['bottom_inner_line'][1]]),
            12: np.array([*self.lines['bottom_inner_line'], self.lines['left_inner_line'][1],
                          self.lines['right_inner_line'][1]])
        }

        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2

        self.court = cv2.cvtColor(cv2.imread('court_configurations/court_reference.png'), cv2.COLOR_BGR2GRAY)

    def build_court_reference(self):
        """
        Create court reference image using the lines positions
        """
        #create a blank image
        court_img = np.zeros((self.court_height + 2 * self.top_bottom_border, self.court_width + 2 * self.right_left_border), dtype=np.uint8)
        
        for line_name, line_points in self.lines.items():
            cv2.line(court_img, tuple(line_points[0]), tuple(line_points[1]), 1, self.line_width)
        
        court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))
        plt.imsave('court_configurations/court_reference.png', court, cmap='gray')
        self.court = court
        return court

    def get_important_lines(self):
        """
        Returns all lines of the court
        """
        lines = np.concatenate([
            self.lines['baseline_top'],
            self.lines['baseline_bottom'],
            self.lines['net'],
            self.lines['left_court_line'],
            self.lines['right_court_line'],
            self.lines['left_inner_line'],
            self.lines['right_inner_line'],
            self.lines['middle_line'],
            self.lines['top_inner_line'],
            self.lines['bottom_inner_line']
        ])
        return lines

    def get_extra_parts(self):
        parts = np.array([
            self.lines['top_extra_part'],
            self.lines['bottom_extra_part']
        ])
        return parts

    def save_all_court_configurations(self):
        """
        Create all configurations of 4 points on court reference
        """
        for i, conf in self.court_conf.items():
            c = cv2.cvtColor(255 - self.court, cv2.COLOR_GRAY2BGR)
            for p in conf:
                c = cv2.circle(c, p, 15, (0, 0, 255), 30)
            cv2.imwrite(f'court_configurations/court_conf_{i}.png', c)

    def get_court_mask(self, mask_type=0):
        """
        Get mask of the court
        """
        mask = np.ones_like(self.court)

        if mask_type == 1:  # lower half court
            mask[:self.lines['net'][0][1] - 1000, :] = 0
        elif mask_type == 2:  # upper half court
            mask[self.lines['net'][0][1]:, :] = 0
        elif mask_type == 3: # court without margins
            mask[:self.lines['baseline_top'][0][1], :] = 0
            mask[self.lines['baseline_bottom'][0][1]:, :] = 0
            mask[:, :self.lines['left_court_line'][0][0]] = 0
            mask[:, self.lines['right_court_line'][0][0]:] = 0

        return mask


if __name__ == '__main__':
    c = CourtReference()
    c.build_court_reference()