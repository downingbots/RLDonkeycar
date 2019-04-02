# https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
import numpy as np
import cv2
import math
import os
import copy
from stat import S_ISREG, ST_MTIME, ST_MODE, ST_CTIME, ST_ATIME
from scipy.stats import linregress
from collections import deque

###########################
# from 
# https://github.com/navoshta/detecting-road-features/tree/master/source/lanetracker
# https://navoshta.com/detecting-road-features/
###########################################

class Line(object):
    """
    Represents a single lane edge line.
    """

    def __init__(self, x, y, h, w):
        """
        Initialises a line object by fitting a 2nd degree polynomial to provided line points.

        Parameters
        ----------
        x   : Array of x coordinates for pixels representing a line.
        y   : Array of y coordinates for pixels representing a line.
        h   : Image height in pixels.
        w   : Image width in pixels.
        """
        # polynomial coefficients for the most recent fit
        self.h = h
        self.w = w
        self.frame_impact = 0
        self.coefficients = deque(maxlen=5)
        self.process_points(x, y)

    def process_points(self, x, y):
        """
        Fits a polynomial if there is enough points to try and approximate a line and updates a queue of coefficients.

        Parameters
        ----------
        x   : Array of x coordinates for pixels representing a line.
        y   : Array of y coordinates for pixels representing a line.
        """
        enough_points = len(y) > 0 and np.max(y) - np.min(y) > self.h * .625
        if enough_points or len(self.coefficients) == 0:
            self.fit(x, y)

    def get_points(self):
        """
        Generates points of the current best fit line.

        Returns
        -------
        Array with x and y coordinates of pixels representing
        current best approximation of a line.
        """
        y = np.linspace(0, self.h - 1, self.h)
        current_fit = self.averaged_fit()
        return np.stack((
            current_fit[0] * y ** 2 + current_fit[1] * y + current_fit[2],
            y
        )).astype(np.int).T

    def averaged_fit(self):
        """
        Returns coefficients for a line averaged across last 5 points' updates.

        Returns
        -------
        Array of polynomial coefficients.
        """
        return np.array(self.coefficients).mean(axis=0)

    def fit(self, x, y):
        """
        Fits a 2nd degree polynomial to provided points and returns its coefficients.

        Parameters
        ----------
        x   : Array of x coordinates for pixels representing a line.
        y   : Array of y coordinates for pixels representing a line.
        """
        self.coefficients.append(np.polyfit(y, x, 2))

    def radius_of_curvature(self):
        """
        Calculates radius of curvature of the line in real world coordinate system (e.g. meters), assuming there are
        27 meters for 720 pixels for y axis and 3.7 meters for 700 pixels for x axis.

        Returns
        -------
        Estimated radius of curvature in meters.
        """
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 27 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        points = self.get_points()
        y = points[:, 1]
        x = points[:, 0]
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        return int(((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0]))

    def camera_distance(self):
        """
        Calculates distance to camera in real world coordinate system (e.g. meters), assuming there are 3.7 meters for
        700 pixels for x axis.

        Returns
        -------
        Estimated distance to camera in meters.
        """
        points = self.get_points()
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        x = points[np.max(points[:, 1])][0]
        return np.absolute((self.w // 2 - x) * xm_per_pix)

################################

class Window(object):
    """
    Represents a scanning window used to detect points likely to represent lane edge lines.
    """

    def __init__(self, y1, y2, x, m=100, tolerance=50):
        """
        Initialises a window object.

        Parameters
        ----------
        y1          : Top y axis coordinate of the window rect.
        y2          : Bottom y axis coordinate of the window rect.
        x           : X coordinate of the center of the window rect
        m           : X axis span, e.g. window rect width would be m*2..
        tolerance   : Min number of pixels we need to detect within a window in order to adjust its x coordinate.
        """
        self.x = x
        self.mean_x = x
        self.y1 = y1
        self.y2 = y2
        self.m = m
        self.tolerance = tolerance

    def pixels_in(self, nonzero, x=None):
        """
        Returns indices of the pixels in `nonzero` that are located within this window.

        Notes
        -----
        Since this looks a bit tricky, I will go into a bit more detail.
        `nonzero` contains two arrays of coordinates of non-zero pixels.
        Say, there were 50 non-zero pixels in the image and `nonzero` would 
        contain two arrays of shape (50, ) with x and y coordinates of those 
        pixels respectively. What we return here is a array of indices
        within those 50 that are located inside this window. Basically the 
        result would be a 1-dimensional array of ints in the [0, 49] range 
        with a size of less than 50.

        Parameters
        ----------
        nonzero : Coordinates of the non-zero pixels in the image.

        Returns
        -------
        Array of indices of the pixels within this window.
        """
        if x is not None:
            self.x = x
        win_indices = (
            (nonzero[0] >= self.y1) & (nonzero[0] < self.y2) &
            (nonzero[1] >= self.x - self.m) & (nonzero[1] < self.x + self.m)
        ).nonzero()[0]
        if len(win_indices) > self.tolerance:
            self.mean_x = np.int(np.mean(nonzero[1][win_indices]))
        else:
            self.mean_x = self.x

        return win_indices

    def coordinates(self):
        """
        Returns coordinates of the bounding rect.

        Returns
        -------
        Tuple of ((x1, y1), (x2, y2))
        """
        return ((self.x - self.m, self.y1), (self.x + self.m, self.y2))

#########################
# gradients.py
# https://raw.githubusercontent.com/navoshta/detecting-road-features/master/source/lanetracker/gradients.py
#########################
class LaneTracker(object):


    def gradient_abs_value_mask(self, img, sobel_kernel=3, axis='x', threshold=(0, 255)):
        """
        Masks the image based on gradient absolute value.
    
        Parameters
        ----------
        image           : Image to mask.
        sobel_kernel    : Kernel of the Sobel gradient operation.
        axis            : Axis of the gradient, 'x' or 'y'.
        threshold       : Value threshold for it to make it to appear in the mask.
    
        Returns
        -------
        Image mask with 1s in activations and 0 in other pixels.
        """
        # Take the absolute value of derivative in x or y given orient = 'x' or 'y'
        if axis == 'x':
            sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if axis == 'y':
            sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        sobel = np.uint8(255 * sobel / np.max(sobel))
        # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
        mask = np.zeros_like(sobel)
        # Return this mask as your binary_output image
        mask[(sobel >= threshold[0]) & (sobel <= threshold[1])] = 1
        return mask
    
    def gradient_magnitude_mask(self, img, sobel_kernel=3, threshold=(0, 255)):
        """
        Masks the image based on gradient magnitude.
    
        Parameters
        ----------
        image           : Image to mask.
        sobel_kernel    : Kernel of the Sobel gradient operation.
        threshold       : Magnitude threshold for it to make it to appear in the mask.
    
        Returns
        -------
        Image mask with 1s in activations and 0 in other pixels.
        """
        # Take the gradient in x and y separately
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the magnitude
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
        # Create a binary mask where mag thresholds are met
        mask = np.zeros_like(magnitude)
        mask[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
        # Return this mask as your binary_output image
        return mask
    
    def gradient_direction_mask(self, img, sobel_kernel=3, threshold=(0, np.pi / 2)):
        """
        Masks the image based on gradient direction.
    
        Parameters
        ----------
        image           : Image to mask.
        sobel_kernel    : Kernel of the Sobel gradient operation.
        threshold       : Direction threshold for it to make it to appear in the mask.
    
        Returns
        -------
        Image mask with 1s in activations and 0 in other pixels.
        """
        # Take the gradient in x and y separately
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the x and y gradients and calculate the direction of the gradient
        direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
        # Create a binary mask where direction thresholds are met
        mask = np.zeros_like(direction)
        # Return this mask as your binary_output image
        mask[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
        return mask
    
    def color_threshold_mask(self, img, threshold=(0, 255)):
        """
        Masks the image based on color intensity.
    
        Parameters
        ----------
        image           : Image to mask.
        threshold       : Color intensity threshold.
    
        Returns
        -------
        Image mask with 1s in activations and 0 in other pixels.
        """
        mask = np.zeros_like(img)
        mask[(img > threshold[0]) & (img <= threshold[1])] = 1
        return mask
    
    def get_edges(self, img, separate_channels=False):
        """
        Masks the image based on a composition of edge detectors: gradient value,
        gradient magnitude, gradient direction and color.
    
        Parameters
        ----------
        img               : Image to mask.
        separate_channels   : Flag indicating if we need to put masks in different color channels.
    
        Returns
        -------
        Image mask with 1s in activations and 0 in other pixels.
        """
        # Convert to HLS color space and separate required channel
        hls = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2HLS).astype(np.float)
        s_channel = hls[:, :, 2]
        # Get a combination of all gradient thresholding masks
        gradient_x = self.gradient_abs_value_mask(s_channel, axis='x', sobel_kernel=3, threshold=(20, 100))
        gradient_y = self.gradient_abs_value_mask(s_channel, axis='y', sobel_kernel=3, threshold=(20, 100))
        magnitude = self.gradient_magnitude_mask(s_channel, sobel_kernel=3, threshold=(20, 100))
        direction = self.gradient_direction_mask(s_channel, sobel_kernel=3, threshold=(0.7, 1.3))
        gradient_mask = np.zeros_like(s_channel)
        gradient_mask[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
        # Get a color thresholding mask
        color_mask = self.color_threshold_mask(s_channel, threshold=(170, 255))
    
        if separate_channels:
            return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
        else:
            mask = np.zeros_like(gradient_mask)
            mask[(gradient_mask == 1) | (color_mask == 1)] = 1
            return mask

#########################
# class LaneTracker(object):
    """
    Tracks the lane in a series of consecutive frames.
    """

    def __init__(self, first_frame, n_windows=9):
        """
        Initialises a tracker object.

        Parameters
        ----------
        first_frame     : First frame of the frame series. We use it to get dimensions and initialise values.
        n_windows       : Number of windows we use to track each lane edge.
        """
        (self.h, self.w, _) = first_frame.shape
        self.win_n = n_windows
        self.left = None
        self.right = None
        self.l_windows = []
        self.r_windows = []
        self.initialize_lines(first_frame)

    def initialize_lines(self, frame):
        """
        Finds starting points for left and right lines (e.g. lane edges) and initialises Window and Line objects.

        Parameters
        ----------
        frame   : Frame to scan for lane edges.
        """
        # Take a histogram of the bottom half of the image
        edges = self.get_edges(frame)
        flat_edges = edges
        # (flat_edges, _) = flatten_perspective(edges)
        histogram = np.sum(flat_edges[int(self.h / 2):, :], axis=0)

        nonzero = flat_edges.nonzero()
        # Create empty lists to receive left and right lane pixel indices
        l_indices = np.empty([0], dtype=np.int)
        r_indices = np.empty([0], dtype=np.int)
        window_height = int(self.h / self.win_n)

        for i in range(self.win_n):
            l_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.l_windows[-1].x if len(self.l_windows) > 0 else np.argmax(histogram[:self.w // 2])
            )
            r_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.r_windows[-1].x if len(self.r_windows) > 0 else np.argmax(histogram[self.w // 2:]) + self.w // 2
            )
            # Append nonzero indices in the window boundary to the lists
            l_indices = np.append(l_indices, l_window.pixels_in(nonzero), axis=0)
            r_indices = np.append(r_indices, r_window.pixels_in(nonzero), axis=0)
            self.l_windows.append(l_window)
            self.r_windows.append(r_window)
        self.left = Line(x=nonzero[1][l_indices], y=nonzero[0][l_indices], h=self.h, w = self.w)
        self.right = Line(x=nonzero[1][r_indices], y=nonzero[0][r_indices], h=self.h, w = self.w)

    def scan_frame_with_windows(self, frame, windows):
        """
        Scans a frame using initialised windows in an attempt to track the lane edges.

        Parameters
        ----------
        frame   : New frame
        windows : Array of windows to use for scanning the frame.

        Returns
        -------
        A tuple of arrays containing coordinates of points found in the specified windows.
        """
        indices = np.empty([0], dtype=np.int)
        nonzero = frame.nonzero()
        window_x = None
        for window in windows:
            indices = np.append(indices, window.pixels_in(nonzero, window_x), axis=0)
            window_x = window.mean_x
        return (nonzero[1][indices], nonzero[0][indices])

    def process(self, frame, draw_lane=True, draw_statistics=True):
        """
        Performs a full lane tracking pipeline on a frame.

        Parameters
        ----------
        frame               : New frame to process.
        draw_lane           : Flag indicating if we need to draw the lane on top of the frame.
        draw_statistics     : Flag indicating if we need to render the debug information on top of the frame.

        Returns
        -------
        Resulting frame.
        """
        edges = self.get_edges(frame)
        # (flat_edges, unwarp_matrix) = flatten_perspective(edges)
        flat_edges = edges
        (l_x, l_y) = self.scan_frame_with_windows(flat_edges, self.l_windows)
        self.left.process_points(l_x, l_y)
        (r_x, r_y) = self.scan_frame_with_windows(flat_edges, self.r_windows)
        self.right.process_points(r_x, r_y)

        if draw_statistics:
            edges = self.get_edges(frame, separate_channels=True)
            # debug_overlay = self.draw_debug_overlay(flatten_perspective(edges)[0])
            # top_overlay = self.draw_lane_overlay(flatten_perspective(frame)[0])
            # debug_overlay = self.draw_debug_overlay(edges[0])
            # top_overlay = self.draw_lane_overlay(frame[0])
            debug_overlay = self.draw_debug_overlay(edges)
            top_overlay = self.draw_lane_overlay(frame)
            debug_overlay = cv2.resize(debug_overlay, (0, 0), fx=0.3, fy=0.3)
            top_overlay = cv2.resize(top_overlay, (0, 0), fx=0.3, fy=0.3)
            frame[:250, :, :] = frame[:250, :, :] * .4
            (h, w, _) = debug_overlay.shape
            frame[20:20 + h, 20:20 + w, :] = debug_overlay
            frame[20:20 + h, 20 + 20 + w:20 + 20 + w + w, :] = top_overlay
            text_x = 20 + 20 + w + w + 20
            self.draw_text(frame, 'Radius of curvature:  {} m'.format(self.radius_of_curvature()), text_x, 80)
            self.draw_text(frame, 'Distance (left):       {:.1f} m'.format(self.left.camera_distance()), text_x, 140)
            self.draw_text(frame, 'Distance (right):      {:.1f} m'.format(self.right.camera_distance()), text_x, 200)

        if draw_lane:
            frame = self.draw_lane_overlay(frame, unwarp_matrix)

        return frame

    def draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def draw_debug_overlay(self, binary, lines=True, windows=True):
        """
        Draws an overlay with debugging information on a bird's-eye view of the road (e.g. after applying perspective
        transform).

        Parameters
        ----------
        binary  : Frame to overlay.
        lines   : Flag indicating if we need to draw lines.
        windows : Flag indicating if we need to draw windows.

        Returns
        -------
        Frame with an debug information overlay.
        """
        if len(binary.shape) == 2:
            img = np.dstack((binary, binary, binary))
        else:
            img = binary
        if windows:
            for window in self.l_windows:
                coordinates = window.coordinates()
                cv2.rectangle(img, coordinates[0], coordinates[1], (1., 1., 0), 2)
            for window in self.r_windows:
                coordinates = window.coordinates()
                cv2.rectangle(img, coordinates[0], coordinates[1], (1., 1., 0), 2)
        if lines:
            cv2.polylines(img, [self.left.get_points()], False, (1., 0, 0), 2)
            cv2.polylines(img, [self.right.get_points()], False, (1., 0, 0), 2)
        return img * 255

    def draw_lane_overlay(self, img, unwarp_matrix=None):
        """
        Draws an overlay with tracked lane applying perspective unwarp to project it on the original frame.

        Parameters
        ----------
        img           : Original frame.
        unwarp_matrix   : Transformation matrix to unwarp the bird's eye view to initial frame. Defaults to `None` (in
        which case no unwarping is applied).

        Returns
        -------
        Frame with a lane overlay.
        """
        # Create an image to draw the lines on
        overlay = np.zeros_like(img).astype(np.uint8)
        points = np.vstack((self.left.get_points(), np.flipud(self.right.get_points())))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        if unwarp_matrix is not None:
            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            overlay = cv2.warpPerspective(overlay, unwarp_matrix, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        return cv2.addWeighted(img, 1, overlay, 0.3, 0)

    def radius_of_curvature(self):
        """
        Calculates radius of the lane curvature by averaging curvature of the edge lines.

        Returns
        -------
        Radius of the lane curvature in meters.
        """
        return int(np.average([self.left.radius_of_curvature(), self.right.radius_of_curvature()]))

##########################




#############################

class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        # orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        # orientation = math.atan2((abs(line[0]) - abs(line[2])), (abs(line[1]) - abs(line[3])))
        orientation = math.atan2((abs(line[0]) - abs(line[2])), (abs(line[1]) - abs(line[3])))
        deg = 90 + math.degrees(orientation)
        if deg < 0:
          deg += 180
        # return 90+math.degrees(orientation)
        return deg

    def checker(self, line_new, line_old):
        '''Check if line have enough distance and angle to be count as similar
        '''
        # for debugging
        mindist = 400
        minangle = 400
        # Parameters to play with
        # min_distance_to_merge = 40   # near
        # min_distance_to_merge2 = 5  # far 
        # max_y = 70
        # max_y = max(line_old[0][1], line_old[1][1], line_new[0][1], line_new[1][1])
        max_y = 70 - max(line_old[1], line_old[3], line_new[1], line_new[3])
        min_distance_to_merge = int(max_y / 2) + 5
        print("max_y %d dist %d" % (max_y,min_distance_to_merge))
     
        # min_angle_to_merge = 25
        min_angle_to_merge = 20
        if self.get_distance(line_old, line_new) < min_distance_to_merge:
          # check the angle between lines
          orientation_new = self.get_orientation(line_new)
          orientation_old = self.get_orientation(line_old)
          angle = abs(orientation_new - orientation_old)
          if self.get_distance(line_old, line_new) < mindist:
            mindist = self.get_distance(line_old, line_new) 
            if minangle > angle:
              minangle = angle
            # if all is ok -- line is similar to others in group
          if abs(orientation_new - orientation_old) < min_angle_to_merge:
            return True
        return False

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        merge_groups = []
        for line_new in lines[1:]:
          groupnum = -1
          mergegroup = []
          for group in groups:
            groupnum += 1
            for line_old in group:
              if self.checker(line_new, line_old):
                mergegroup.append(groupnum)
                break
          mergegrouplen = len(mergegroup)
          if mergegrouplen == 0 or len(group) == 0:
            # add group
            groups.append([line_new])
          else:
            # merge all groups that line is in
            for i in range(mergegrouplen-2):
              groups[mergegroup[0]].extend(groups[mergegroup[mergegrouplen-i-1]])
              del(groups[mergegroup[mergegrouplen-i-1]])
              print("merged line into %d groups" % mergegrouplen)
            # add line to merged group
            groups[mergegroup[0]].append(line_new)
        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        # if 45 < orientation < 135:
        # if 15 < orientation < 165:
        if 15 < orientation < 165:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        if lines is not None:
            for line_i in [l[0] for l in lines]:
                orientation = self.get_orientation(line_i)
                # print("orientation %d"% orientation)
                # if vertical
                # if 60 < orientation < 90:
                if 15 < orientation < 165:
                    lines_y.append(line_i)
                else:
                    lines_x.append(line_i)
        else:
          return None

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        # for i in [lines_x, lines_y]:
        for i in [lines_y]:
                if len(i) > 0:
                    groups = self.merge_lines_pipeline_2(i)
                    merged_lines = []
                    print(groups)
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments1(group))

                    merged_lines_all.extend(merged_lines)

        return merged_lines_all


class LaneLines:

  #############################
  # derived from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

  def ccw(self, A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
  # return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

  # Return true if line segments AB and CD intersect
  def intersect(self, line1, line2):
    A = line1[0]
    B = line1[1]
    C = line2[0]
    D = line2[1]
    return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

  #############################
  # from https://github.com/dmytronasyrov/CarND-LaneLines-P1/blob/master/P1.ipynb

  def left_right_lines(self,lines):
    lines_all_left = []
    lines_all_right = []
    slopes_left = []
    slopes_right = []
    
    if lines is not None:
      for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            
            if slope > 0:
                lines_all_right.append(line)
                slopes_right.append(slope)
            else:
                lines_all_left.append(line)
                slopes_left.append(slope)
                
    filtered_left_lns = self.filter_lines_outliers(lines_all_left, slopes_left, True)
    filtered_right_lns = self.filter_lines_outliers(lines_all_right, slopes_right, False)
    
    return filtered_left_lns, filtered_right_lns

  def filter_lines_outliers(self,lines, slopes, is_left, min_slope = 0.5, max_slope = 0.9):
    if len(lines) < 2:
        return lines
    
    lines_no_outliers = []
    slopes_no_outliers = []
    
    for i, line in enumerate(lines):
        slope = slopes[i]
        
        if min_slope < abs(slope) < max_slope:
            lines_no_outliers.append(line)
            slopes_no_outliers.append(slope)

    slope_median = np.median(slopes_no_outliers)
    slope_std_deviation = np.std(slopes_no_outliers)
    filtered_lines = []
    
    for i, line in enumerate(lines_no_outliers):
        slope = slopes_no_outliers[i]
        intercepts = np.median(line)

        if slope_median - 2 * slope_std_deviation < slope < slope_median + 2 * slope_std_deviation:
            filtered_lines.append(line)

    return filtered_lines

  def median(self, lines, prev_ms, prev_bs):
    if prev_ms is None:
        prev_ms = []
        prev_bs = []
    
    xs = []
    ys = []
    xs_med = []
    ys_med = []
    m = 0
    b = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs += [x1, x2]
            ys += [y1, y2]
    
    if len(xs) > 2 and len(ys) > 2:
#         m, b = np.polyfit(xs, ys, 1)
        m, b, r_value_left, p_value_left, std_err = linregress(xs, ys)

        if len(prev_ms) > 0:
            prev_ms.append(m)
            prev_bs.append(b)
        else:
            return np.poly1d([m, b])
    
    if len(prev_ms) > 0:
        return np.poly1d([np.average(prev_ms), np.average(prev_bs)])
    else:
        return None

  #############################
  # from https://github.com/dmytronasyrov/CarND-LaneLines-P1/blob/master/P1.ipynb
  def getROI(self,img):
    # w 160 h 120
    # remove top 40 lines
    width = img.shape[1]
    height = img.shape[0]
    # print("w %d h %d" % (width, height))
    roi_img = img[50:height, 0:width]
    return roi_img

  ###############

  def linelen(self,line):
     if line is None:
       return -1
     return math.hypot(line[0][0] - line[1][0], line[0][1] - line[1][0])



  def orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0][0] - line[1][0])), abs((line[0][1] - line[1][1])))
        return math.degrees(orientation)

  def scale_x(self, x1, y1, x2, y2, x3, y3):
    dx = (x1 - x2)
    dy = (y1 - y2)
    diffy = y3 - y1
    if dy == 0:
      newx = x1 + diffy
    else:
      newx = x1 + diffy*dx/dy
    return int((newx + x3)/2)

  def lrclines(self, lines, roi):
    global lline, rline, cline, curpos
    # curpos ('l/r/c of l/r/c line': 0-8
    # for

    croi = copy.deepcopy(roi)
    if lines is not None:
      # for line in [lline,rline,cline]:
      for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[1][0]
        y2 = line[1][1]
        cv2.line(croi,(x1,y1),(x2,y2),(0,255,0),2)
        iline = (x1,y1,x2,y2)
    cv2.imshow('ymergedlines',croi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    global seq
    out = self.image_path("/tmp/movie4", seq)
    cv2.imwrite(out, croi)
    print("wrote %s" % (out))
    seq += 1

    return 0,0

  def final_lines(self, img, yellow, white):
    # do look for horizontal lines, and make a single point in middle
    # if multiple horizontal lines, connect the dots to make a new line
    # do average of close nearly parallel lines
      
    yfinal = None
    if yellow is None:
      yfinal = yellow
    elif len(yellow) == 1:
      yfinal = yellow[0]
    else:
      horiz = []
      vert = []
      for line in yellow:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[1][0]
        y2 = line[1][1]
        if (abs(y2-y1)*2 < abs(x2 - x1)):
          # horizontal
          horiz.append(line)
        else:
          # vertical or tweaner
          vert.append(line)
          if yfinal is None or self.linelen(line) > self.linelen(yfinal):
            yfinal = line
            # print("yfinal:")
            # print(yfinal)
      maxdeltaorient = 10
      # combine lines
      if vert is not None:
        for vline in vert:
          # if (vline == yfinal).all():
          if np.array_equal(vline, yfinal):
            pass
          else:
            # case 0: vline intersect yfinal 
            if self.intersect(vline,yfinal):
              pass
            # case 1: vline is below yfinal 
            elif max(vline[0][1],vline[1][1])<min(yfinal[0][1],yfinal[1][1]):
              # append the two?
              tmp = yfinal
              if yfinal[0][1] < yfinal[1][1]:
                tmp[0][0] = vline[0][0]
                tmp[0][1] = vline[0][1]
              else:
                tmp[1][0] = vline[1][0]
                tmp[1][1] = vline[1][1]

              tmporient = self.orientation(tmp)
              finorient = self.orientation(yfinal)
              if (abs(tmporient-finorient) < maxdeltaorient):
                yfinal = tmp
            # case 2: vline is above yfinal 
            elif min(vline[0][1],vline[1][1]) > max(yfinal[0][1],yfinal[1][1]):
              # append the two?
              tmp = yfinal
              if yfinal[0][1] > yfinal[1][1]:
                tmp[0][0] = vline[0][0]
                tmp[0][1] = vline[0][1]
              else:
                tmp[1][0] = vline[1][0]
                tmp[1][1] = vline[1][1]

              tmporient = self.orientation(tmp)
              finorient = self.orientation(yfinal)
              if (abs(tmporient-finorient) < maxdeltaorient):
                yfinal = tmp

            # case 3: vline and yfinal are similar lengths, parallel lines
            # probably both sides of yellow line. average them.
            elif (self.linelen(vline) >= .75 * self.linelen(yfinal) and
              abs(self.orientation(vline) - self.orientation(yfinal)) < maxdeltaorient):
              # set top
              if yfinal[0][1] > yfinal[1][1]:
                if yfinal[0][1] > vline[0][1]:
                  yfinal[0][0] = self.scale_x(vline[0][0], vline[0][1], 
                        vline[1][0], vline[1][1], yfinal[0][0], yfinal[0][1])
                else:
                  yfinal[0][0] = self.scale_x(yfinal[0][0], yfinal[0][1], 
                        yfinal[1][0], yfinal[1][1], vline[0][0], vline[0][1])
              else:
                if yfinal[1][1] > vline[1][1]:
                  yfinal[1][0] = self.scale_x(vline[0][0], vline[0][1], 
                        vline[1][0], vline[1][1], yfinal[1][0], yfinal[1][1])
                else:
                  yfinal[1][0] = self.scale_x(yfinal[0][0], yfinal[0][1], 
                        yfinal[1][0], yfinal[1][1], vline[1][0], vline[1][1])
              # set bottom
              if yfinal[0][1] < yfinal[1][1]:
                if yfinal[1][1] < vline[1][1]:
                  yfinal[0][0] = self.scale_x(yfinal[0][0], yfinal[0][1], 
                        yfinal[1][0], yfinal[1][1], vline[0][0], vline[0][1])
                else:
                  yfinal[0][0] = self.scale_x(vline[0][0], vline[0][1], 
                        vline[1][0], vline[1][1], yfinal[0][0], yfinal[0][1])
              else:
                if yfinal[1][1] > vline[1][1]:
                  yfinal[1][0] = self.scale_x(vline[0][0], vline[0][1], 
                        vline[1][0], vline[1][1], yfinal[1][0], yfinal[1][1])
                else:
                  yfinal[1][0] = self.scale_x(yfinal[0][0], yfinal[0][1], 
                        yfinal[1][0], yfinal[1][1], vline[1][0], vline[1][1])


      if horiz is not None:
        for hline in horiz:
   
          dist = math.hypot(hline[0][0]-hline[1][0], hline[0][1]-hline[1][1])
          if dist > 30:
            continue
          midptx = int((hline[0][0] + hline[1][0])/2)
          midpty = int((hline[0][1] + hline[1][1])/2)
          
          # case 1 midpt below 
          if yfinal is None:
              yfinal = hline # get shape right
              yfinal[0][0] = midptx
              yfinal[0][1] = midpty
              yfinal[0][0] = midptx
              yfinal[0][1] = midpty+1
          elif midpty < min(yfinal[0][1], yfinal[1][0]):
            if yfinal[0][1] < yfinal[1][0]:
              tmp = yfinal
              tmp[0][0] = midptx
              tmp[0][1] = midpty
            else:
              tmp = yfinal
              tmp[1][0] = midptx
              tmp[1][1] = midpty
            tmporient = self.orientation(tmp)
            finorient = self.orientation(yfinal)
            if (abs(tmporient-finorient) < maxdeltaorient):
              yfinal = tmp
          elif midpty > max(yfinal[0][1], yfinal[1][0]):
            if yfinal[0][1] > yfinal[1][0]:
              tmp = yfinal
              tmp[0][0] = midptx
              tmp[0][1] = midpty
            else:
              tmp = yfinal
              tmp[1][0] = midptx
              tmp[1][1] = midpty
            tmporient = self.orientation(tmp)
            finorient = self.orientation(yfinal)
            if (abs(tmporient-finorient) < maxdeltaorient):
              yfinal = tmp

    # for white lines, look for equidistant LR lines from yellow with opp slope
    lwfinal = None
    rwfinal = None
    if white is not None:
     for wline in white:
      if yfinal is None:
        worient = self.orientation(wline)
        if worient > 90:
          if lwfinal is None:
            lwfinal = wline
          elif self.linelen(wline) > self.linelen(lwfinal):
            lwfinal = wline
        else:
          if rwfinal is None:
            rwfinal = wline
          elif self.linelen(wline) > self.linelen(rwfinal):
            rwfinal = wline
      elif max(wline[0][0], wline[1][0]) < min(yfinal[0][0], yfinal[1][0]):
        if lwfinal is None:
          lwfinal = wline
        elif self.linelen(wline) > self.linelen(lwfinal):
          lwfinal = wline
      elif min(wline[0][0], wline[1][0]) > max(yfinal[0][0], yfinal[1][0]):
        if rwfinal is None:
          rwfinal = wline
        elif self.linelen(wline) > self.linelen(rwfinal):
          rwfinal = wline
      else:
        worient = self.orientation(wline)
        yorient = self.orientation(yfinal)
        if self.intersect(wline,yfinal):
          continue
        if worient < yorient:
          # left
          if lwfinal is None:
            lwfinal = wline
          elif self.linelen(wline) > self.linelen(lwfinal):
            lwfinal = wline
        else:
          # right
          if rwfinal is None:
            rwfinal = wline
          elif self.linelen(wline) > self.linelen(rwfinal):
            rwfinal = wline
    print("yellow:")
    print(yfinal)
    # print("white-left:")
    # print(lwfinal)
    # print("white-right:")
    # print(rwfinal)
    return yfinal, lwfinal, rwfinal

  def binary_hsv_mask(self, img, color_range):
    lower = np.array(color_range[0])
    upper = np.array(color_range[1])
    return cv2.inRange(img, lower, upper)

  def process_lines(self, img, color):

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hls = cv2.cvtColor(np.copy(img), cv2.COLOR_RGB2HLS).astype(np.float)
    # region of interest
    # For Simulation
    roi = self.getROI(img)
    # For Oakland
    # roi = self.getROI(hsv_img)
    # cmask = self.binary_hsv_mask(roi, color)
    # cimg = cv2.bitwise_and(roi, roi, mask = cmask)
    # cv2.imshow('yellow image',cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # edges = cv2.Canny(cimg, 100, 200) # [100,200][30, 130][150,255]
    edges = cv2.Canny(roi, 100, 200) # [100,200][30, 130][150,255]
    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ylines = cv2.HoughLinesP(yedges*roi, 0.5, np.pi/180, 20, None, 180, 120)
    # rho – Distance resolution of the accumulator in pixels.
    # theta – Angle resolution of the accumulator in radians.
    # threshold – Accumulator threshold parameter. 
    # minLineLength – Minimum line length. Line segments shorter than that are rejected.
    # maxLineGap – Maximum allowed gap between points on the same line to link them.
    # ylines = cv2.HoughLinesP(yedges, 0.5, np.pi/180, 20, None, 10, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/90, 10, 10, 10, 10)
    croi = copy.deepcopy(roi)
    if lines is not None:
      for line in lines:
        for x1,y1,x2,y2 in line:
          cv2.line(croi,(x1,y1),(x2,y2),(0,255,0),2)

    # cv2.imshow('lines', croi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#    filtered_left_lns, filtered_right_lns = self.left_right_lines(lines)
#    croi = copy.deepcopy(roi)
#    if filtered_left_lns is not None:
#      for line in filtered_left_lns:
#        for x1,y1,x2,y2 in line:
#          cv2.line(croi,(x1,y1),(x2,y2),(0,255,0),2)
#    if filtered_right_lns is not None:
#      for line in filtered_right_lns:
#        for x1,y1,x2,y2 in line:
#          cv2.line(croi,(x1,y1),(x2,y2),(0,255,0),2)
#    cv2.imshow('ymergedlines',croi)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    hb = HoughBundler()
    mergedlines = hb.process_lines(lines)
    '''
    '''
    croi = copy.deepcopy(roi)
    if mergedlines is not None:
      for line in mergedlines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[1][0]
        y2 = line[1][1]
        cv2.line(croi,(x1,y1),(x2,y2),(0,255,0),2)
        iline = (x1,y1,x2,y2)
    return mergedlines

  def image_path(self, tub_path, frame_id):
        return os.path.join(tub_path, str(frame_id) + "_cam-image_array_.jpg")

  def test_tub(self, tub_path):
        seqs = [ int(f.split("_")[0]) for f in os.listdir(tub_path) if f.endswith('.jpg') ]
        seqs.sort()
        entries = ((os.stat(self.image_path(tub_path, seq))[ST_ATIME], seq) for seq in seqs)

        (last_ts, seq) = next(entries)
        clips = [[seq]]
        for next_ts, next_seq in entries:
            if next_ts - last_ts > 100:  #greater than 1s apart
                clips.append([next_seq])
            else:
                clips[-1].append(next_seq)
            last_ts = next_ts

        for clip in clips:
          for imgseq in clip:
            # if imgseq < 2795:
              # continue
            imgname = self.image_path(tub_path, imgseq)
            img = cv2.imread(imgname)

            # color not currently used
            yellow = [[20, 80, 100], [35, 255, 255]]
            ylines = self.process_lines(img,yellow)
            roi = self.getROI(img)
            steering, position = self.lrclines(ylines,roi)
            continue
            #



            if ylines is None:
              yellow = [[20, 0, 100], [30, 255, 255]]
              ylines = self.process_lines(img,yellow)

            saturation = 40
            white = [[0,0,255-saturation],[255,saturation,255]]
            yline = None
            lwline = None
            rwline = None
            # wlines = self.process_lines(img,white)
            # yline, lwline, rwline = self.final_lines(roi, ylines, wlines)

            if yline is not None:
              cv2.line(roi,(yline[0][0], yline[0][1]),(yline[1][0],yline[1][1]),(0,255,0),2)
            if lwline is not None:
              cv2.line(roi,(lwline[0][0], lwline[0][1]),(lwline[1][0],lwline[1][1]),(0,255,0),2)
            if rwline is not None:
              cv2.line(roi,(rwline[0][0], rwline[0][1]),(rwline[1][0],rwline[1][1]),(0,255,0),2)
# to make movie, run:
# ffmpeg -framerate 4 -i /tmp/movie/1%04d_cam-image_array_.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
            # out = self.image_path("/tmp/movie", imgseq)
            # cv2.imwrite(out, roi)
            # cv2.imshow(imgname,roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print("wrote %s" % (out))


  def test_tub2(self, tub_path):
        seqs = [ int(f.split("_")[0]) for f in os.listdir(tub_path) if f.endswith('.jpg') ]
        seqs.sort()
        entries = ((os.stat(self.image_path(tub_path, seq))[ST_ATIME], seq) for seq in seqs)

        (last_ts, seq) = next(entries)
        clips = [[seq]]
        for next_ts, next_seq in entries:
            if next_ts - last_ts > 100:  #greater than 1s apart
                clips.append([next_seq])
            else:
                clips[-1].append(next_seq)
            last_ts = next_ts

        first_frame = True
        for clip in clips:
          for imgseq in clip:
            # if imgseq < 2795:
              # continue
            imgname = self.image_path(tub_path, imgseq)
            img = cv2.imread(imgname)
            if first_frame:
              LT = LaneTracker(img)
              first_frame = False
            frame = LT.process(img, draw_lane=False, draw_statistics=True)
            # cv2.imshow(imgname,frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            out = self.image_path("/tmp/movie2", imgseq)
            cv2.imwrite(out, frame)
            print("wrote %s" % (out))

global lline, rline, cline, curpos
global seq
lline = None
rline = None 
cline = None 
curpos = 4
seq = 0
y = LaneLines()
# y.test_tub("/home/ros/rope.dk/tub_2_18-03-17/")
y.test_tub2("/home/ros/d2/data/tub_6_18-06-10.bck2")
