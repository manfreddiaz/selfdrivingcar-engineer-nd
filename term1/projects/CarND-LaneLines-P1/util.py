import math
import cv2
import numpy as np


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def calculate_lane(shape, lane_points, left=True):
    (vx, vy, x0, y0) = cv2.fitLine(np.array(lane_points), cv2.DIST_L2, 0, 0.01, 0.01)
    if left:
        bound = max(x for x, y in lane_points)
    else:
        bound = min(x for x, y in lane_points)

    bottom_point = (int((shape[0] - y0[0]) * vx[0] / vy[0] + x0[0]), shape[0])
    top_point = (bound, int((bound - x0[0]) * vy[0] / vx[0] + y0[0]))

    return top_point, bottom_point

avg_top_left_lane = None
avg_bottom_left_lane = None
avg_top_right_lane = None
avg_bottom_right_lane = None
alpha = 0.5 # Confluence by Vivek Yadav, tried average first (alpha=0.5), not too much difference between both


def weighted_average(x, y):
    return int(x * (1 - alpha) + y * alpha)


def draw_lines(img, lines, color=[255, 0, 0], thickness=7):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    global avg_top_left_lane, avg_bottom_left_lane
    global avg_top_right_lane, avg_bottom_right_lane

    left_lane_points = []
    right_lane_points = []
    shape = img.shape

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if -0.9 < slope < -0.6 and x1 < shape[1] / 2 and x2 < shape[1] / 2:
                left_lane_points.append((x1, y1))
                left_lane_points.append((x2, y2))
            elif 0.4 < slope < 0.9 and x1 > shape[1] / 2 and x2 > shape[1] / 2:
                right_lane_points.append((x1, y1))
                right_lane_points.append((x2, y2))

    if len(left_lane_points) > 0:
        top_left_lane, bottom_left_lane = calculate_lane(img.shape, left_lane_points)
        if avg_top_left_lane is None and avg_bottom_left_lane is None:
            avg_top_left_lane = top_left_lane
            avg_bottom_left_lane = bottom_left_lane
        else:
            avg_top_left_lane = tuple(map(weighted_average, avg_top_left_lane, top_left_lane))
            avg_bottom_left_lane = tuple(map(weighted_average, avg_bottom_left_lane, bottom_left_lane))

        cv2.line(img, avg_top_left_lane, avg_bottom_left_lane, color, thickness)

    if len(right_lane_points) > 0:
        top_right_lane, bottom_right_lane = calculate_lane(img.shape, right_lane_points, False)
        if avg_bottom_right_lane is None and avg_top_right_lane is None:
            avg_top_right_lane = top_right_lane
            avg_bottom_right_lane = bottom_right_lane
        else:
            avg_bottom_right_lane = tuple(map(weighted_average, avg_bottom_right_lane, bottom_right_lane))
            avg_top_right_lane = tuple(map(weighted_average, avg_top_right_lane, top_right_lane))

        cv2.line(img, avg_top_right_lane, avg_bottom_right_lane, color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def calculate_roi(image):
    # should return a numpy array
    shape = image.shape
    x_horizon = 0.1 * shape[0]  # overly fitted
    y_horizon = 0.1 * shape[1]  # overly fitted
    top_left = (shape[1] / 2 + y_horizon, shape[0] / 2 + x_horizon)
    top_right = (shape[1] / 2 - y_horizon, shape[0] / 2 + x_horizon)
    bottom_right = (y_horizon, shape[0])
    bottom_left = (shape[1] - y_horizon, shape[0])
    return np.array([[top_right, top_left, bottom_left, bottom_right]], dtype=np.int32)

default_config = {
    'gaussian_kernel_size': 5,
    'canny_low_threshold': 50,
    'canny_high_threshold': 150,
    'hough_rho': 1,
    'hough_theta': math.pi / 180,
    'hough_threshold': 1,
    'hough_min_line_length': 40,
    'hough_max_line_gap': 20
}


def pipeline(image, config=default_config):
    gray_image = grayscale(image)

    blurred_image = gaussian_blur(gray_image, config['gaussian_kernel_size'])

    edges = canny(blurred_image, config['canny_low_threshold'], config['canny_high_threshold'])

    masked_image = region_of_interest(edges, calculate_roi(image))

    lines_detected = hough_lines(masked_image, config['hough_rho'], config['hough_theta'], config['hough_threshold'],
                                 config['hough_min_line_length'], config['hough_max_line_gap'])

    lanes_detected = weighted_img(lines_detected, image)

    return lanes_detected
