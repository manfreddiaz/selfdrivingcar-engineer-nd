from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import matplotlib.image as mpimg

# Read in and grayscale the image
# Note: in the previous example we were reading a .jpg
# Here we read a .png and convert to 0,255 bytescale

for file in os.listdir("challenge/"):
    print file
    image = cv2.imread('challenge/' + file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dst = cv2.fastNlMeansDenoisingMulti(gray, 2, 5, None, 4, 7, 35)
    plt.imshow(gray, cmap='gray')
    plt.show()
    kernel_size = 15
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    plt.imshow(blur_gray, cmap='gray')
    plt.show()
    # Define our parameters for Canny and apply
    low_threshold = 40
    high_threshold = 100
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    plt.imshow(edges)
    plt.show()
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask

    imshape = image.shape
    x_horizon = 0.1 * imshape[0]
    y_horizon = 0.1 * imshape[1]
    print imshape
    #vertices = np.array([[ center_image, (0, imshape[0]), (imshape[1], imshape[0])]], dtype=np.int32)
    right = (imshape[1] / 2 + y_horizon / 2, imshape[0] / 2 + x_horizon)
    left = (imshape[1] / 2 - y_horizon / 2, imshape[0] / 2 + x_horizon)
    right_bottom = (y_horizon, imshape[0])
    left_bottom = (imshape[1] - y_horizon, imshape[0])
    vertices = np.array([[left , right, left_bottom, right_bottom]], dtype=np.int32)
    print vertices
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    #
    # plt.imshow(masked_edges)
    # plt.show()
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 1     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)



    # Iterate over the output "lines" and draw lines on a blank image
    left_points = []
    right_points = []
    left_slopes = []
    right_slopes = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > -0.9 and slope < -0.1:
                left_points.append((x1, y1))
                left_points.append((x2, y2))
                cv2.arrowedLine(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                left_slopes.append(slope)
            elif slope > 0.1 and slope < 0.9:
                right_points.append((x1, y1))
                right_points.append((x2, y2))
                cv2.arrowedLine(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
                right_slopes.append(slope)

     #np.sqrt((imshape[1] / 2) ** 2 + (imshape[0] / 2) ** 2) / 4

    (l_vx, l_vy, l_x0, l_y0) = cv2.fitLine(np.array(left_points), cv2.DIST_L2, 0, 0.01, 0.01)
    (r_vx, r_vy, r_x0, r_y0) = cv2.fitLine(np.array(right_points), cv2.DIST_L2, 0, 0.01, 0.01)
    #
    # l_diagonal_bottom_size = (left_bottom[0] - l_x0) / l_vx
    # l_diagonal_top_size = (left[0] - l_x0) / l_vx / 1.5
    # r_diagonal_bottom_size = (right_bottom[0] - r_x0) / r_vx
    # r_diagonal_top_size = (right[0] - r_x0) / r_vx / 1.5
    #
    # (l_x, l_y) = (-l_diagonal_bottom_size * l_vx + l_x0, -l_diagonal_bottom_size * l_vy + l_y0)
    # (l_x1, l_y1) = (l_diagonal_top_size * l_vx + l_x0, l_diagonal_top_size * l_vy + l_y0)
    # (r_x, r_y) = (-r_diagonal_bottom_size * r_vx + r_x0, -r_diagonal_bottom_size * r_vy + r_y0)
    # (r_x1, r_y1) = (r_diagonal_top_size * r_vx + r_x0, r_diagonal_top_size * r_vy + r_y0)

    # (l_vx, l_vy, l_x0, l_y0) = cv2.fitLine(np.array(left_points), cv2.DIST_L2, 0, 0.01, 0.01)
    # (r_vx, r_vy, r_x0, r_y0) = cv2.fitLine(np.array(right_points), cv2.DIST_L2, 0, 0.01, 0.01)
    #
    # l_diagonal_bottom_size = (left_bottom[0] - l_x0) / l_vx
    # l_diagonal_top_size = (left[0] - l_x0) / l_vx / 1.5
    # r_diagonal_bottom_size = (right_bottom[0] - r_x0) / r_vx
    # r_diagonal_top_size = (right[0] - r_x0) / r_vx / 1.5
    #

    left_max = max(x for x,y in left_points)
    right_min = min(x for x,y in right_points)

    (l_x, l_y) = ((imshape[0] - l_y0) * l_vx / l_vy + l_x0, imshape[0])
    (l_x1, l_y1) = (left_max, (left_max - l_x0) * l_vy / l_vx + l_y0)
    (r_x, r_y) = ((imshape[0] - r_y0) * r_vx / r_vy + r_x0, imshape[0])
    (r_x1, r_y1) = (right_min, (right_min - r_x0) * r_vy / r_vx + r_y0)

    # (r_x, r_y) = (-r_diagonal_bottom_size * r_vx + r_x0, -r_diagonal_bottom_size * r_vy + r_y0)
    # (r_x1, r_y1) = (r_diagonal_top_size * r_vx + r_x0, r_diagonal_top_size * r_vy + r_y0)
    #cv2.circle(line_image, (int(l_x), int(l_y)), 50, (0,0,255))
    #cv2.circle(line_image, (int(l_x1), int(l_y1)), 50, (0, 255, 0))
    #cv2.line(line_image, (int(l_x1), int(l_y1)), (int(l_x), int(l_y)), (255,0,0), 5)
    #cv2.line(line_image, (int(r_x1), int(r_y1)), (int(r_x), int(r_y)), (0,255,0), 5)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))
    # Draw the  lines on the edge image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    #cv2.fillConvexPoly(lines_edges, vertices, (0,0,255, 50))
    # plt.imshow(masked_edges)
    # plt.show()
    plt.imshow(lines_edges)
    plt.show()
