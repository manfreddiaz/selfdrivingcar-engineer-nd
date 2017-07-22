import cv2
import numpy as np
import glob

pattern_size = (8, 6)
real_world_grid = np.zeros((6 * 8, 3), np.float32)
real_world_grid[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

real_world_points = [] # 3d points in real world space
transformed_points = []  # 2d points in image plane.


calibration_images = glob.glob("calibration_wide/GO*.jpg")

for idx, image_file in enumerate(calibration_images):
    chessboard_image = cv2.imread(image_file, flags=cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(chessboard_image, pattern_size, None)
    if found:
        real_world_points.append(real_world_grid)
        transformed_points.append(corners)

test_image = cv2.imread("calibration_wide/test_image-2.jpg")
cv2.imshow("distorted", test_image)
test_image_size = (test_image.shape[1], test_image.shape[0])

calibrated, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = \
    cv2.calibrateCamera(real_world_points, transformed_points, test_image_size, cameraMatrix=None, distCoeffs=None)

print("Camera Matrix \n\t[fx  0  cx] \n\t[0  fy  cy] \n\t[0   0   1]")
print("(cx, cy) => principal point (center)")
print("(fx, fy) => focal length in pixels\n")
print(camera_matrix, "\n")

print("Distortion Coefficients: \n\t[k1, k2, p1, p2, [k3, k4, k5, k6]] where: ")
print("\tk1, k2, ... => radial distortion coefficients.")
print("\tp1, p2 => tangential distortion coefficients\n")
print(distortion_coefficients, "\n")

print(rotation_vectors)
print(translation_vectors)

undistorted = cv2.undistort(test_image, camera_matrix, distortion_coefficients, newCameraMatrix=camera_matrix)
cv2.imshow("undistorted", undistorted)

# warping
undistorted_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
found, corners = cv2.findChessboardCorners(undistorted_gray, pattern_size, None)
test_image_size = (undistorted_gray.shape[1], undistorted_gray.shape[0])
offset = 50

if found:
    cv2.drawChessboardCorners(undistorted_gray, pattern_size, corners, found)
    cv2.imshow("croners", undistorted_gray)
    source_points = np.float32([corners[0], corners[7], corners[-1], corners[-8]])
    dest_points = np.float32([[offset, offset], [test_image_size[0] - offset, offset],
                [test_image_size[0] - offset, test_image_size[1] - offset],
                [offset, test_image_size[1] - offset]])

    transform_matrix = cv2.getPerspectiveTransform(source_points, dest_points)
    warped_image = cv2.warpPerspective(undistorted_gray, transform_matrix, test_image_size)
    cv2.imshow("warped", warped_image)
    cv2.waitKey()