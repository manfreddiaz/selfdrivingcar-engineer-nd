#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import util as u
import os
from moviepy.editor import VideoFileClip

# for file_name in os.listdir("test_images/"):
#
#     #reading in an image
#     image = cv2.imread('test_images/' + file_name)
#     print(file_name)
#
#     plt.imshow(u.pipeline(image))
#     plt.show()
# for file in os.listdir('.'):
#     if file.endswith('.mp4'):

import os
for file in os.listdir("test_images/"):
    image = cv2.imread('test_images/' + file)
    marked_image = u.pipeline(image)
    cv2.imwrite('test_images/' + 'marked-' + file, marked_image)

# clip = VideoFileClip('challenge.mp4')
# white_clip = clip.fl_image(u.pipeline)
# white_clip.write_videofile('marked/marked-' + 'challenge.mp4', audio=False)