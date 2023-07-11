import cv2
import numpy as np
import os

image_files = os.listdir('/root/autodl-tmp/train/all_images/')
for image in image_files:
    # Read the TXT file
    image_data = cv2.imread('/root/autodl-tmp/train/all_images/' + image)
    annotation_file = os.path.splitext(image)[0] + ".txt"
    with open('/root/autodl-tmp/train/all_gts/'+annotation_file, 'r') as f:
        mask_info = f.read().splitlines()

    # Create a blank mask
    mask = np.zeros_like(image_data, dtype=np.uint8)

    # Parse the annotation
    for line in mask_info:
        line = line.strip().split(',')
        vertices = list(map(int, line))

        # Reshape vertices to get a list of (x, y) coordinates
        vertices = np.array(vertices).reshape(-1, 2)

        # Draw polygon on the mask
        cv2.fillPoly(mask, [vertices], (255, 255, 255))

    # Display or save the mask
    cv2.imwrite('/root/autodl-tmp/train/mask/'+image, mask)
