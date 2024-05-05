import cv2
from IPython.display import Image, display
from info import MPII_INFO
import json
import numpy as np


dataset_info = MPII_INFO()
# dataset_info['skeleton_info']
kps_name2index = {dataset_info['keypoint_info'][key]['name']:key for key in dataset_info['keypoint_info'] }


# VISUALIZE n-th image from gt_json
def visualize_with_keypoints(nth_image, annot_truth, keypoints):
  ROOT_FOLDER = "/datagrid/personal/baljibil"
  
  index = nth_image
  im_path = annot_truth[index]['image']
  # Load the image
  image = cv2.imread(ROOT_FOLDER + '/data/MPII_COCO/images/' + im_path)

  # Load annotation keypoints from JSON files


  # Draw keypoints on the image
  for kp in keypoints:
      x, y = int(kp[0]), int(kp[1])
      cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
  for id,val in dataset_info['skeleton_info'].items():
      # if int(keypoints1[kps_name2index[val['link'][0]]][0])  == -1:
      #   continue
      # if int(keypoints1[kps_name2index[val['link'][1]]][0])  == -1:
      #   continue
      start_point = (int(keypoints[kps_name2index[val['link'][0]]][0] )), int(keypoints[kps_name2index[val['link'][0]]][1])
      end_point = (int(keypoints[kps_name2index[val['link'][1]]][0] )), int(keypoints[kps_name2index[val['link'][1]]][1])
      cv2.line(image, start_point, end_point, val['color'], 2)  # Yellow color for lines
  # Resize the image if necessary
  # This step is optional, depending on your images

  # Split the image horizontally
#   image1 = image[:, :image.shape[1]//2]
#   image2 = image[:, image.shape[1]//2:]

  # Display the annotated images side by side
  cv2.imwrite(f'images/tm_results/image{index}.jpg', image)

  display(Image(filename=f'images/tm_results/image{index}.jpg'))


def save_with_keypoints(nth_image, annot_truth, path, keypoints, skeleton=None, bounding_box=None):
  ROOT_FOLDER = "/datagrid/personal/baljibil"
  
  index = nth_image
  im_path = annot_truth[index]['image']
  print(im_path)
  # Load the image
  image = cv2.imread(ROOT_FOLDER + '/data/MPII_COCO/images/' + im_path)

  # Load annotation keypoints from JSON files
  # pred

  # Draw keypoints on the image
  for kp in keypoints:
      x, y = int(kp[0]), int(kp[1])
      cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

  if skeleton != None:
    for id,val in dataset_info['skeleton_info'].items():
        # if int(keypoints1[kps_name2index[val['link'][0]]][0])  == -1:
        #   continue
        # if int(keypoints1[kps_name2index[val['link'][1]]][0])  == -1:
        #   continue
        start_point = (int(keypoints[kps_name2index[val['link'][0]]][0] )), int(keypoints[kps_name2index[val['link'][0]]][1])
        end_point = (int(keypoints[kps_name2index[val['link'][1]]][0] )), int(keypoints[kps_name2index[val['link'][1]]][1])
        cv2.line(image, start_point, end_point, val['color'], 2)  # Yellow color for lines
  # Resize the image if necessary
  # This step is optional, depending on your images

  if bounding_box != None:
    center, scale = bounding_box
    scale_px = 200 * 1.25
    bbox_left = (int(center[0] - (scale_px * scale) / 2), int(center[1] - (scale_px * scale) / 2))
    bbox_right = (int(center[0] + (scale_px * scale) / 2), int(center[1] + (scale_px * scale) / 2))
    print("bbleft:", bbox_left)
    print("bbright:", bbox_right)
    cv2.rectangle(image, bbox_left, bbox_right, (0,255,255), 2)

  # Display the annotated images side by side
  cv2.imwrite(f'{path}', image)


def save_with_multiple_keypoints(nth_image, annot_truth, path, keypoints, skeleton=None, bounding_box=None, keypoints1=None, keypoints2 = None, label1=None,
                                 label2=None):
  # if given multiple keypoints, draw keypoint 1 on first and all others on other.
  ROOT_FOLDER = "/datagrid/personal/baljibil"
  
  index = nth_image
  im_path = annot_truth[index]['image']
  print(im_path)
  # Load the image
  image = cv2.imread(ROOT_FOLDER + '/data/MPII_COCO/images/' + im_path)
  image1 = image.copy()
  # Load annotation keypoints from JSON files
  # pred

  # Draw keypoints on the image
  for kp in keypoints:
      x, y = int(kp[0]), int(kp[1])
      cv2.circle(image, (x, y), 3, (0, 255, 0, 150), -1)
  if keypoints1 is not None:
    for kp in keypoints1:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image1, (x, y), 3, (0, 255, 0, 100), -1)
  if keypoints2 is not None:
    for kp in keypoints2:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(image1, (x, y), 3, (0, 0, 255, 100), -1)

  if skeleton != None:
    for id,val in dataset_info['skeleton_info'].items():
        # if int(keypoints1[kps_name2index[val['link'][0]]][0])  == -1:
        #   continue
        # if int(keypoints1[kps_name2index[val['link'][1]]][0])  == -1:
        #   continue
        start_point = (int(keypoints[kps_name2index[val['link'][0]]][0] )), int(keypoints[kps_name2index[val['link'][0]]][1])
        end_point = (int(keypoints[kps_name2index[val['link'][1]]][0] )), int(keypoints[kps_name2index[val['link'][1]]][1])
        cv2.line(image, start_point, end_point, val['color'], 2)  # Yellow color for lines
  # Resize the image if necessary
  # This step is optional, depending on your images

  if bounding_box != None:
    center, scale = bounding_box
    scale_px = 200 * 1.25
    bbox_left = (int(center[0] - (scale_px * scale) / 2), int(center[1] - (scale_px * scale) / 2))
    bbox_right = (int(center[0] + (scale_px * scale) / 2), int(center[1] + (scale_px * scale) / 2))
    print("bbleft:", bbox_left)
    print("bbright:", bbox_right)
    cv2.rectangle(image, bbox_left, bbox_right, (0,255,255), 2)

  
    # Add labels on top of the images
  if label1 and label2 is not None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)
    label1_size = cv2.getTextSize(label1, font, font_scale, thickness)[0]
    label2_size = cv2.getTextSize(label2, font, font_scale, thickness)[0]
    label1_position = ((image.shape[1] - label1_size[0]) // 2, label1_size[1] + 5)
    label2_position = ((image.shape[1] + image1.shape[1] - label2_size[0]) // 2, label2_size[1] + 5)
    cv2.putText(image, label1, label1_position, font, font_scale, color, thickness)
    cv2.putText(image1, label2, label2_position, font, font_scale, color, thickness)
  # Display the annotated images side by side
  combined_image = image
  if keypoints2 is not None:
    combined_image = np.concatenate((image, image1), axis=1)
  cv2.imwrite(f'{path}', combined_image)
