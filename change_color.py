import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

max_rel_error = 0.2
max_prediction = 60.0
min_prediction = 1.0
# Step 1: Read the PNG file

# list_man = ['rank_0_54.png',
#             'rank_0_64.png',
#             'rank_0_84.png',
#             'rank_0_112.png',
#             'rank_0_113.png',
#             'rank_0_120.png',
#             'rank_0_121.png',
#             'rank_1_106.png',
#             'rank_1_121.png',
#             'rank_2_54.png',
#             'rank_2_121.png',
#             'rank_3_120.png'
#             ]
# list_outputs = os.listdir('output/waymo/figure/waymo_costdcnet_bn/output_depth/')


# for name in list_man:
    
#     image = cv2.imread('output/waymo/figure/waymo_pretrained-costdc/output_depth/'+name, cv2.IMREAD_UNCHANGED)
#     image_gt = cv2.imread('output/waymo/figure/waymo_pretrained-costdc/ground_truth/'+name, cv2.IMREAD_UNCHANGED)
#     # Scaling and clipping (min and max)
#     image = image / 256
#     image[image > max_prediction] = max_prediction
#     image[image < min_prediction] = min_prediction

#     # Scaling and clipping (max, min is already 0)
#     image_gt = image_gt / 256
#     image_gt[image_gt > max_prediction] = max_prediction

#     errormap = abs(image - image_gt) / image_gt
#     errormap[image_gt == 0.0] = 0.0
#     errormap[errormap > max_rel_error] = max_rel_error

#     # Step 2: Apply the 'jet' colormap to non-zero pixels
#     colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(errormap/max_rel_error, alpha=255), cv2.COLORMAP_HOT)

#     # Step 3: Set background pixels (0 value) to black
#     colormap_image[image == 0] = [0, 0, 0]

#     # Step 4: Save the image
#     cv2.imwrite('figure_generated/errormap_pretrained/'+name, colormap_image)

#     # Optionally, display the image
#     plt.imshow(cv2.cvtColor(colormap_image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

# # output_depth prediction

# for name in list_man:
#     image = cv2.imread('output/waymo/figure/waymo_pretrained-costdc/output_depth/'+name, cv2.IMREAD_UNCHANGED)
#     image = image / 256
#     image[image > max_prediction] = max_prediction
#     image[image < min_prediction] = min_prediction
#     image /= max_prediction

#     # Step 2: Apply the 'jet' colormap to non-zero pixels
#     colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=255), cv2.COLORMAP_JET)

#     # Step 3: Set background pixels (0 value) to black
#     colormap_image[image == 0] = [0, 0, 0]

#     # Step 4: Save the image
#     cv2.imwrite('figure_generated/prediction_pretrained_colored/'+name, colormap_image)

#     # Optionally, display the image
#     plt.imshow(cv2.cvtColor(colormap_image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()


# Step 1: Read the PNG file
# adapt_path = 'waymo_costdcnet_adapt'
# # bn_path = 'waymo_costdcnet_bn'
# bn_path = 'waymo_pretrained-costdc'

# adapt_output_paths = sorted(glob.glob(
#     os.path.join('output/waymo/figure', adapt_path, 'output_depth', '*')))
# bn_output_paths = sorted(glob.glob(
#     os.path.join('output/waymo/figure', bn_path, 'output_depth', '*')))

# adapt_gt_paths = sorted(glob.glob(
#     os.path.join('output/waymo/figure', adapt_path, 'ground_truth', '*')))
# bn_gt_paths = sorted(glob.glob(
#     os.path.join('output/waymo/figure', bn_path, 'ground_truth', '*')))


# for our_output_path, baseline_output_path, out_gt_path in zip(adapt_output_paths, bn_output_paths, adapt_gt_paths):
#     ours_name = our_output_path.split('/')[-1]
#     baselines_name = baseline_output_path.split('/')[-1]
#     assert ours_name == baselines_name
#     name = ours_name
#     # load gt and scale it
#     image_gt = cv2.imread(out_gt_path, cv2.IMREAD_UNCHANGED)
#     image_gt = image_gt / 256
#     image_gt[image_gt > max_prediction] = max_prediction

#     num_valid = np.sum(image_gt > 0)

#     # load our output and scale it
#     output_ours = cv2.imread(our_output_path, cv2.IMREAD_UNCHANGED)
#     output_ours = output_ours / 256
#     output_ours[output_ours > max_prediction] = max_prediction
#     output_ours[output_ours < min_prediction] = min_prediction

#     # load baseline output and scale it
#     output_baseline = cv2.imread(baseline_output_path, cv2.IMREAD_UNCHANGED)
#     output_baseline = output_baseline / 256
#     output_baseline[output_baseline > max_prediction] = max_prediction
#     output_baseline[output_baseline < min_prediction] = min_prediction

#     # Generate errormap
#     errormap_ours = abs(output_ours - image_gt) / abs(image_gt)
#     errormap_ours[image_gt == 0.0] = 0.0
#     errormap_ours[errormap_ours > max_rel_error] = max_rel_error
#     # Apply the jet colormap to non-zero pixels
#     errormap_ours_colored = cv2.applyColorMap(cv2.convertScaleAbs(
#         errormap_ours/max_rel_error, alpha=255), cv2.COLORMAP_HOT)
#     # Step 3: Set background pixels (0 value) to black
#     errormap_ours_colored[output_ours == 0] = [0, 0, 0]
#     # Step 4: Save the image
#     out_path = 'figure_generated3/ours_errormap_colored/' + name
#     cv2.imwrite(out_path, errormap_ours_colored)

#     # Optionally, display the image
#     # plt.imshow(cv2.cvtColor(errormap_ours_colored, cv2.COLOR_BGR2RGB))
#     # plt.axis('off')
#     # plt.show()

#     # Generate errormap
#     errormap_baseline = abs(output_ours - image_gt) / abs(image_gt)
#     errormap_baseline[image_gt == 0.0] = 0.0
#     errormap_baseline[errormap_baseline > max_rel_error] = max_rel_error
#     # Apply the jet colormap to non-zero pixels
#     errormap_baseline_colored = cv2.applyColorMap(cv2.convertScaleAbs(
#         errormap_baseline/max_rel_error, alpha=255), cv2.COLORMAP_HOT)
#     # Step 3: Set background pixels (0 value) to black
#     errormap_baseline_colored[output_ours == 0] = [0, 0, 0]
#     # Step 4: Save the image
#     out_path = 'figure_generated3/pretrained_errormap_colored/' + name
#     cv2.imwrite(out_path, errormap_baseline_colored)

#     diff = errormap_ours - errormap_baseline
#     points_excel = diff < 0
#     avg_rel_error = np.sum((diff / errormap_ours)[errormap_ours > 0.0])

#     # Optionally, display the image
#     # plt.imshow(cv2.cvtColor(errormap_baseline_colored, cv2.COLOR_BGR2RGB))
#     # plt.axis('off')
#     # plt.show()

list_outputs = os.listdir('visualizations/voiced-kitti_waymo-finetuned/outputs/ground_truth/')

# for name in list_man:
#     image = cv2.imread('output/waymo/figure/waymo_pretrained-costdc/output_depth/'+name, cv2.IMREAD_UNCHANGED)
#     image_gt = cv2.imread('output/waymo/figure/waymo_pretrained-costdc/ground_truth/'+name, cv2.IMREAD_UNCHANGED)
#     image = image / 256
#     image[image > max_prediction] = max_prediction
#     image[image < min_prediction] = min_prediction

#     image_gt = image_gt / 256
#     image_gt[image_gt > max_prediction] = max_prediction

#     errormap = abs(image - image_gt) / abs(image_gt)
#     errormap[image_gt == 0.0] = 0.0
#     errormap[errormap > max_rel_error] = max_rel_error
#     # Step 2: Apply the 'jet' colormap to non-zero pixels
#     colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(
#         errormap/max_rel_error, alpha=255), cv2.COLORMAP_HOT)

#     # Step 3: Set background pixels (0 value) to black
#     colormap_image[image == 0] = [0, 0, 0]

#     # Step 4: Save the image
#     cv2.imwrite('figure_generated3/pretrained_errormap_colored/'+name, colormap_image)

#     # Optionally, display the image
#     plt.imshow(cv2.cvtColor(colormap_image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()
    
    

# for name in list_man:
#     image = cv2.imread('output/waymo/figure/waymo_costdcnet_adapt/output_depth/'+name, cv2.IMREAD_UNCHANGED)
#     image = image / 256
#     image[image > max_prediction] = max_prediction
#     image[image < min_prediction] = min_prediction
#     image /= max_prediction

#     # Step 2: Apply the 'jet' colormap to non-zero pixels
#     colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=255), cv2.COLORMAP_JET)

#     # Step 3: Set background pixels (0 value) to black
#     colormap_image[image == 0] = [0, 0, 0]

#     # Step 4: Save the image
#     cv2.imwrite('figure_generated3/adapt_prediction_colored/'+name, colormap_image)

#     # Optionally, display the image
#     plt.imshow(cv2.cvtColor(colormap_image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()


for name in list_outputs:
    image = cv2.imread('visualizations/voiced-kitti_waymo-finetuned/outputs/ground_truth/'+name, cv2.IMREAD_UNCHANGED)
    image = image / 256
    image[image > max_prediction] = max_prediction
    image /= max_prediction

    # Step 2: Apply the 'jet' colormap to non-zero pixels
    colormap_image = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=255), cv2.COLORMAP_JET)

    # Step 3: Set background pixels (0 value) to black
    colormap_image[image == 0] = [0, 0, 0]

    # Step 4: Save the image
    cv2.imwrite('visualizations/voiced-kitti_waymo-finetuned/outputs/ground_truth_colored/'+name, colormap_image)
    print(name)

    # Optionally, display the image
    plt.imshow(cv2.cvtColor(colormap_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Step 1: Read the PNG file