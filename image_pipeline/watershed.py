from skimage import data
from skimage.exposure import histogram

from PIL import Image, ImageChops
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import watershed, expand_labels
from skimage.color import label2rgb
from skimage import data

"""
Image processing pipeline class for comparison with the Twin-VAE.

Best result for phase-contrast: "subt: 1 ks: 5 ks_mult: 7, crp: 15, MAE: 1.66, DEV: 0.29 COR: 0.23"
Best result for bright-field: "subt: 4 ks: 3 ks_mult: 19, crp: 11, MAE: 2.39, DEV: 0.32 COR: 0.32"
"""

# Arguments
show_sample_histogram = False # Setermines whether a greyscale histogram of a preselected image is displayed
reference_image_id = 0 # Selects the nth image for the histogram showcase
show_processed_image = False # Determines whether the processed images with their regions are plotted
filepath = "../data/128p_pc/test.csv" #"../data/128p_bf/test.csv"
image_size = 128
MAE_THRESH = 2 # Threshold for notification of results better than this value
DEV_THRESH = 0.35 # Threshold for notification of results better than this value
COR_THRESH = 0.22 # Threshold for notification of results better than this value

# Data reading
data_info = pd.read_csv(filepath, header=None)
image_arr = np.asarray(data_info.iloc[:, 0]) # First column contains the image paths
label_arr = np.asarray(data_info.iloc[:, 1]) # Second column is the labels
data_len = len(data_info.index)
max_cells = 31
		
loaded_images = []
for image_path in image_arr:
	img_as_img = cv2.imread("../" + image_path[2:], cv2.IMREAD_GRAYSCALE)
	loaded_images.append(img_as_img)

# Show the histogram	
if show_sample_histogram:
	histogram, bin_edges = np.histogram(loaded_images[0], bins=256, range=(0, 256))
	plt.figure()
	plt.title("Grayscale Histogram")
	plt.xlabel("grayscale value")
	plt.ylabel("number of pixels")
	plt.plot(bin_edges[0:-1], histogram)
	plt.show()

# Grid search over all important parameters for the image processing pipeline
for ks in range(3,6,2): # Kernel sizes
	for ks_mult in range(5,11,1): # Amplitude of the kernel filter
		for subt in range (0,3): # Subtraction of 0 or 1 cell (to compensate for background region, if needed)
			for crp in range (15,16): # Cropping of image borders in a 128x128 image
				num_img = 0 # The counter for nonzero labelled images
				dist_sum = 0
				dev_sum = 0
				cor_sum = 0
				
				preds = np.zeros(data_len)
				lbls = np.zeros(data_len)
				bar_devs = np.zeros(max_cells)
				occurrences = np.zeros(max_cells)
				for idx in range(0,data_len):
					if label_arr[idx]==0:
						continue
						
					occurrences[label_arr[idx]] += 1

					gray = loaded_images[idx]
					crop = gray[crp:image_size-crp, crp:image_size-crp]
					kernel = np.ones((ks,ks),np.float32)/ks_mult
					dst = cv2.filter2D(crop,-1,kernel)

					_, thresh_img = cv2.threshold(dst,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
					
					# Adaptive thresholding for comparison, doesn't seem to improve results
					#thresh_img = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
					#cv2.imshow('image',thresh_img)
					#cv2.waitKey(0)

					edges = sobel(thresh_img)
					markers = np.zeros_like(thresh_img)
					foreground, background = 1, 2
					markers[thresh_img < 100] = background
					markers[thresh_img > 200] = foreground

					ws = watershed(edges, markers)
					seg1 = label(ws == foreground)

					histo, bin_edge = np.histogram(seg1, bins=30, range=(0, 30))
					count = np.count_nonzero(histo)
					count = count-subt # Subtracting the background when subt=1
					
					preds[idx] = count
					lbls[idx] = label_arr[idx]
					
					# Analysis of result
					num_img = num_img + 1
					dist = np.abs(np.subtract(count, label_arr[idx]))
					dist_sum = dist_sum + dist
					dev = 1 - np.divide(count, label_arr[idx])
					dev_sum = dev_sum + np.abs(dev)
					if round(count)==round(label_arr[idx]):
						cor_sum = cor_sum + 1

					# Show the segmentation
					if show_processed_image:
						expanded = expand_labels(seg1, distance=8)
						fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5),
												 sharex=True, sharey=True)

						color1 = label2rgb(seg1, image=thresh_img, bg_label=0)
						axes[0].imshow(color1)
						axes[0].set_title('Sobel+Watershed')

						color2 = label2rgb(expanded, image=thresh_img, bg_label=0)
						axes[1].imshow(color2)
						axes[1].set_title('Expanded labels')

						for a in axes:
							a.axis('off')
						fig.tight_layout()
						plt.show()
					
				# Determine actual results values by averaging over the number of images with nonzero labels.
				MAE = dist_sum / num_img
				DEV = dev_sum / num_img
				COR = cor_sum / num_img
				
				suffix = ""
				if COR > COR_THRESH or DEV < DEV_THRESH or MAE < MAE_THRESH:
					suffix = " performing well"
					#Saving predictions and labels to file
					for i in range(0, len(preds)):
						if label_arr[i] == 0:
							continue
						bar_devs[int(label_arr[i])] += np.abs(1 - (preds[i] / label_arr[i]))

					bar_devs = np.divide(bar_devs, occurrences)
					print(str(bar_devs))

					# Print the result for this parameter set to console
					print("subt: " + str(subt) + " ks: " + str(ks) + " ks_mult: " + str(ks_mult) + ", crp: " + str(crp) + ", MAE: " + str(round(MAE,2)) + ", DEV: " + str(round(DEV,2)) + " COR: " + str(round(COR,2)) + suffix)