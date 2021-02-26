import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
from PIL import Image, ImageChops
import pandas as pd
import numpy as np
import random

from learner import batch_size, img_size, seed, syn_train_name, syn_test_name, img_noise

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

all_images = []

class CustomDatasetFromImages(Dataset):
	def __init__(self, csv_path):
		# Transforms
		self.to_tensor = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			#transforms.RandomAffine(0, translate=(2 / img_size, 2 / img_size), scale=None, shear=None, resample=False, fillcolor=0),
			#transforms.RandomRotation(3),
			transforms.RandomResizedCrop(img_size, scale=(0.9,0.9), ratio=(1.0,1.0)),
			transforms.ToTensor(), ])
		# transforms.Normalize((0.5,), (0.5,))])

		self.data_info = pd.read_csv(csv_path, header=None)
		# First column contains the image paths
		self.image_arr = np.asarray(self.data_info.iloc[:, 0])
		# Second column is the labels
		self.label_arr = np.asarray(self.data_info.iloc[:, 1])
		
		self.data_len = len(self.data_info.index)

		loaded_images = []
		for image_path in self.image_arr:
			img_as_img = Image.open(prefix + image_path[2:])
			img_as_img.load()
			img_as_img = img_as_img.convert('L') # Luminosity
			loaded_images.append(img_as_img)
		self._loaded_images = loaded_images

	def __getitem__(self, index):
		# image_name = self.image_arr[index] # Get image name from csv
		image_label = self.label_arr[index] # Get image label from csv

		img_as_img = self._loaded_images[index]

		angle = random.choice([0, 90]) # Don't need -90, because that is achieved in combination with the flips.
		img_as_img = TF.rotate(img_as_img, angle)

		img_as_tensor = self.to_tensor(img_as_img) # Already scaled to [0,1]
		noise_map = torch.autograd.Variable(torch.randn(img_size, img_size) * img_noise)
		img_as_tensor = img_as_tensor.add(noise_map)

		return img_as_tensor, image_label

	def __len__(self):
		return self.data_len

prefix = os.getcwd().replace("\\", "/")[:-4]  # gets the current path up to /vae and removes the /vae to get to the data directory

num_workers = 0 # Use just 0 or 1 workers to prevent heavy memory overhead and slower loading
pin_memory = True

#syn_train_set = CustomDatasetFromImages(prefix + '/data/128p/'+syn_train_name+'.csv') # train_gen
syn_train_set = CustomDatasetFromImages(prefix + '/data/128p_pc/'+syn_train_name+'.csv') # train_gen
syn_train_load = torch.utils.data.DataLoader(dataset=syn_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

#syn_test_set = CustomDatasetFromImages(prefix + '/data/128p/'+syn_test_name+'.csv') # test_gen
syn_test_set = CustomDatasetFromImages(prefix + '/data/128p_pc/'+syn_test_name+'.csv') # test_gen
syn_test_load = torch.utils.data.DataLoader(dataset=syn_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

#nat_train_set = CustomDatasetFromImages(prefix + '/data/128p/train.csv')
nat_train_set = CustomDatasetFromImages(prefix + '/data/128p_pc/train.csv')
nat_train_load = torch.utils.data.DataLoader(dataset=nat_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

#nat_test_set = CustomDatasetFromImages(prefix + '/data/128p/test.csv')
nat_test_set = CustomDatasetFromImages(prefix + '/data/128p_pc/test.csv')
nat_test_load = torch.utils.data.DataLoader(dataset=nat_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
