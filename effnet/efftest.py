from efficientnet_pytorch import EfficientNet

import os
from PIL import Image
import pandas as pd
import numpy as np
import random
import time
import warnings

import torch
from torch.utils.data.dataset import Dataset
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

"""
This file is used to transfer the state of the art EfficientNet to a regressional task.
Used for comparison with the Twin-VAE.
Instructions for the general code bits can be found in learner.py.
"""

warnings.filterwarnings('ignore')
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)

# Arguments
networkname = "efficientnet-b1"
test_interval = 100
save_interval = 1000
batch_size = 64
max_epochs = 105
learning_rate = 3e-5
max_cells = 31
dir = "b1-3e-5-long"
datadir = "/data/128p_pc/"

img_size = EfficientNet.get_image_size(networkname)

os.makedirs("../logs", exist_ok=True)
os.makedirs("../logs/" + dir, exist_ok=True)
train_mae_log = []
test_mae_log = []
train_dev_log = []
test_dev_log = []
train_cor_log = []
test_cor_log = []

class CustomDatasetFromImages(Dataset):
	def __init__(self, csv_path):
		# Transforms
		self.to_tensor = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomAffine(0, translate=(2 / img_size, 2 / img_size), scale=None, shear=None, resample=False, fillcolor=0),
			#transforms.RandomRotation(3),
			transforms.RandomResizedCrop(img_size, scale=(1.0,1.0), ratio=(1.0,1.0)),
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
			#img_as_img = img_as_img.convert('L') # Luminosity
			img_as_img = img_as_img.convert('RGB') # Luminosity
			#img = tfms(img).unsqueeze(0)	
			loaded_images.append(img_as_img)
		self._loaded_images = loaded_images

	def __getitem__(self, index):
		# image_name = self.image_arr[index] # Get image name from csv
		image_label = self.label_arr[index] # Get image label from csv
		img_as_img = self._loaded_images[index]
		angle = random.choice([0, 90]) # Don't need -90, because that is achieved in combination with the flips.
		img_as_img = TF.rotate(img_as_img, angle)
		img_as_tensor = self.to_tensor(img_as_img) # Already scaled to [0,1]
		return img_as_tensor, image_label

	def __len__(self):
		return self.data_len

prefix = ".."

num_workers = 0 # Use just 0 or 1 workers to prevent heavy memory overhead and slower loading
pin_memory = True

nat_train_set = CustomDatasetFromImages(prefix + datadir + 'train.csv')
nat_train_load = torch.utils.data.DataLoader(dataset=nat_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

nat_test_set = CustomDatasetFromImages(prefix + datadir + 'test_gen.csv')
nat_test_load = torch.utils.data.DataLoader(dataset=nat_test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

class EffNetReg(nn.Module):
	def __init__(self):
		super(EffNetReg, self).__init__()
		self.pretrain = EfficientNet.from_pretrained(networkname)
			
	def forward(self, x):
		embedding = self.pretrain.extract_features(x)
		embedding = embedding.view(embedding.size(0), -1)
		#print(str(embedding.size())) #Can be used to print required size for fc layer
		embedding = self.fc(embedding) # nn.Softplus()
		return embedding
		#self.regressor = nn.DataParallel(self.regressor)
		
# Classify with EfficientNet
#for model in models:
#	model.eval()
#with torch.no_grad():
#    logits = models[0](img)
#features = model.extract_features(img)
#preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()
fc_size = None
if networkname == "efficientnet-b0":
	fc_size = 62720
elif networkname == "efficientnet-b1":
	fc_size = 81920
elif networkname == "efficientnet-b2":
	fc_size = 114048
else: 
	fc_size = 153600

model = EffNetReg()
for param in model.parameters():
	param.requires_grad = False
model.fc = nn.Linear(fc_size,1)#declare here to automatically have requires_grad=true
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

def loss_function(prediction, label):
	regr_loss = F.mse_loss(prediction, label.view(-1, 1))
	return regr_loss

start_time = None

def model_train(epoch, loader):
	model.train()
	ccs = []
	labls = []
	num_unlabeled = 0
	for batch_idx, (data, labels) in enumerate(loader):
		
		data = data.cuda()
		labels = labels.float().cuda()
		
		optimizer.zero_grad()
		count = model(data)
		
		count[labels == 0] = 0 # Sets the counted cells to 0 for unlabeled data, so that regressor_loss=0
		num_unlabeled += (labels == 0).sum()
		loss = loss_function(count, labels)
		ccs.append(count.cpu().detach().numpy())
		labls.append(labels.cpu().detach().numpy())
		
		loss.backward()
		optimizer.step()
	preds = np.around(np.concatenate(ccs, axis=None)) # Elementwise round of cellcounts
	lbls = np.around(np.concatenate(labls, axis=None)) # Elementswise round of labels

	correct = np.sum(preds == lbls) # Count elementwise equality of predictions and labels
	len_set = len(loader.dataset)
	correct -= num_unlabeled  # Remove zero_indices from numerator
	correct = float(correct) / float(len_set - num_unlabeled) # Remove zero_indices from denominator

	dist_sum = np.sum(np.abs(np.subtract(preds, lbls))) # Elementwise addition of dist between preds and lbls
	len_labeled = float(len_set - num_unlabeled)
	MAE = dist_sum / len_labeled
	dev = np.ones(len_set) - np.divide(preds, lbls) # Deviation contains NaNs because syn data has lbl=0
	avg_dev = np.sum(np.abs(np.where(np.isnan(dev), 0, dev))) / len_labeled # Take the avg only of those deviations that weren't NaN

	train_mae_log.append(str(round(MAE,3)) + "\n")
	train_dev_log.append(str(round(avg_dev,3)) + "\n")
	train_cor_log.append(str(round(correct,3)) + "\n")
	print("epoch: " + str(epoch) + " MAE: " + str(round(MAE,3)) + " DEV: " + str(round(avg_dev,3)) + " COR: " + str(round(correct,3)) + " time: " + str(round(time.time() - start_time,1)) + "s")
	return correct, MAE, avg_dev
	
def model_test(epoch, loader):
	model.eval()
	ccs = []
	labls = []
	num_unlabeled = 0
	
	bar_devs = np.zeros(max_cells)
	occurrences = np.zeros(max_cells)
	for batch_idx, (data, labels) in enumerate(loader):
		data = data.cuda()
		
		for l in labels:
			occurrences[l] += 1
			
		labels = labels.float().cuda()
		
		count = model(data)

		count[labels == 0] = 0 # Sets the counted cells to 0 for unlabeled data, so that regressor_loss=0
		num_unlabeled += (labels == 0).sum()
		loss = loss_function(count, labels)

		ccs.append(count.cpu().detach().numpy())
		labls.append(labels.cpu().detach().numpy())

	preds = np.around(np.concatenate(ccs, axis=None)) # Elementwise round of cellcounts
	lbls = np.around(np.concatenate(labls, axis=None)) # Elementswise round of labels

	correct = np.sum(preds == lbls) # Count elementwise equality of predictions and labels
	len_set = len(loader.dataset)
	correct -= num_unlabeled  # Remove zero_indices from numerator
	correct = float(correct) / float(len_set - num_unlabeled) # Remove zero_indices from denominator

	dist_sum = np.sum(np.abs(np.subtract(preds, lbls))) # Elementwise addition of dist between preds and lbls
	len_labeled = float(len_set - num_unlabeled)
	MAE = dist_sum / len_labeled
	dev = np.ones(len_set) - np.divide(preds, lbls) # Deviation contains NaNs because syn data has lbl=0
	avg_dev = np.sum(np.abs(np.where(np.isnan(dev), 0, dev))) / len_labeled # Take the avg only of those deviations that weren't NaN

	test_mae_log.append(str(round(MAE,3)) + "\n")
	test_dev_log.append(str(round(avg_dev,3)) + "\n")
	test_cor_log.append(str(round(correct,3)) + "\n")
	
	#Saving predictions and labels to file
	for i in range(0, len(preds)):
		if lbls[i] == 0:
			continue
		bar_devs[int(lbls[i])] += np.abs(1 - (preds[i] / lbls[i]))

	bar_devs = np.divide(bar_devs, occurrences)
	print(str(bar_devs))

	print("test: epoch: " + str(epoch) + " MAE: " + str(round(MAE,3)) + " DEV: " + str(round(avg_dev,3)) + " COR: " + str(round(correct,3)) + " time: " + str(round(time.time() - start_time,1)) + "s")
	return correct, MAE, avg_dev


#Training
for epoch in range(1, max_epochs+1):
	start_time = time.time()
	model_train(epoch, nat_train_load)
	
	if epoch%test_interval==0:
		#Testing
		start_time = time.time()
		model_test(epoch, nat_test_load)
		
	if epoch%save_interval==0:
		torch.save({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
			}, "../logs/" + dir + '/%d.pth' % epoch)
			
		train_mae_logfile = open("../logs/" + dir + "/train_mae_log.txt", "w")
		test_mae_logfile = open("../logs/" + dir + "/test_mae_log.txt", "w")
		train_dev_logfile = open("../logs/" + dir + "/train_dev_log.txt", "w")
		test_dev_logfile = open("../logs/" + dir + "/test_dev_log.txt", "w")
		train_cor_logfile = open("../logs/" + dir + "/train_cor_log.txt", "w")
		test_cor_logfile = open("../logs/" + dir + "/test_cor_log.txt", "w")
		
		for line in train_mae_log:
			train_mae_logfile.write(line)
		for line in train_dev_log:
			train_dev_logfile.write(line)
		for line in train_cor_log:
			train_cor_logfile.write(line)	
		for line in test_mae_log:
			test_mae_logfile.write(line)
		for line in test_dev_log:
			test_dev_logfile.write(line)
		for line in test_cor_log:
			test_cor_logfile.write(line)
			
		train_mae_logfile.close()
		test_mae_logfile.close()
		train_dev_logfile.close()
		test_dev_logfile.close()
		train_cor_logfile.close()
		test_cor_logfile.close()