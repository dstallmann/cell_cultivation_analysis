from __future__ import print_function
import os
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
import time
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
#import umap
import warnings
import numpy as np
import math
import shutil
import sys
from torch import nn

import VAE
from radam import RAdam
import plotlyplot

import plotly
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

try:
	import winsound
except ImportError:
	pass

warnings.filterwarnings('ignore')
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)

# Optional: Deterministic GPU behaviour for reproducability
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# region Variables
# Loaded parameters
batch_size = None
learning_rate = None
bottleneck = None
decoder_l_factor = None
regressor_l_factor = None
KLD_l_factor = None
directory = ""
seed = None
syn_train_name = None
syn_test_name = None
img_noise = None

# Loaded from dataloader
syn_train_loader = None
syn_test_loader = None
nat_train_loader = None
nat_test_loader = None

epoch = 1
regressor_start = 0
img_size = 128
boundary_dim = img_size * img_size
model = None
optimizer = None
decoder_nat_loss = 0
decoder_syn_loss = 0
KLD_nat_loss = 0
KLD_syn_loss = 0
regressor_nat = 0
regressor_syn = 0

decoder_nat_log = []
decoder_syn_log = []
KLD_nat_log = []
KLD_syn_log = []
regressor_nat_log = []
regressor_syn_log = []
correct_nat_log = []
correct_syn_log = []

distance_sum = 0

eval_interval = 1000
# endregion

# region Utility functions
def set_seed(seed):
	"""
	Sets the seed at all seed-sensitive modules
	@param seed: int; the seed to allow reproducibility
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	
def reset_loss_sums():
	"""
	Sets all summed losses to 0 for the next iteration
	"""
	global decoder_nat_loss, decoder_syn_loss, KLD_syn_loss, KLD_nat_loss, regressor_nat, regressor_syn
	decoder_nat_loss = decoder_syn_loss = KLD_syn_loss = KLD_nat_loss = regressor_nat = regressor_syn = 0

def save_log(log, title):
	"""
	Saves the log files to the "logs" subfolder
	@param log: string; contains the logged data
	@param title: string; prefix name of the file
	"""
	logfile = open(directory + "/logs/" + title + "_log.txt", "w")
	for line in log:
		logfile.write(str(line) + "\n")
	logfile.close()

def sample_representation(mode, data, noise):
	"""
	As part of the evaluation, it samples random images from the representational space and creates (hopefully) similar images, by sampling again nearby.
	@param mode: string, differentiates between types of wanted images
	@param data: contains a data batch from the synthetic-test dataloader
	@param noise: contains a noise batch generated in evaluate()
	"""
	if mode == "orig_syn":
		model.mode = "synth"
		z = model.sample_start(data)
	elif mode == "orig_nat":
		model.mode = "natural"
		z = model.sample_start(data)
	else:
		model.mode = mode
		z = model.sample_start(data).add(noise)
	sample, cc = model.sample_end(z)
	img = sample.view(batch_size, 1, img_size, img_size)
	save_image(img.cpu(), directory + "/" + "random_" + mode + "_" + str(epoch) + ".png")
	sample_txt = open(directory + "/random_" + mode + "_cc.txt", "a")
	sample_txt.write(str(cc))
	sample_txt.close()

def load_log(filename):
	"""
	Loads a logfile for continuation
	@param filename: string; the name of the file to be loaded
	@return: string; the loaded logfile
	"""
	logfile = open(directory + filename, "r")
	log = logfile.read().splitlines()
	log = [float(i) for i in log]
	logfile.close()
	return log

def playSound():
	"""
	Plays a sound when called (differentiates between Unix and Windows systems). Uncomment to use.
	"""
	if os.name == "posix":
		duration = 0.5  # seconds
		freq = 80  # Hz
		#os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
	elif os.name == "nt":
		duration = 500  # milliseconds
		freq = 80  # Hz
		#winsound.Beep(freq, duration)

# endregion

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, cl, target, natural):
	"""
	Universally calculates the loss, be it for training or testing. Hardcoded to use mse_loss. Change below to binary_cross_entropy if desired.
	@param recon_x: images reconstructed by the decoder(s)
	@param x: original images for comparison
	@param mu: latent mean
	@param logvar: latent log variance
	@param cl: cell count predictions for given images
	@param target: cell count label for given labeled images
	@param natural: bool, true if x is of type natural
	@return: float, float, float, the summed loss as well as the Kullback-Leibler divergence and the loss of the regressor in separate
	"""
	global decoder_nat_loss, decoder_syn_loss, KLD_syn_loss, KLD_nat_loss, regressor_nat, regressor_syn
	#decoder_loss = F.binary_cross_entropy(recon_x, x.view(-1, 1, img_size, img_size), reduction='sum') * decoder_l_factor
	decoder_loss = F.mse_loss(recon_x, x) * decoder_l_factor

	# see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	beta = 1 / (batch_size * boundary_dim)  # disentanglement factor#extremely small
	KLD = -0.5 * beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * KLD_l_factor

	regressor_loss = F.mse_loss(cl, target.view(-1, 1)) * regressor_l_factor
	if epoch < regressor_start:
		regressor_loss *= 0

	if natural:
		decoder_nat_loss += decoder_loss
		KLD_nat_loss += KLD
		regressor_nat += regressor_loss
	else:
		decoder_syn_loss += decoder_loss
		KLD_syn_loss += KLD
		regressor_syn += regressor_loss

	if KLD > 1e10:
		playSound()
		sys.exit('KLD diverged')

	return decoder_loss + KLD + regressor_loss, KLD, regressor_loss

def model_train(loader, natural):
	"""
	Main training function, called by learn(), calls the VAE, handles the loaded data, does the backpropagation etc.
	@param loader: the dataloader that contains a numpy-representation of the images and their labels
	@param natural: bool, true if the data in loader is of type natural
	@return: proportion of correctly counted cells (correct), as well as the mean error and the mean deviation, averaged over all the data contained in loader
	"""
	model.train()
	train_loss = mse_loss = KLD_loss = num_unlabeled = 0
	ccs = []
	labls = []
	for batch_idx, (data, labels) in enumerate(loader):
		data = data.cuda()
		labels = labels.float().cuda()

		optimizer.zero_grad()
		model.mode = 'natural' if natural else 'synth'
		recon_batch, mu, logvar, cc = model(data)

		cc[labels == 0] = 0 # Sets the counted cells to 0 for unlabeled data, so that regressor_loss=0
		num_unlabeled += (labels == 0).sum()
		loss, KLD, MSE = loss_function(recon_batch, data, mu, logvar, cc, labels, natural)

		ccs.append(cc.cpu().detach().numpy())
		labls.append(labels.cpu().detach().numpy())

		loss.backward()
		train_loss += loss.item()
		mse_loss += MSE
		KLD_loss += KLD
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

	if natural:
		decoder_nat_log.append((train_loss - mse_loss).item() / len_set) # =Decoder loss + KLD
		KLD_nat_log.append(KLD_loss.item() / len_set)
		regressor_nat_log.append(mse_loss.item() / len_set)
		correct_nat_log.append(correct)
	else:
		decoder_syn_log.append((train_loss - mse_loss).item() / len_set) # =Decoder loss + KLD
		KLD_syn_log.append(KLD_loss.item() / len_set)
		regressor_syn_log.append(mse_loss.item() / len_set)
		correct_syn_log.append(correct)
	return correct, MAE, avg_dev

def model_test(epo, natural):
	"""
	Main test function, called by learn(), calls the VAE, handles the loaded data, does the backpropagation etc.
	@param epo: string, epoch, just for logging purposes
	@param natural: true if the the natural pipeline should be tested
	@return: proportion of correctly counted cells (correct), as well as the mean error, averaged over all the data contained in loader
	"""
	model.eval()
	with torch.no_grad():
		n = batch_size

		if natural:
			loader = nat_test_loader
			prefix = "nat"
		else:
			loader = syn_test_loader
			prefix = "syn"

		log_cor_file = open(directory + "/logs/test_" + prefix + "_cor_log.txt", "a") # Correct
		log_mae_file = open(directory + "/logs/test_" + prefix + "_mae_log.txt", "a") # MAE
		log_dev_file = open(directory + "/logs/test_" + prefix + "_dev_log.txt", "a") # DEV
		log_sam_file = open(directory + "/logs/test_" + prefix + "_sam_log.txt", "a") # Sample

		ccs = []
		labls = []
		num_unlabeled = 0
		for batch_idx, (data, labels) in enumerate(loader):
			data = data.cuda()
			labels = labels.float().cuda()

			model.mode = 'natural' if natural else 'synth'
			recon_batch, mu, logvar, cc = model(data)

			cc[labels == 0] = 0  # Sets the counted cells to 0 for unlabeled data, so that regressor_loss=0
			num_unlabeled += (labels == 0).sum()
			_, _, _ = loss_function(recon_batch, data, mu, logvar, cc, labels, natural)

			ccs.append(cc.cpu().detach().numpy())
			labls.append(labels.cpu().detach().numpy())

			if batch_idx == 0 and epo % 1000 == 0:
				# Save test sample
				comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, img_size, img_size)[:n]])
				save_image(comparison.cpu(), directory + "/" + prefix + "_" + str(epo) + ".png", nrow=n)

				# Save switch sample
				model.mode = 'synth' if natural else 'natural'
				recon_batch, _, _, _ = model(data)
				comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, img_size, img_size)[:n]])
				save_image(comparison.cpu(), directory + "/switch_" + prefix + "_" + str(epo) + ".png", nrow=n)

		preds = np.concatenate(ccs, axis=None) # Elementwise round of cellcounts
		lbls = np.concatenate(labls, axis=None) # Elementswise round of labels

		log_sam_file.write(str(np.round(preds, 2)) + "\n" + str(lbls) + "\n")
		preds = np.around(preds)
		#lbls = np.around(lbls)

		correct = np.sum(preds == lbls)  # Count elementwise equality of predictions and labels
		len_set = len(loader.dataset)
		correct -= num_unlabeled  # Remove zero_indices from numerator
		correct = float(correct) / float(len_set - num_unlabeled)  # Remove zero_indices from denominator

		dist_sum = np.sum(np.abs(np.subtract(preds, lbls))) # Elementwise addition of dist between preds and lbls
		MAE = dist_sum / float(len_set - num_unlabeled)

		len_labeled = float(len_set - num_unlabeled)
		dev = np.ones(len_set) - np.divide(preds, lbls) # Deviation contains NaNs because syn data has lbl=0
		avg_dev = np.sum(np.abs(np.where(np.isnan(dev), 0, dev))) / len_labeled # Take the avg only of those deviations that weren't NaN

		log_cor_file.write(str(correct)+"\n")
		log_mae_file.write(str(MAE)+"\n")
		log_dev_file.write(str(avg_dev)+"\n")

		#logfile.write(str(correct) + " correct, MAE: " + str(MAE) + ", DEV: " + str(avg_dev) + " in " + prefix + " set in epoch " + str(epoch) + "\n\n")
		log_cor_file.close()
		log_mae_file.close()
		log_dev_file.close()
		log_sam_file.close()

		global distance_sum
		distance_sum = dist_sum
		return correct, MAE

def learn(start_epoch, max_epochs, plot_interval, test_interval, checkpoint_interval, delete_checkpoints):
	"""
	Main machine learning function. Handles epoch management, calls training and testing, creates evaluations, checkpoints and logs
	@param start_epoch: int, the epoch the learning should start in (can be non 0 when loading a model)
	@param max_epochs: int, the number of epochs to train (substracted by start_epoch)
	@param plot_interval: int, interval for plotting the UMAP
	@param test_interval: int, interval for creating a test on the current network state
	@param checkpoint_interval: int, interval for creating a .pth checkpoint and writing logs
	@param delete_checkpoints: bool, if true, deletes old checkpoints, thus only keeping the most up to date one.
	"""
	global epoch
	for epoch in range(start_epoch, int(max_epochs) + 1):
		sys.stdout.flush()
		#while os.path.exists("stop.txt"): # Allows pausing during training
		#	time.sleep(5)
		start_time = time.time()
		correct_syn, MAE_syn, avg_dev_syn = model_train(syn_train_loader, False)
		correct_nat, MAE_nat, avg_dev_nat = model_train(nat_train_loader, True)
		len_nat = len(nat_train_loader.dataset)
		len_syn = len(syn_train_loader.dataset)
		print("Train Epoch: " + str(
			epoch) + "\tNAT: Dec: {:.3f}\tKLD: {:.4f}\tCor: {:.3f}\tMAE: {:.2f}\tDEV: {:.3f}\tRegr: {:.3f}\t\tSYN: Dec: {:.3f}\tKLD: {:.4f}\tCor: {:.3f}\tMAE: {:.2f}\tDEV: {:.3f}\tRegr: {:.3f}\ttime: {:.2f}s"
			  .format(decoder_nat_loss / len_nat, KLD_nat_loss / len_nat, correct_nat, MAE_nat, avg_dev_nat, regressor_nat / len_nat,
					  decoder_syn_loss / len_syn, KLD_syn_loss / len_syn, correct_syn, MAE_syn, avg_dev_syn, regressor_syn / len_syn,
					  time.time() - start_time))
		reset_loss_sums()
		if epoch % test_interval == 0:
			correct_syn, MAE_syn = model_test(epoch, False)
			correct_nat, MAE_nat = model_test(epoch, True)
			len_nat = len(nat_test_loader.dataset)
			len_syn = len(syn_test_loader.dataset)
			print("=> Test Epoch: " + str(
				epoch) + "\tDec_nat: {:.3f}\tKLD_nat: {:.4f}\tCor_nat: {:.3f}\tMAE_nat: {:.2f}\tRegr_nat: {:.3f}\tDec_syn: {:.3f}\tKLD_syn: {:.4f}\tCor_syn: {:.3f}\tMAE_syn: {:.2f}\tRegr_syn: {:.3f}\ttime: {:.2f}s"
				  .format(decoder_nat_loss / len_nat, KLD_nat_loss / len_nat, correct_nat, MAE_nat, regressor_nat / len_nat,
						  decoder_syn_loss / len_syn, KLD_syn_loss / len_syn, correct_syn, MAE_syn, regressor_syn / len_syn,
						  time.time() - start_time))
			reset_loss_sums()
			represent()
		if epoch % eval_interval == 0:
			evaluate()
		if epoch % checkpoint_interval == 0:
			save_log(decoder_nat_log, "Decoder_nat")
			save_log(decoder_syn_log, "Decoder_syn")
			save_log(KLD_nat_log, "KLD_nat")
			save_log(KLD_syn_log, "KLD_syn")
			save_log(regressor_nat_log, "Regressor_nat")
			save_log(regressor_syn_log, "Regressor_syn")
			save_log(correct_nat_log, "Correct_nat")
			save_log(correct_syn_log, "Correct_syn")
			torch.save({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
			}, directory + '/%d.pth' % epoch)
			if delete_checkpoints and epoch > checkpoint_interval:  # the first one has to exist already
				os.remove(directory + "/" + str(epoch - checkpoint_interval) + ".pth")
		if epoch % plot_interval == 0:
			plotlyplot.directory = directory
			plotlyplot.createPlots(100, 50, directory)
	showcase()
	playSound()

def evaluate():
	"""
	Evaluates the current state of the inner representation by random sampling and sampling another time close by (noise) to allow for
	a check of visual consistency.
	"""
	model.eval()
	stddev = 1  # And mean=0
	for batch_idx, (data, _) in enumerate(syn_test_loader):
		data = data.cuda()
		if batch_idx == 0:
			noise = torch.autograd.Variable(torch.randn(batch_size, bottleneck).cuda() * stddev)
			sample_representation("orig_nat", data, noise)
			sample_representation("natural", data, noise)
			sample_representation("orig_syn", data, noise)
			sample_representation("synth", data, noise)

def represent():
	"""
	Allows to store the current representation of the inner network to either a .pt and .log file or immediately as a plotly figure in form of a UMAP.
	"""
	model.eval()
	with torch.no_grad():

		all_data = []
		all_targets = []

		for batch_idx, (data, labels) in enumerate(nat_test_loader):
			all_data.append(data)
			all_targets.append(labels.float()+50) # +50 for nat data, for distinction between nat and syn
		for batch_idx, (data, labels) in enumerate(syn_test_loader):
			all_data.append(data)
			all_targets.append(labels.float())

		all_data = torch.cat(all_data, 0) # Merges the list of tensors
		all_data = all_data.cuda()
		all_targets = torch.cat(all_targets, 0)

		representation = model.representation(all_data)
		
		torch.save(representation, directory + "/representations/repr" + str(epoch) + ".pt")
		with open(directory + "/representations/tar" + str(epoch) + ".log", "w") as f:
			for t in all_targets:
				f.write(str(t.item()) + "\n")

		# Optional: Plotting of the UMAP in each represent()
		#sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
		#reducer = umap.UMAP()
		#embedding = reducer.fit_transform(representation.cpu())
		# flatui = ["#ff0000", "#000000", "#001800", "#003000", "#004800", "#006000", "#007800", "#009000", "#00a800", "#00c000", "#00d800"]
		# plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette(flatui)[x] for x in all_targets.int()])
		#plt.scatter(embedding[:, 0], embedding[:, 1], c=all_targets.cpu())
		#plt.gca().set_aspect('equal', 'datalim')
		#plt.title('UMAP projection of cell data', fontsize=24);
		#plt.savefig(directory + "/umap_" + str(epoch) + ".png")
		#plt.clf()

len_test_nat = None
len_test_syn = None

def showcase():
	"""
	Creates overview plots from logfiles to visualize network performance. Not required for the learning workflow.
	"""
	from PIL import Image
	from PIL import ImageFont
	from PIL import ImageDraw

	# Optional: Varied loading process for showcases, when not done at the end of training
	# directory = "results/dirname"
	# checkpoint_path = directory + "/50000.pth"
	# checkpoint = torch.load(checkpoint_path)
	# epoch = checkpoint['epoch']
	"""
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in checkpoint['state_dict'].items():
		# name = k[7:] # remove `module.`
		name = k.replace(".module", "")  # removing ‘.moldule’ from key
		new_state_dict[name] = v
	# load params
	model.load_state_dict(new_state_dict)

	optimizer.load_state_dict(checkpoint['optimizer'])
	print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
	"""
	os.makedirs(directory + "/showcase", exist_ok=True)

	global decoder_nat_loss, decoder_syn_loss, KLD_syn_loss, KLD_nat_loss, regressor_nat, regressor_syn

	actual_showcase(False, False)
	reset_loss_sums()
	actual_showcase(True, False)
	reset_loss_sums()
	actual_showcase(False, True)
	reset_loss_sums()
	actual_showcase(True, True)

def actual_showcase(natural, isTest):
	"""
	Executes the actual opening of files and creation of plot image data for the showcase.
	@param natural: bool, if true, the showcase should display natural data results, else synthetic
	@param isTest: bool, if true, the showcase should display test data results, else training
	"""
	max_cells = 31
	model.eval()
	with torch.no_grad():
		if isTest:
			midfix = "test"
		else:
			midfix = "train"

		if natural:
			if isTest:
				loader = nat_test_loader
				global len_test_nat
				len_test_nat = len(nat_test_loader)
			else:
				loader = nat_train_loader
			prefix = "nat"
		else:
			if isTest:
				loader = syn_test_loader
				global len_test_syn
				len_test_syn = len(syn_train_loader)
			else:
				loader = syn_train_loader
			prefix = "syn"

		log_sam_train_file = open(directory + "/showcase/sam_train_log.txt", "a")
		log_sam_test_file = open(directory + "/showcase/sam_test_log.txt", "a")
		log_file = open(directory + "/showcase/values.txt", "a")

		ccs = []
		labls = []
		#file_names = []
		num_unlabeled = 0
		occurrences = np.zeros(max_cells)
		bar_corrects = np.zeros(max_cells)
		bar_maes = np.zeros(max_cells)
		bar_devs = np.zeros(max_cells)
		for batch_idx, (data, labels) in enumerate(loader): # , file_batch)
			for l in labels:
				occurrences[l] += 1
				if prefix == "nat" and l == 0:
					occurrences[l] -= 1

			data = data.cuda()
			labels = labels.float().cuda()

			model.mode = 'natural' if natural else 'synth'
			recon_batch, mu, logvar, cc = model(data)

			cc[labels == 0] = 0 # Sets the counted cells to 0 for unlabeled data, so that regressor_loss=0
			num_unlabeled += (labels == 0).sum()
			_, _, _ = loss_function(recon_batch, data, mu, logvar, cc, labels, natural)

			ccs.append(cc.cpu().detach().numpy())
			labls.append(labels.cpu().detach().numpy())
			#for file in file_batch:
				#file_names.append(file)

		preds = np.concatenate(ccs, axis=None)  # elementwise round of cc
		lbls = np.concatenate(labls, axis=None)  # elementswise round of labels

		# Optional: Draw label and prediction onto result images
		"""for i in range(len(lbls)):
			if lbls[i] != 0:
				img = Image.open(file_names[i]).convert('RGB')
				img = img.resize((img.width * 4, img.width * 4), Image.NEAREST)
				mode = img.mode
				new_img = Image.new(mode, (256, 292))
				new_img.paste(img, (0, 36, 256, 292))
				draw = ImageDraw.Draw(new_img)
				font = ImageFont.truetype("arial.ttf", 32)
				draw.text((4, 0), "P: " + str(np.round(preds[i],2)), (255, 255, 255), font=font)
				draw.text((128, 0), "L: " + str(lbls[i]), (255, 255, 255), font=font)
				new_img.save(directory + "/showcase/" + prefix + "_sample" + str(i) + ".jpg")"""

		pred_log = []
		lbl_log = []
		# filename_log = []
		for x in range(len(lbls)):
			if lbls[x] != 0:
				# filename_log.append(str(file_names[x]))
				pred_log.append(str(np.round(preds[x], 2)))
				lbl_log.append(str(lbls[x]))
		# log_data_file.write(str(filename_log))
		if midfix == "train":
			log_sam_train_file.write(str(pred_log) + "\n" + str(lbl_log) + "\n")
		else:
			log_sam_test_file.write(str(pred_log) + "\n" + str(lbl_log) + "\n")
		# log_data_file.write(str(file_names))
		# log_sam_file.write(str(np.round(preds[preds != 0], 2)) + "\n" + str(lbls[lbls != 0]) + "\n")
		preds = np.around(preds)

		correct = np.sum(preds == lbls) # Count elementwise equality of preds and lbls
		len_set = len(loader.dataset)
		correct -= num_unlabeled  # Remove zero_indices from numerator
		correct = float(correct) / float(len_set - num_unlabeled) # Remove zero_indices from denominator

		dist_sum = np.sum(np.abs(np.subtract(preds, lbls))) # Elementwise addition of dist between predictions and labels
		MAE = dist_sum / float(len_set - num_unlabeled)

		len_labeled = float(len_set - num_unlabeled)
		dev = np.ones(len_set) - np.divide(preds, lbls) # Deviation contains NaNs because syn data has lbl=0
		avg_dev = np.sum(np.abs(np.where(np.isnan(dev), 0, dev))) / len_labeled # Rake the avg only of those deviations that weren't NaN

		log_file.write(prefix + " " + midfix + " correct is: " + str(correct) + "\n")
		log_file.write(prefix + " " + midfix + " MAE is: " + str(MAE) + "\n")
		log_file.write(prefix + " " + midfix + " DEV is: " + str(avg_dev) + "\n")

		log_file.close()

		for i in range(0,len(preds)):
			if lbls[i] == 0:
				continue
			if preds[i] == lbls[i]:
				bar_corrects[int(lbls[i])] += 1
			bar_maes[int(lbls[i])] += np.abs(preds[i]-lbls[i])
			bar_devs[int(lbls[i])] += np.abs(1-(preds[i]/lbls[i]))

		bar_corrects = np.multiply(np.divide(bar_corrects,occurrences), 100)
		bar_maes = np.divide(bar_maes, occurrences)
		bar_devs = np.divide(bar_devs, occurrences)

		fig = go.Figure([go.Bar(x=list(range(0,max_cells)), y=bar_corrects)])
		fig.update_layout(
			title=go.layout.Title(text=prefix + " " + midfix + " % correct", xref="paper", x=0),
			xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="# cells in image", )),
			yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="% correct", ))
		)
		plotly.offline.plot(fig, filename=directory + "/showcase/" + prefix + "_" + midfix + "_correct.html", auto_open=False) # Includes fig.show()

		fig = go.Figure([go.Bar(x=list(range(0, max_cells)), y=bar_maes)])
		fig.update_layout(
			title=go.layout.Title(text=prefix + " " + midfix + " MAE", xref="paper", x=0),
			xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="# cells in image", )),
			yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="MAE", ))
		)
		plotly.offline.plot(fig, filename=directory + "/showcase/" + prefix + "_" + midfix + "_MAE.html", auto_open=False) # Includes fig.show()

		fig = go.Figure([go.Bar(x=list(range(0, max_cells)), y=bar_devs)])
		fig.update_layout(
			title=go.layout.Title(text=prefix + " " + midfix + " DEV", xref="paper", x=0),
			xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="# cells in image", )),
			yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="DEV", ))
		)
		plotly.offline.plot(fig, filename=directory + "/showcase/" + prefix + "_" + midfix + "_DEV.html", auto_open=False) # Includes fig.show()

		fig = go.Figure([go.Bar(x=list(range(0, max_cells)), y=occurrences)])
		fig.update_layout(
			title=go.layout.Title(text=prefix + " " + midfix + " occurrences", xref="paper", x=0),
			xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="# cells in image", )),
			yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="# images with that cell count", ))
		)
		plotly.offline.plot(fig, filename=directory + "/showcase/" + prefix + "_" + midfix + "_proportion.html", auto_open=False) # Includes fig.show()

def init_weights(m):
	"""
	Performans a xavier_normal initialization for the network weights and fills biases with a low start value.
	@param m: the part of the model that should be initialized by this
	"""
	if type(m) == nn.Linear:
		torch.nn.init.xavier_normal(m.weight)
		m.bias.data.fill_(0.01)

def main(lr, inter_dim, bneck, d_l_f, r_l_f, KLD_l_f, result_dir, checkpoint_path, max_epochs, regr_start,  plot_interval, test_interval, checkpoint_interval, delete_checkpoints, weight_decay, dropout_enc, dropout_fc, leak_enc, leak_dec, bs, s, syn_tr_name, syn_te_name, n):
	"""
	Main overall management function. Recieves parameters from the command-line interface, merges them with global variables if needed, creates required
	directories, handles model loading, code backup etc.
	@param lr: for this and other params description see cly.pi
	@return: returns a score for the trained hyperparameter run. Only used in metalearning.py
	"""
	global learning_rate, bottleneck, decoder_l_factor, regressor_l_factor, KLD_l_factor, model, optimizer, directory, regressor_start, batch_size, seed, syn_train_name, syn_test_name, img_noise
	learning_rate = lr
	inter_dim = int(inter_dim)
	bneck = int(bneck)
	bottleneck = bneck
	decoder_l_factor = d_l_f
	regressor_l_factor = r_l_f
	KLD_l_factor = KLD_l_f
	directory = "results/" + result_dir
	regressor_start = regr_start
	batch_size = bs
	seed = s
	set_seed(seed)
	syn_train_name = syn_tr_name
	syn_test_name = syn_te_name
	img_noise = n

	from dataloader import syn_train_load, syn_test_load, nat_train_load, nat_test_load
	global syn_train_loader, syn_test_loader, nat_train_loader, nat_test_loader
	syn_train_loader = syn_train_load
	syn_test_loader = syn_test_load
	nat_train_loader = nat_train_load
	nat_test_loader = nat_test_load

	os.makedirs(directory, exist_ok=True)
	os.makedirs(directory + "/code", exist_ok=True)
	os.makedirs(directory + "/logs", exist_ok=True)
	os.makedirs(directory + "/representations", exist_ok=True)

	sys.stdout = open(directory + "/stdout.txt", "a")

	print("Testing: lr: " + str(lr) + " inter_dim: " + str(inter_dim) + " bneck: " + str(bneck))

	# Optional: Use special weight initialization or not
	model = VAE.VAE(inter_dim, bneck, dropout_enc, dropout_fc, leak_enc, leak_dec).cuda()
	#model.apply(init_weights)

	# Optional: Use RAdam or Adam
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	#optimizer = RAdam(model.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=weight_decay)

	if checkpoint_path is not None:
		checkpoint_path = "results/" + result_dir + "/" + checkpoint_path
		if os.path.isfile(checkpoint_path):
			print("Loading checkpoint '{}'".format(checkpoint_path))
			checkpoint = torch.load(checkpoint_path)
			start_epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("Loaded checkpoint '{}' (epoch {})"
				  .format(checkpoint_path, checkpoint['epoch']))

			global decoder_nat_log, decoder_syn_log, KLD_nat_log, KLD_syn_log, regressor_nat_log, regressor_syn_log, correct_nat_log, correct_syn_log
			decoder_nat_log = load_log("/logs/Decoder_nat_log.txt")
			decoder_syn_log = load_log("/logs/Decoder_syn_log.txt")
			KLD_nat_log = load_log("/logs/KLD_nat_log.txt")
			KLD_syn_log = load_log("/logs/KLD_syn_log.txt")
			regressor_nat_log = load_log("/logs/Regressor_nat_log.txt")
			regressor_syn_log = load_log("/logs/Regressor_syn_log.txt")
			correct_nat_log = load_log("/logs/Correct_nat_log.txt")
			correct_syn_log = load_log("/logs/Correct_syn_log.txt")
		else:
			print("No checkpoint found at '{}'".format(checkpoint_path))
			quit()
	else:
		start_epoch = 1
		
		# Optional: Create a copy of the used code
		#cur_dir = "./PycharmProjects/vae/"#"./"
		cur_dir = __file__[:-10]
		copyfile(cur_dir + "cli.py", directory + "/code/cli.py")
		copyfile(cur_dir + "dataloader.py", directory + "/code/dataloader.py")
		copyfile(cur_dir + "./learner.py", directory + "/code/learner.py")
		copyfile(cur_dir + "./metalearning.py", directory + "/code/metalearning.py")
		copyfile(cur_dir + "./plotlyplot.py", directory + "/code/plotlyplot.py")
		copyfile(cur_dir + "./radam.py", directory + "/code/radam.py")
		copyfile(cur_dir + "./VAE.py", directory + "/code/VAE.py")
		f=open(directory + "/code/args.txt","w+")
		f.write('\n'.join(sys.argv[1:]))
		f.close()

	learn(start_epoch, max_epochs, plot_interval, test_interval, checkpoint_interval, delete_checkpoints)
	return 1  # return 1e6 / distance_sum
