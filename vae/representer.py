import os, glob
import seaborn as sns
import umap
import torch
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

""" Used to create UMAPs from representations saved during training"""

dir = "results/bf" # Group folder, i.e. the folder that contains folders with runs

def create_umap(name):
	"""
	Creates a UMAP from .log and .pt files within 'name'
	@param name: string, name of the folder to work on (dir)
	"""
	global dir
	direc = dir + "/" + name + "/"
	os.chdir(direc + "representations/")
	
	# Palette size of 2x50 required. 1-49 for labeled nat data, 51-100 for labeled syn data, 50 for unlabeled nat data
	palette = sns.color_palette("Blues_d", 30)# Syn data in blue
	palette.extend(sns.dark_palette("purple", 20)) # Unimportant, just a filler
	palette.extend(sns.color_palette("Reds_d", 30))# Nat data in red
	palette.extend(sns.dark_palette("purple", 20))# Unimportant, just a filler
	palette[49]="#50B689"# Unlabeled nat data in green
	# print("size of palette " + str(len(palette)))
	
	for file in glob.glob("*.pt"):
			representation = torch.load(file)
			tarfile = file[:-3] # Removes the .pt ending
			tarfile = "tar" + tarfile[4:] + ".log"
			all_targets = []
			with open(tarfile, "r") as f:
				for tar in f:
					all_targets.append(float(tar.strip()))

			sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
			reducer = umap.UMAP()
			embedding = reducer.fit_transform(representation.cpu())
			
			print("scattering")
			# print(all_targets)
			plt.scatter(embedding[:, 0], embedding[:, 1], c=[palette[int(y-1)] for y in all_targets], alpha=0.8)
			plt.gca().set_aspect('equal', 'datalim')
			plt.title('UMAP projection of cell data', fontsize=24);
			plt.savefig("./umap_" + str(file[4:-3]) + ".png")
			plt.clf()
	os.chdir("../../../../")

Parallel(n_jobs=4)(delayed(create_umap)(name) for name in os.listdir(dir))
	