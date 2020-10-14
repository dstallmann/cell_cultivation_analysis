import torch
from torch import nn

torch.manual_seed(2)
torch.cuda.manual_seed_all(2)

class Flatten(nn.Module):
	def forward(self, data):
		return data.view(data.size(0), -1)

class UnFlatten(nn.Module):
	def forward(self, data):
		return data.view(data.size(0), -1, 1, 1)

def reparameterize(mu, logvar):
	mu = mu.cuda()
	std = logvar.mul(0.5).exp_().cuda()
	# return torch.normal(mu, std).cuda()
	eps = torch.randn(*mu.size()).cuda() # Statistical epsilon
	# eps = torch.randn_like(std).cuda()
	return mu + eps * std

class VAE(nn.Module):
	def __init__(self, h_dim, z_dim, dropout_enc, dropout_fc, leak_enc, leak_dec):
		super(VAE, self).__init__()

		self.mode = 'natural'

		# size:=(size−kernel_size+2*pad)/stride+1
		self.synth_reader = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=5, stride=2),# 128=>63
			nn.LeakyReLU(leak_enc),
			nn.Dropout(dropout_enc),
			nn.Conv2d(32, 64, kernel_size=5, stride=2),# 63=>30
			nn.LeakyReLU(leak_enc),
			nn.Dropout(dropout_enc),
			nn.Conv2d(64, 128, kernel_size=5, stride=2),
			nn.LeakyReLU(leak_enc),
			nn.Dropout(dropout_enc),
			nn.Conv2d(128, 256, kernel_size=5, stride=2),
			nn.LeakyReLU(leak_enc),
			nn.Dropout(dropout_enc),
		)
		self.synth_reader = nn.DataParallel(self.synth_reader)
		
		self.natural_reader = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=5, stride=2),
			nn.LeakyReLU(leak_enc),
			nn.Dropout(dropout_enc),
			nn.Conv2d(32, 64, kernel_size=5, stride=2),
			nn.LeakyReLU(leak_enc),
			nn.Dropout(dropout_enc),
			nn.Conv2d(64, 128, kernel_size=5, stride=2),
			nn.LeakyReLU(leak_enc),
			nn.Dropout(dropout_enc),
			nn.Conv2d(128, 256, kernel_size=5, stride=2),
			nn.LeakyReLU(leak_enc),
			nn.Dropout(dropout_enc),
		)
		self.natural_reader = nn.DataParallel(self.natural_reader)

		self.common_encoder = nn.Sequential(
			nn.Conv2d(256, h_dim, kernel_size=5, stride=2),
			nn.LeakyReLU(leak_enc),
			nn.Dropout(dropout_enc),
			Flatten()
		)
		self.common_encoder = nn.DataParallel(self.common_encoder)

		self.fc1 = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Dropout(dropout_fc),
		)
		self.fc1 = nn.DataParallel(self.fc1)

		self.fc2 = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Dropout(dropout_fc),
		)
		self.fc2 = nn.DataParallel(self.fc2)

		self.fc3 = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.Dropout(dropout_fc),
		)
		self.fc3 = nn.DataParallel(self.fc3)

		self.common_decoder = nn.Sequential(
			UnFlatten(),
			nn.ConvTranspose2d(h_dim, 256, kernel_size=5, stride=2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(leak_dec),
		)
		self.common_decoder = nn.DataParallel(self.common_decoder)

		self.synth_writer = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(leak_dec),
			nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(leak_dec),
			nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(leak_dec),
			nn.ConvTranspose2d(32, 32, kernel_size=2, stride=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(leak_dec),
			nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2),
			nn.Sigmoid(),
		)
		self.synth_writer = nn.DataParallel(self.synth_writer)
		
		self.natural_writer = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(leak_dec),
			nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(leak_dec),
			nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(leak_dec),
			nn.ConvTranspose2d(32, 32, kernel_size=2, stride=1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(leak_dec),
			nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2),
			nn.Sigmoid(),
		)
		self.natural_writer = nn.DataParallel(self.natural_writer)

		self.regressor = nn.Sequential(
			nn.Linear(z_dim, 256),
			nn.Dropout(dropout_fc),
			nn.Linear(256, 128),
			nn.Dropout(dropout_fc),
			nn.Linear(128, 1), # nn.Softplus()
		)
		self.regressor = nn.DataParallel(self.regressor)

	def bottleneck(self, h):
		mu = self.fc1(h).cuda() # Mean
		logvar = self.fc2(h).cuda() # Variance
		z = reparameterize(mu, logvar).cuda()
		return z, mu, logvar

	def representation(self, x):
		if self.mode == 'natural':
			h = self.natural_reader(x).cuda()
		else:
			h = self.synth_reader(x).cuda()
		h = self.common_encoder(h).cuda()
		return self.bottleneck(h)[0] # [naturaltestlen+synthtestlen, bottlenecksize]

	def forward(self, x):
		if self.mode == 'natural':
			h = self.natural_reader(x).cuda()
		else:
			h = self.synth_reader(x).cuda()
		h = self.common_encoder(h).cuda()
		z, mu, logvar = self.bottleneck(h)
		cc = self.regressor(z) # Cell count
		z = self.fc3(z)
		y = self.common_decoder(z).cuda()
		if self.mode == 'natural':
			y = self.natural_writer(y)
		else:
			y = self.synth_writer(y)
		return y, mu, logvar, cc

	def sample_start(self, x):
		if self.mode == 'natural':
			h = self.natural_reader(x).cuda()
		else:
			h = self.synth_reader(x).cuda()
		h = self.common_encoder(h).cuda()
		z, mu, logvar = self.bottleneck(h)
		return z

	def sample_end(self, z):
		cc = self.regressor(z)
		z = self.fc3(z).cuda()
		y = self.common_decoder(z).cuda()
		if self.mode == 'natural':
			y = self.natural_writer(y)
		else:
			y = self.synth_writer(y)
		return y, cc
