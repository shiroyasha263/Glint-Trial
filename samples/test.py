import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time
import mitsuba as mi
import drjit as dr

try:
	import tinycudann as tcnn
except ImportError:
	print("This sample requires the tiny-cuda-nn extension for PyTorch.")
	print("You can install it by running:")
	print("============================================================")
	print("tiny-cuda-nn$ cd bindings/torch")
	print("tiny-cuda-nn/bindings/torch$ python setup.py install")
	print("============================================================")
	sys.exit()

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from common import read_image, write_image, ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

mi.set_variant('cuda_ad_rgb')

class Image(torch.nn.Module):
	def __init__(self, filename, device):
		super(Image, self).__init__()
		self.data = read_image(filename)
		self.shape = self.data.shape
		self.data = torch.from_numpy(self.data).float().to(device)

	def forward(self, xs):
		with torch.no_grad():
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.
			shape = self.shape

			xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
			indices = xs.long()
			lerp_weights = xs - indices.float()

			x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
			y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
			x1 = (x0 + 1).clamp(max=shape[1]-1)
			y1 = (y0 + 1).clamp(max=shape[0]-1)

			return (
				self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
				self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
				self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
				self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
			)

class MaterialLookUp(torch.nn.Module):
	def __init__(self, spp):
		super(MaterialLookUp, self).__init__()
		self.spp = spp
  
	def forward(self, xs):
		w_costheta = 2 * xs[:, 0]
		w_phi = 2 * torch.pi * xs[:, 1]
		w_sintheta = torch.sqrt(torch.maximum(0, 1 - w_costheta ** 2))
		w_direction = torch.stack((w_sintheta * torch.cos(w_phi), w_sintheta * torch.sin(w_phi), w_costheta)).t()

		

def get_args():
	parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

	parser.add_argument("image", nargs="?", default="data/images/test.jpg", help="Image to match")
	parser.add_argument("config", nargs="?", default="data/config_hash.json", help="JSON config for tiny-cuda-nn")
	parser.add_argument("n_steps", nargs="?", type=int, default=10000000, help="Number of training steps")
	parser.add_argument("result_filename", nargs="?", default="", help="Number of training steps")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	print("================================================================")
	print("This script replicates the behavior of the native CUDA example  ")
	print("mlp_learning_an_image.cu using tiny-cuda-nn's PyTorch extension.")
	print("================================================================")

	print(f"Using PyTorch version {torch.__version__} with CUDA {torch.version.cuda}")

	device = torch.device("cuda")
	args = get_args()

	with open(args.config) as config_file:
		config = json.load(config_file)

	image = Image(args.image, device)
	n_channels = image.data.shape[2]

	model = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=n_channels, encoding_config=config["encoding"], network_config=config["network"]).to(device)
	print(model)

	#===================================================================================================
	# The following is equivalent to the above, but slower. Only use "naked" tcnn.Encoding and
	# tcnn.Network when you don't want to combine them. Otherwise, use tcnn.NetworkWithInputEncoding.
	#===================================================================================================
	# encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
	# network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
	# model = torch.nn.Sequential(encoding, network)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	# Variables for saving/displaying image results
	resolution = image.data.shape[0:2]
	img_shape = resolution + torch.Size([image.data.shape[2]])
	n_pixels = resolution[0] * resolution[1]

	half_dx =  0.5 / resolution[0]
	half_dy =  0.5 / resolution[1]
	#List of all possible x pixels and y pixels at center, xv, yv, are the points x and y coordinates of the xs, ys mesh created
	xs = torch.linspace(half_dx, 1-half_dx, resolution[0], device=device)
	ys = torch.linspace(half_dy, 1-half_dy, resolution[1], device=device)
	xv, yv = torch.meshgrid([xs, ys])
	
	#Gives the index locations as an output
	xy = torch.stack((yv.flatten(), xv.flatten())).t()
	
	path = f"reference.jpg"
	print(f"Writing '{path}'... ", end="")
	image_output = image(xy).reshape(img_shape).detach().cpu().numpy()
	if image_output.shape[2] == 1:
		image_output = np.repeat(image_output, 3, axis=2)
	write_image(path, image_output)
	print("done.")

	mi.Point2f(xy)
	print(xy)

	prev_time = time.perf_counter()

	batch_size = 2**18
	interval = 10

	print(f"Beginning optimization with {args.n_steps} training steps.")

	try:
		batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
		traced_image = torch.jit.trace(image, batch)
	except:
		# If tracing causes an error, fall back to regular execution
		print(f"WARNING: PyTorch JIT trace failed. Performance will be slightly worse than regular.")
		traced_image = image

	for i in range(args.n_steps):
		batch = torch.rand([batch_size, 2], device=device, dtype=torch.float32)
		targets = traced_image(batch)
		output = model(batch)

		relative_l2_error = (output - targets.to(output.dtype))**2 / (output.detach()**2 + 0.01)
		loss = relative_l2_error.mean()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % interval == 0:
			loss_val = loss.item()
			torch.cuda.synchronize()
			elapsed_time = time.perf_counter() - prev_time
			print(f"Step#{i}: loss={loss_val} time={int(elapsed_time*1000000)}[µs]")

			path = f"{i}.jpg"
			print(f"Writing '{path}'... ", end="")
			with torch.no_grad():
				image_output = model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
				if image_output.shape[2] == 1:
					image_output = np.repeat(image_output, 3, axis=2)
				write_image(path, image_output)
			print("done.")

			# Ignore the time spent saving the image
			prev_time = time.perf_counter()

			if i > 0 and interval < 1000:
				interval *= 10

	if args.result_filename:
		print(f"Writing '{args.result_filename}'... ", end="")
		with torch.no_grad():
			write_image(args.result_filename, model(xy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
		print("done.")

	tcnn.free_temporary_memory()