import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time
import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as pp

mi.set_variant('cuda_ad_rgb')

def canonical_to_dir(p: mi.Point2f) -> mi.Vector3f:
	cosTheta = 2 * p.x - 1
	phi = 2 * dr.pi * p.y
	sinTheta = dr.sqrt(dr.maximum(0, 1 - cosTheta**2))
	return mi.Vector3f(sinTheta * dr.cos(phi), sinTheta * dr.sin(phi), cosTheta)

def dir_to_canonical(wi: mi.Vector3f) -> mi.Point2f:
	cosTheta = wi.z
	phi = dr.atan2(wi.y, wi.x)
	phi = dr.select(phi < 0, phi + 2 * dr.pi, phi)
	return mi.Point2f((cosTheta + 1) * 0.5, phi / (2 * dr.pi))

def hem_canonical_to_dir(p: mi.Point2f) -> mi.Vector3f:
    cosTheta = 1 - p.x
    phi = 2 * dr.pi * p.y
    sinTheta = dr.sqrt(dr.maximum(0, 1 - cosTheta**2))
    return mi.Vector3f(sinTheta * dr.cos(phi), sinTheta * dr.sin(phi), cosTheta)

def hem_dir_to_canonical(wi: mi.Vector3f) -> mi.Point2f:
    cosTheta = -wi.z
    phi = dr.atan2(wi.y, wi.x)
    phi = dr.select(phi < 0, phi + 2 * dr.pi, phi)
    return mi.Point2f((cosTheta + 1), phi / (2 * dr.pi))

def rotate_align_scalar(v1, v2):
	v1 = v1 / np.linalg.norm(v1)
	v2 = v2 / np.linalg.norm(v2)
	axis = np.cross(v1, v2)
	
	cosA = np.dot(v1, v2)
	k = 1.0 / (1.0 + cosA)
	
	return mi.ScalarTransform4f(
		[[(axis[0] * axis[0] * k) + cosA, (axis[1] * axis[0] * k) - axis[2], (axis[2] * axis[0] * k) + axis[1], 0],
		[(axis[0] * axis[1] * k) + axis[2], (axis[1] * axis[1] * k) + cosA, (axis[2] * axis[1] * k) - axis[0], 0], 
		[(axis[0] * axis[2] * k) - axis[1], (axis[1] * axis[2] * k) + axis[0], (axis[2] * axis[2] * k) + cosA, 0],
		[0, 0, 0, 1]] 
	)

def scene_from_normal(normal):
	given_normal = np.array([0, 0, 1])
	desired_normal = normal

	m = rotate_align_scalar(given_normal, desired_normal)

	return mi.load_dict({
		'type': 'scene',
		'light': {'type': 'envmap', 'filename': 'C:/Users/Vishu.Main-Laptop/Downloads/Work/tiny-cuda-nn/data/images/studio_small_09_1k.exr'},
		'rectangle' : {
			'type': 'rectangle',
			'to_world': m,
			'bsdf': {
				'type': 'roughconductor',
				'material': 'Al',
				'distribution': 'ggx',
				'alpha': 0.1,
				'sample_visible' : True,
			},
		}
	})
	
def compute_image(scene: mi.Scene, directions: mi.Vector3f, spps: int):
	rng = mi.PCG32(size=dr.width(directions))

	result = mi.Color3f(0, 0, 0)

	bsdf_context = mi.BSDFContext()
	ray = mi.Ray3f(o=directions, d=-directions) # Ray towards 0.0
	si = scene.ray_intersect(ray)
	i = mi.UInt32(0)

	loop = mi.Loop(name="", state=lambda: (rng, i, result))

	while loop(i < spps):
		# TODO: Add better MIS (e.g. bidirectional approach)
		# Sample the BSDF
		bsdf_sample, bsdf_val = si.bsdf().sample(bsdf_context, si, 
												 rng.next_float32(), 
												 mi.Point2f(rng.next_float32(), rng.next_float32()), 
												 si.is_valid())
		
		# Create new ray
		new_direction = si.sh_frame.to_world(bsdf_sample.wo)
		new_ray = mi.Ray3f(o=si.p, d=new_direction)
		# Intersect the new ray
		si_new = scene.ray_intersect(new_ray)
		# Monte Carlo estimator
		result +=  bsdf_val * si_new.emitter(scene, si.is_valid()).eval(si_new)
		
		i = i + 1

	result = result / spps
	
	return result


def rotate_align(v1: mi.Vector3f, v2: mi.Vector3f):
	check = (1 + dr.sign(dr.dot(v1, v2) + 1 - 0.0001)) / 2
	axis = dr.cross(v1, v2)
	
	cosA = dr.dot(v1, v2)
	k = mi.Float(1.0 / (1.0 + cosA * check))
	
	return mi.Matrix3f(
		[[(axis[0] * axis[0] * k) + cosA, (axis[1] * axis[0] * k) - axis[2], (axis[2] * axis[0] * k) + axis[1]],
		[(axis[0] * axis[1] * k) + axis[2], (axis[1] * axis[1] * k) + cosA, (axis[2] * axis[1] * k) - axis[0]], 
		[(axis[0] * axis[2] * k) - axis[1], (axis[1] * axis[2] * k) + axis[0], (axis[2] * axis[2] * k) + cosA]] 
	)

def matmul(rotmat : mi.Matrix3f, vec: mi.Vector3f):
	new_vector = mi.Vector3f(
		rotmat[0][0] * vec[0] + rotmat[1][0] * vec[1] + rotmat[2][0] * vec[2],
		rotmat[0][1] * vec[0] + rotmat[1][1] * vec[1] + rotmat[2][1] * vec[2],
		rotmat[0][2] * vec[0] + rotmat[1][2] * vec[1] + rotmat[2][2] * vec[2]
	)
	return new_vector

def compute_grid(normals: mi.Vector3f, directions: mi.Vector3f, spps: int):
	given_normal = mi.Vector3f(0, 0, 1)
	rotation_mat = rotate_align(normals, given_normal)
	inverse_mat = rotate_align(given_normal, normals)
	local_view_directions = directions
	
	si = mi.SurfaceInteraction3f()
	si.n = normals
	si.p = mi.Point3f(0, 0, 0)
	si.wi = local_view_directions
	
	envmap_scene = mi.load_dict({
		'type': 'scene',
		'light': {'type': 'envmap', 'filename': 'C:/Users/Vishu.Main-Laptop/Downloads/Work/tiny-cuda-nn/data/images/studio_small_09_1k.exr'}
	})

	material = mi.load_dict({
		'type': 'roughconductor',
		'material': 'Al',
		'distribution': 'ggx',
		'alpha': 0.1,
		'sample_visible' : True,
	})

	i = mi.Float(0)
	rng = mi.PCG32(size=dr.width(directions))

	result = mi.Color3f(0, 0, 0)
	bsdf_context = mi.BSDFContext()

	grid_loop = mi.Loop(name="", state=lambda: (rng, i, result))
	
	while grid_loop(i < spps):
		# TODO: Add better MIS (e.g. bidirectional approach)
		# Sample the BSDF
		bsdf_sample, bsdf_val = material.sample(bsdf_context, si, 
													rng.next_float32(), 
													mi.Point2f(rng.next_float32(), rng.next_float32()), 
													True)
		
		# Create new ray
		new_direction = matmul(inverse_mat, bsdf_sample.wo)
		new_ray = mi.Ray3f(o=si.p, d=new_direction)
		# Intersect the new ray
		si_new = envmap_scene.ray_intersect(new_ray)
		# Monte Carlo estimator
		result +=  bsdf_val * si_new.emitter(envmap_scene, True).eval(si_new)
		#result += directions
		
		i = i + 1
	
	result = result / spps
	
	return result


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

class Grid(torch.nn.Module):
	def __init__(self):
		super(Grid, self).__init__()

	def forward(self, dir, spp):
		with torch.no_grad():
			# Bilinearly filtered lookup from the image. Not super fast,
			# but less than ~20% of the overall runtime of this example.
			dir_tensor = mi.TensorXf(dir)
			xs_points = mi.Point2f(dir_tensor[:, 2:4].numpy())
			nxs_points = mi.Point2f(dir_tensor[:, 0:2].numpy())
			xs_directions = hem_canonical_to_dir(xs_points)
			nxs_directions = canonical_to_dir(nxs_points)
			return compute_grid(nxs_directions, xs_directions, spp)

def get_args():
	parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")

	parser.add_argument("image", nargs="?", default="data/images/test.jpg", help="Image to match")
	parser.add_argument("config", nargs="?", default="data/config_hash_4D.json", help="JSON config for tiny-cuda-nn")
	parser.add_argument("n_steps", nargs="?", type=int, default=35000, help="Number of training steps")
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

	view_resolution = 512
	n_channels = 3

	model = tcnn.NetworkWithInputEncoding(n_input_dims=4, n_output_dims=n_channels, encoding_config=config["encoding"], network_config=config["network"]).to(device)
	print(model)

	#===================================================================================================
	# The following is equivalent to the above, but slower. Only use "naked" tcnn.Encoding and
	# tcnn.Network when you don't want to combine them. Otherwise, use tcnn.NetworkWithInputEncoding.
	#===================================================================================================
	# encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"])
	# network = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=n_channels, network_config=config["network"])
	# model = torch.nn.Sequential(encoding, network)
	learning_rate = 0.05
	print(learning_rate)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# Variables for saving/displaying image results
	print("Grid shape: ", [view_resolution, view_resolution])
	print("Number of channels: ", n_channels)
	img_shape = [view_resolution, view_resolution, n_channels]
	grid = Grid()

	half_dx =  0.5 / view_resolution
	half_dy =  0.5 / view_resolution
	xs = torch.linspace(half_dx, 1-half_dx, view_resolution, device=device)
	ys = torch.linspace(half_dy, 1-half_dy, view_resolution, device=device)
	xn = torch.rand(1, device=device, dtype=torch.float32)
	yn = torch.rand(1, device=device, dtype=torch.float32)
	nyv, nxv, xv, yv = torch.meshgrid([yn, xn, ys, xs])

	nynxxy = torch.stack((nyv.flatten(), nxv.flatten(), yv.flatten(), xv.flatten())).t()
 
	path = f"Grid4DReference.jpg"
	print(f"Writing '{path}'... ", end="")
	reference_spp = 10 ** 5
	grid_output = grid(nynxxy, reference_spp).numpy().reshape(img_shape)
	write_image(path, grid_output)
	print("done.")

	prev_time = time.perf_counter()

	batch_size = 2**18
	interval = 10
	train_spp = 2**14

	print(f"Beginning optimization with {args.n_steps} training steps.")

	sensor = mi.load_dict({
		"type": "perspective",
		"film": {
			"type": "hdrfilm",
			"width": 512,
			"height": 512,
			"rfilter": {"type": "box"}
		},
		"sampler": {
			"type": "independent",
			"sample_count": 128
		},
		"to_world" : mi.ScalarTransform4f.look_at(origin=[0, 0, 1], target=[0, 0, 0], up=[0, 1, 0]),
		"fov": 120,
		"near_clip": 0.1,
		"far_clip": 1000
	})

	model_path = f"model.pt"

	for i in range(args.n_steps):
		batch = torch.rand([batch_size, 4], device=device, dtype=torch.float32)
		targets = torch.from_numpy(grid(batch, train_spp).numpy()).float().to(device)
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
			print(f"Step#{i}: loss={loss_val} time={int(elapsed_time)}[s]")

			path = f"4D{i}.jpg"
			print(f"Writing '{path}'... ", end="")
			with torch.no_grad():
				image_output = model(nynxxy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy()
				if image_output.shape[2] == 1:
					image_output = np.repeat(image_output, 3, axis=2)
				write_image(path, image_output)
			print("done.")

			# Ignore the time spent saving the image
			prev_time = time.perf_counter()
	
			torch.save(model.state_dict(), f"model.pt")
			if i > 0 and interval < 1000:
				interval *= 10

	if args.result_filename:
		print(f"Writing '{args.result_filename}'... ", end="")
		with torch.no_grad():
			write_image(args.result_filename, model(nynxxy).reshape(img_shape).clamp(0.0, 1.0).detach().cpu().numpy())
		print("done.")

	class NormalIntegrator(mi.SamplingIntegrator):
		def __init__(self, arg0: mi.Properties):
			mi.SamplingIntegrator.__init__(self, arg0)
		
		def sample(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.RayDifferential3f, medium=None, active=True):
			si = scene.ray_intersect(ray)
			color = mi.Color3f(0.0)
			wi_world = si.to_world(si.wi)
			n = dir_to_canonical(si.sh_frame.n)
			p = dir_to_canonical(wi_world)
			dir_count = dr.width(p)
			directions = np.concatenate((n.numpy(), p.numpy()), axis=1)
			directions = torch.from_numpy(directions).float().to(device)
			model_out = model(directions).reshape(dir_count, 3).clamp(0.0, 1.0).detach().cpu().numpy()
			model_color_out = mi.Color3f(model_out)
			color[si.is_valid()] = model_color_out
			color[~si.is_valid()] = si.emitter(scene, True).eval(si)
			return (color, si.is_valid(), [])
	
	normal = np.array([0, 0, 1])
	test_scene = scene_from_normal(normal)	
	normal_int = NormalIntegrator(mi.Properties())
	img_cached = normal_int.render(test_scene, sensor=sensor, spp=64)
	path = f"Grid4DRender.jpg"
	print(f"Writing '{path}'... ", end="")
	write_image(path, img_cached.numpy().reshape(512, 512, 3))
	print("done.")

	tcnn.free_temporary_memory()