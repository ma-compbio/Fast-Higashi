import torch
import torch.jit as jit
import numpy as np
from functools import partial

def project2orthogonal(matrix: torch.Tensor, rank:int, compute_device:torch.device):
	dim_1, dim_2 = matrix.shape[-2], matrix.shape[-1]
	if rank is None: rank = min(matrix.shape[-2:])
	try:
		if matrix.shape[-2] / matrix.shape[-1] >= 0.5:
			try:
				U, S, Vh = torch.linalg.svd(matrix, full_matrices=False, driver='gesvda')
				final =  U[..., :rank].to(compute_device) @ Vh[..., :rank, :].to(compute_device)
				if torch.any(torch.isnan(final)) or torch.any(torch.isinf(final)):
					U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
					final = U[..., :rank].to(compute_device) @ Vh[..., :rank, :].to(compute_device)
			except Exception as e:
				U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
				final = U[..., :rank].to(compute_device) @ Vh[..., :rank, :].to(compute_device)
		else:
			U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
			final = U[..., :rank].to(compute_device) @ Vh[..., :rank, :].to(compute_device)
		# a = U[..., :rank].to(compute_device) @ Vh[..., :rank, :].to(compute_device)
		# if torch.sum(torch.isnan(a)) > 0:
		# 	print("a", a, torch.sum(torch.isnan(a)), a.shape)
		if torch.any(torch.isnan(final)):
			print("gesvd & default failed")
			raise BaseException
		return final, S[..., :rank]
	except Exception as e:
		print(f'error {e}. using eigh, shape = {matrix.shape}')

		kk = 1e-2
		U, S, V, Vh = None, None, None, None
		mode = dim_2 > dim_1
		X = matrix @ matrix.transpose(-1, -2) if mode else matrix.transpose(-1, -2) @ matrix
		t = X.diagonal(dim1=-1, dim2=-2)
		t += kk
		del t
		eigvals, eigvecs = torch.linalg.eigh(X)
		eigvecs = eigvecs[..., -rank:].flip(-1)
		if mode:
			U = eigvecs
			Vh = U.transpose(-1, -2) @ matrix
			V, R = torch.linalg.qr(Vh.transpose(-1, -2))
			V = V.mul_(R.diagonal(dim1=-1, dim2=-2).sign()[..., None, :])
		else:
			V = eigvecs
			U = matrix @ V
			U, R = torch.linalg.qr(U)
			U = U.mul_(R.diagonal(dim1=-1, dim2=-2).sign()[..., None, :])
		UVh = U @ V.transpose(-1, -2)
		assert (U.transpose(-1, -2) @ U - torch.eye(U.shape[-1], device=U.device)).max().item() < 1e-4
		assert (V.transpose(-1, -2) @ V - torch.eye(V.shape[-1], device=V.device)).max().item() < 1e-4
		return UVh, S


# @jit.script
# def project2orthogonal(matrix: torch.Tensor, rank:int, compute_device:torch.device):
# 	dim_1, dim_2 = matrix.shape[-2], matrix.shape[-1]
# 	if rank is None: rank = min(matrix.shape[-2:])
# 	# try:
# 	U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
# 	return U[..., :rank].to(compute_device) @ Vh[..., :rank, :].to(compute_device), S[..., :rank]
#
# @jit.script
# def project2orthogonal_ill(matrix: torch.Tensor, rank:int, compute_device:torch.device):
# 	dim_1, dim_2 = matrix.shape[-2], matrix.shape[-1]
# 	if rank is None: rank = min(matrix.shape[-2:])
# 	# except Exception as e:
# 	print('ill conditioned matrix. using eigh, shape = {matrix.shape}')
#
# 	kk = 1e-2
# 	# U, S, V, Vh = None, None, None, None
# 	mode = dim_2 > dim_1
# 	X = matrix @ matrix.transpose(-1, -2) if mode else matrix.transpose(-1, -2) @ matrix
# 	t = X.diagonal(dim1=-1, dim2=-2)
# 	t += kk
# 	del t
# 	eigvals, eigvecs = torch.linalg.eigh(X)
# 	eigvecs = eigvecs[..., -rank:].flip(-1)
# 	if mode:
# 		U = eigvecs
# 		Vh = U.transpose(-1, -2) @ matrix
# 		V, R = torch.linalg.qr(Vh.transpose(-1, -2))
# 		V = V.mul_(R.diagonal(dim1=-1, dim2=-2).sign()[..., None, :])
# 	else:
# 		V = eigvecs
# 		U = matrix @ V
# 		U, R = torch.linalg.qr(U)
# 		U = U.mul_(R.diagonal(dim1=-1, dim2=-2).sign()[..., None, :])
# 	UVh = U @ V.transpose(-1, -2)
# 	assert (U.transpose(-1, -2) @ U - torch.eye(U.shape[-1], device=U.device)).max().item() < 1e-4
# 	assert (V.transpose(-1, -2) @ V - torch.eye(V.shape[-1], device=V.device)).max().item() < 1e-4
# 	return UVh.to(compute_device), None
#

def torch_svd_eigh(matrix, rank=None):
	if rank is None: rank = min(matrix.shape[-2:])
	dim_1, dim_2 = matrix.shape[-2], matrix.shape[-1]
	kk = 1
	kkk = 1e-10
	if dim_2 > dim_1:
		X = matrix @ matrix.transpose(-1, -2)
		t = X.diagonal(dim1=-1, dim2=-2)
		t += kk
		del t
		S, U = torch.linalg.eigh(X)
		S = S[..., -rank:].flip(-1).sub_(kk).clip_(min=kkk).sqrt_()
		U = U[..., -rank:].flip(-1)
		Vh = (U.transpose(-1, -2) @ matrix).div_(S[..., None])
	else:
		X = matrix.transpose(-1, -2) @ matrix
		t = X.diagonal(dim1=-1, dim2=-2)
		t += kk
		del t
		S, V = torch.linalg.eigh(X)
		S = S[..., -rank:].flip(-1).sub_(kk).clip_(min=kkk).sqrt_()
		V = V[..., -rank:].flip(-1)
		U = (matrix @ V).div_(S[..., None, :])
		Vh = V.transpose(-1, -2)
	# assert torch.linalg.norm(U * S[..., None, :] @ Vh - matrix) / torch.linalg.norm(matrix) < 1e-6
	assert torch.linalg.norm(U.transpose(-1, -2) @ U - torch.eye(U.shape[-1], device=U.device)) / len(matrix) < 1e-6
	assert torch.linalg.norm(Vh @ Vh.transpose(-1, -2) - torch.eye(Vh.shape[-2], device=Vh.device)) / len(matrix) < 1e-6
	return U, S, Vh