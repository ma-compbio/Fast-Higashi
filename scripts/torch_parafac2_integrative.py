import math, time, itertools, gc, os, pickle, sys, copy
import numpy as np

import torch
from opt_einsum import contract
from tqdm.auto import tqdm, trange

from sklearn.neighbors import NearestNeighbors
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix
import torch.nn.functional as F

from torch_parafac_integrative import parafac
from util import shift_csr, shift_coo, trim_sparse

torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_norm(tensor_slices):
	norms = []
	for block_index, tensor_slice in enumerate(tensor_slices):
		temp = 0
		if isinstance(tensor_slice, list):
			for dim1_index, tensor_s in enumerate(tensor_slice):
				tensor_s = tensor_s.to(device).to_dense()
				temp += (torch.sum(tensor_s ** 2)).sum()
		elif type(tensor_slice).__name__ == 'Sparse':
			temp = np.linalg.norm(tensor_slice.values)**2
		else: raise NotImplementedError
		temp = math.sqrt(float(temp))
		norms.append(temp)
	return torch.as_tensor(norms)

# Code from scHiCluster
@torch.no_grad()
def neighbor_ave_gpu(A, pad, device):
	if pad == 0:
		return torch.from_numpy(A).float().to(device)
	ll = pad * 2 + 1
	# conv_filter = torch.as_tensor([[0.25, 0.5, 0.25],[0.5, 1, 0.5], [0.25, 0.5, 0.25]]).float().to(device)[None, None, :, :]
	conv_filter = torch.ones(1, 1, ll, ll).to(device)
	B = F.conv2d(A[:, None, :, :].float().to(device), conv_filter, padding=pad * 2)
	B = B[:, 0, pad:-pad, pad:-pad]
	return (B / (ll * ll)).to(A.device)


# Code from scHiCluster
def batch_random_walk_gpu(A, rp, epochs=60, device=None):
	ngene = A.shape[1]
	P = (A / (torch.sum(A, dim=1, keepdim=True) + 1e-15)).to(device)
	Q = torch.eye(ngene).to(device)[None]
	Q = Q.repeat(A.shape[0], 1, 1)
	I = torch.eye(ngene).to(device)[None]
	for i in range(epochs):
		Q_new = (1 - rp) * I + rp * torch.bmm(Q, P)
		delta = torch.norm(Q - Q_new, 2)
		Q = Q_new
		if delta  < 0.01 * A.shape[0]:
			break
	return Q.to(A.device)


def incomplete_random_walk(A):
	# a shape of (bs, #bin1, #bin2)
	local_sim = torch.bmm(A, A.permute(0, 2, 1)) # size of (bs, #bin1, #bin1)
	index = torch.arange(A.shape[1])
	local_sim[:, index, index] = 0
	zero_cov_index = (torch.sum(local_sim, dim=2) == 0).nonzero(as_tuple=True)
	if len(zero_cov_index[0]) > 0:
		local_sim[zero_cov_index[0], zero_cov_index[1], zero_cov_index[1]] = 1
	local_sim = batch_random_walk_gpu(local_sim, 0.5, 10, device=device)
	local_sim[:, index, index] = 0
	zero_cov_index = (torch.sum(local_sim, dim=2) == 0).nonzero(as_tuple=True)
	if len(zero_cov_index[0]) > 0:
		local_sim[zero_cov_index[0], zero_cov_index[1], zero_cov_index[1]] = 1
	local_sim = local_sim / torch.sum(local_sim, dim=2, keepdim=True)
	A = torch.baddbmm(A, local_sim, A, beta=0.5, alpha=0.5, out=A)
	return A


def rwr(y, context):
	if y.shape[-1] < 7000:
		return incomplete_random_walk(y.permute(2, 0, 1)).permute(1, 2, 0)
	else:
		bs = 500
		for i in range(0, y.shape[-1], bs):
			y[:, :, slice(i, i+bs)] = incomplete_random_walk(y[:, :, slice(i, i+bs)].to(context['device']).permute(2, 0, 1)).permute(1, 2, 0).to(y.device)
		return y


def conv(tensor_slices, context):
	y = tensor_slices
	if y.shape[-1] < 7000:
		return neighbor_ave_gpu(y.permute(2, 0, 1), 1, device=device).permute(1, 2, 0)
	else:
		bs = 500
		for i in range(0, y.shape[-1], bs):
			y[:, :, slice(i, i+bs)] = neighbor_ave_gpu(y[:, :, slice(i, i+bs)].to(context['device']).permute(2, 0, 1), 1, device=device).permute(1, 2, 0).to(y.device)
		return y


def project2orthogonal(matrix, rank=None, old=None):
	dim_1, dim_2 = matrix.shape[-2], matrix.shape[-1]
	if rank is None: rank = min(matrix.shape[-2:])
	try:
		U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
		return U[..., :rank] @ Vh[..., :rank, :], S[..., :rank]
	except Exception as e:
		# print(e)
		# U, S, Vh = torch.linalg.svd(matrix + torch.randn(matrix.shape)*1e-5, full_matrices=False)
		# return U[..., :rank] @ Vh[..., :rank, :], S
		print(f'error {e}. using eigh, shape = {matrix.shape}')
		# kk = torch.linalg.norm(matrix, dim=(-1, -2)).square_().div_(min(dim_1, dim_2)).div_(5)[..., None]
		kk = 1e-2
		U, S, V, Vh = None, None, None, None
		mode = dim_2 > dim_1
		X = matrix @ matrix.transpose(-1, -2) if mode else matrix.transpose(-1, -2) @ matrix
		t = X.diagonal(dim1=-1, dim2=-2)
		t += kk
		del t
		eigvals, eigvecs = torch.linalg.eigh(X)
		# S = eigvals[..., -rank:].flip(-1).sub_(kk).clip_(min=0).sqrt_()
		# S = torch.linalg.eigvalsh(X)[..., -rank:].flip(-1).sub_(kk).clip_(min=0).sqrt_()
		# S = torch.linalg.svdvals(X)[..., -rank:].flip(-1).sub_(kk).clip_(min=0).sqrt_()
		# S = torch.linalg.svd(X)[1][..., :rank].sub_(kk).clip_(min=0).sqrt_()
		# S = torch.linalg.qr(matrix, mode='r')[1].diagonal(dim1=-1, dim2=-2).abs_()
		eigvecs = eigvecs[..., -rank:].flip(-1)
		if mode:
			U = eigvecs
			Vh = U.transpose(-1, -2) @ matrix
			V, R = torch.linalg.qr(Vh.transpose(-1, -2))
			V = V.mul_(R.diagonal(dim1=-1, dim2=-2).sign()[..., None, :])
			# idx = torch.diag(R) < 1e-2
		else:
			V = eigvecs
			U = matrix @ V
			U, R = torch.linalg.qr(U)
			U = U.mul_(R.diagonal(dim1=-1, dim2=-2).sign()[..., None, :])
		UVh = U @ V.transpose(-1, -2)
		assert (U.transpose(-1, -2) @ U - torch.eye(U.shape[-1], device=U.device)).max().item() < 1e-4
		assert (V.transpose(-1, -2) @ V - torch.eye(V.shape[-1], device=V.device)).max().item() < 1e-4
		return UVh, S


def initialize(
		tensor_slices, rank, size_list, data_list, share_factors, separate_meta_embedding, bs=32,
		do_conv=True, do_rwr=True,
):
	_t = time.perf_counter()
	context = dict(device=device, dtype=torch.float32)
	context_A = dict(device='cpu', dtype=torch.float32)
	# context_A = dict(device=device, dtype=torch.float32)
	cum_size_list = np.concatenate([[0], np.cumsum(size_list)])
	meta_embedding = torch.empty([tensor_slices[0].shape[2], rank], **context)
	A_list = [
		# [torch.ones([tensor_slice.shape[0], size], **context_A) for _ in data_list]
		[torch.randn([tensor_slice.shape[0], size], **context_A)*1e-2 + 1 for _ in data_list]
		if share_factors[0] == 'free' else
		# [torch.ones([tensor_slice.shape[0], size], **context_A)] * len(data_list)
		[torch.randn([tensor_slice.shape[0], size], **context_A)*1e-2 + 1] * len(data_list)
		for tensor_slice, size in zip(tensor_slices, size_list)]
	B_list = [
		# [torch.eye(size, **context) for _ in data_list]
		[torch.eye(size, **context).add_(torch.randn(size, **context), alpha=1e-2) for _ in data_list]
		if share_factors[1] == 'free' else
		# [torch.eye(size, **context)] * len(data_list)
		[torch.eye(size, **context).add_(torch.randn(size, **context), alpha=1e-2)] * len(data_list)
		for tensor_slice, size in zip(tensor_slices, size_list)]
	tensor_slices_batch_chrom = zip(*[
		tensor_slice.split([slc.stop for slc in data_list[:-1]], dim=2)
		for tensor_slice in tensor_slices
	])
	C = torch.empty([tensor_slices[0].shape[-1], sum(size_list)], **context)
	print(f'time elapsed: {time.perf_counter() - _t:.2f}')
	sys.stdout.flush()
	for tensor_slices_chrom, dl_slice in zip(tensor_slices_batch_chrom, data_list):
		n = dl_slice.stop - dl_slice.start
		# Y = torch.zeros([n, n], device=device, dtype=torch.float32)
		# C = torch.empty([n, sum(size_list)], **context)
		for tensor_slice, size, start, stop in tqdm(zip(
				tensor_slices_chrom, size_list, cum_size_list[:-1], cum_size_list[1:],
		), total=len(size_list), desc="initializing params"):
			# Y.zero_()
			# for i in range(0, len(tensor_slice), bs):
			# 	X = densify(tensor_slice, i, i + bs, do_conv=do_conv, do_rwr=do_rwr, context=dict(device=device))
			# 	# X = tensor_slice[i: i+bs]
			# 	# X = X.to_pytorch(device=device).to_dense()
			# 	X = X.reshape(-1, X.shape[-1])
			# 	Y.addmm_(X.T, X)
			# 	# sparse mm is slow
			# 	# X = X.reshape(-1, X.shape[-1])
			# 	# x = X.to_pytorch(device=device)
			# 	# y = X.permute(1, 0).to_pytorch(device=device)
			# 	# Y.add_(torch.sparse.mm(y, x))
			# 	del X
			# lam = 1.
			# Y[np.diag_indices(len(Y))] += lam
			# eigval, eigvec = torch.linalg.eigh(Y)
			#
			# # C[:, start: stop] = eigvec[:, -size:].mul_(eigval[-size:].sub_(lam).clip_(min=1e-2).sqrt_()).flip(1)
			# C[dl_slice, start: stop] = eigvec[:, -size:].mul_(eigval[-size:].sub_(lam).clip_(min=1e-2).sqrt_()).flip(1)

			
			X = tensor_slice
			X = tensor_slice.reshape(X.shape[0] * X.shape[1], X.shape[-1]).to_csr().T
			from sklearn.decomposition import TruncatedSVD
			svd = TruncatedSVD(n_components=size)
			temp = svd.fit_transform(X)
			# print (temp.shape)
			# temp = np.random.randn(X.shape[-1], size)
			C[dl_slice, start: stop] = torch.from_numpy(temp).float().to(device)
			# del eigval, eigvec
		# del Y
	if separate_meta_embedding:
		D_list = []
		for dl_slice in data_list:
			U, S, Vh = torch.linalg.svd(C[dl_slice], full_matrices=False)
			meta_embedding[dl_slice] = U[:, :rank]
			SVh = Vh[:rank].mul_(S[:rank, None])
			D_list.append([SVh[:, start: stop].clone() for start, stop in zip(cum_size_list[:-1], cum_size_list[1:])])
			del U, S, Vh, SVh
		D_list = list(map(list, zip(*D_list)))
	else:
		U, S, Vh = torch.linalg.svd(C, full_matrices=False)
		meta_embedding = U[:, :rank]
		SVh = Vh[:rank].mul_(S[:rank, None])
		D_list = [
			[SVh[:, start: stop].clone() for dl_slice in data_list]
			for start, stop in zip(cum_size_list[:-1], cum_size_list[1:])
		]
		del U, S, Vh, SVh
	del C
	if tuple(share_factors) == ('free', 'free', 'shared') and False:
		# try to match the sign of singular vectors
		for factors in zip(A_list, B_list, tqdm(D_list)):
			for A, B, D, dl_slice in zip(*factors, data_list):
				U, S, Vh = torch.svd(D.T)
				print(f'singular values: {S.min().item():.2e} {S.max().item():.2e}')
				A.copy_(S[None])
				B.copy_(U)
				D.copy_(Vh)
			D = factors[2][0]
			for d in factors[2][1:]:
				D.add_(d)
				del d
			D.div_(len(factors[2]))
			factors[2][:] = [D] * len(factors[2])
			del D
	elif share_factors[2] == 'shared':
		for Ds in D_list:
			D = Ds[0]
			for d in Ds[1:]: D.add_(d)
			D.div_(len(Ds))
			Ds[:] = [D] * len(Ds)
			del D
	for dl_slice in data_list:
		V = meta_embedding[dl_slice]
		t = V.T @ V
		t[np.diag_indices(len(t))] -= 1
		# assert t.abs().max() < 1e-1
	print(f'time elapsed: {time.perf_counter() - _t:.2f}')
	sys.stdout.flush()
	return None, A_list, B_list, meta_embedding, D_list


def densify(tensor_slices, start, stop, do_conv, do_rwr, context, mixer=None, save_context=None):
	if save_context is None:
		save_context = context
	tensor_slice = tensor_slices[start: stop]
	x = tensor_slice
	dense_gpu = torch.zeros(tuple(x.shape), **save_context)
	# print(time.perf_counter() - _t)
	# method 2.1
	# _t = time.perf_counter()
	indices = tuple(
		torch.tensor(_, dtype=torch.long).pin_memory().to(save_context['device'], non_blocking=True)
		for _ in x.indices
	)
	values = torch.tensor(x.values).pin_memory().to(save_context['device'], non_blocking=True)
	dense_gpu[indices] = values
	tensor_slice = dense_gpu
	if do_conv:
		tensor_slice = conv(tensor_slice, context=dict(device=device))
	if do_rwr:
		tensor_slice = rwr(tensor_slice, context=dict(device=device))
	# if mixer is not None and mixer.knn_model is not None:
	# 	tensor_slice = mixer.transform(tensor_slice)
	# tensor_slice = tensor_slice.clone()
	# mean, std = torch.std_mean(tensor_slice, dim=(-1, -2), unbiased=False, keepdim=True)
	# tensor_slice.sub_(mean)
	# tensor_slice.div_(std.add_(1e-10).pow(.5))
	return tensor_slice


@torch.no_grad()
def _compute_projection_and_project_tensor_slices_batch(
		tensor_slices_list, factors, size_list=None, rank=None,
		projected_tensor_list=None, bs=1, data_list=None, do_conv=True, do_rwr=True,
		separate_meta_embedding=True, mixer=None, share_U=True, mixer_weight=0.,
		n_iter_V=1,
		rec_error_core_norm=None,
		verbose=True,
):
	projection_list, A_list, B_list, D_list, meta_embedding = factors

	# SVD_term = [torch.zeros([rank, slc.stop-slc.start], dtype=torch.float32, device=device) for slc in data_list]
	SVD_term = torch.zeros(meta_embedding.shape[::-1], dtype=torch.float32, device=device)
	rec_error_x_U = np.zeros([len(tensor_slices_list), 1])
	rec_error_x_V = 0
	rec_error_tensor_norm = np.zeros([len(tensor_slices_list), 1])
	for block_index, (tensor_slices, A, B, D, size, projection, projected_tensor_out) in enumerate(zip(
			tqdm(tensor_slices_list), A_list, B_list, D_list, size_list, projection_list, projected_tensor_list
	)):
		## Note, A,B,D are all lists here
		# No matter what, A,B,C lists are all length of data_list to make coding easier
		C = [torch.matmul(meta_embedding, d) for d in D]
		
		# rec_error = 0
		for i in range(0, len(tensor_slices), bs):
			# Fetch and densify the X
			slice_ = slice(i, i + bs)
			# cpu_flag = tensor_slices.shape[-1] > 7000
			cpu_flag = False
			tensor_slice = densify(tensor_slices, i, i+bs, do_conv, do_rwr,
			                       context=dict(device=device),
			                       mixer=mixer, save_context=dict(device=device) if not cpu_flag else dict(device="cpu"))
			rec_error_tensor_norm[block_index] += torch.linalg.norm(tensor_slice).square_().item()
			temp = torch.zeros(tensor_slice.shape[:2] + (B[0].shape[0],), dtype=torch.float32, device=device)
			
			tensor_project_by_U = torch.empty(
				[tensor_slice.shape[0], B[0].shape[0], tensor_slice.shape[2]],
				dtype=torch.float32, device=device if not cpu_flag else 'cpu',
			)
			if share_U:
				for dl_index, dl_slice in enumerate(data_list):
					# if not cpu_flag:
					lhs = contract('ir,jr,kr->ikj', A[dl_index][slice_].to(device), B[dl_index], C[dl_index][dl_slice])
					rhs = tensor_slice[:, :, dl_slice].to(device)
					temp.baddbmm_(rhs, lhs)
					# else:
					# 	bs_c = 4000
					# 	tp = tensor_slice[:, :, dl_slice]
					# 	for i in range(0, tp.shape[-1], bs_c):
					# 		lhs = contract('ir,jr,kr->ikj', A[dl_index][slice_].to(device), B[dl_index],
					# 		               C[dl_index][dl_slice][slice(i, i + bs_c)])
					# 		rhs = tp[:, :, slice(i, i + bs_c)].to(device)
					# 		temp.baddbmm_(rhs, lhs)
							
					del lhs, rhs
				U, S = project2orthogonal(temp)
				
				if S is None:
					rec_error_x_U[block_index] += temp.view(-1).inner(U.view(-1)).item()
				else:
					assert (S.sum().item() - temp.view(-1).inner(U.view(-1)).item()) / S.sum().item() < 1e-5, (
						S.sum().item(), temp.view(-1).inner(U.view(-1)).item(),
						S.sum().item() - temp.view(-1).inner(U.view(-1)).item(),
						(S.sum().item() - temp.view(-1).inner(U.view(-1)).item()) / S.sum().item()
					)
					rec_error_x_U[block_index] += S.sum().item()
				del S
				projection[0][slice_] = U.to(projection[0].device)
				
				# tensor_project_by_U = torch.bmm(U.transpose(-1, -2), tensor_slice)
				if cpu_flag:
					tensor_project_by_U = torch.bmm(U.transpose(-1, -2), tensor_slice.to(device)).cpu()
				else:
					torch.bmm(U.transpose(-1, -2), tensor_slice, out=tensor_project_by_U)
			else:
				for dl_index, dl_slice in enumerate(data_list):
					lhs = contract('ir,jr,kr->ikj', A[dl_index][slice_].to(device), B[dl_index], C[dl_index][dl_slice])
					rhs = tensor_slice[:, :, dl_slice]
					temp[:] = rhs @ lhs
					U, S = project2orthogonal(temp)
					if S is None:
						rec_error_x_U[block_index] += temp.view(-1).inner(U.view(-1)).item()
					else:
						assert (S.sum().item() - temp.view(-1).inner(U.view(-1)).item()) / S.sum().item() < 1e-5, (
							S.sum().item(), temp.view(-1).inner(U.view(-1)).item(),
							S.sum().item() - temp.view(-1).inner(U.view(-1)).item(),
							(S.sum().item() - temp.view(-1).inner(U.view(-1)).item()) / S.sum().item()
						)
						rec_error_x_U[block_index] += S.sum().item()
					del S
					projection[dl_index][slice_] = U.to(projection[dl_index].device)
					tensor_project_by_U[:, :, dl_slice] = U.transpose(-1, -2) @ tensor_slice[:, :, dl_slice]
			del temp, tensor_slice
			
			for dl_index, dl_slice in enumerate(data_list):
				lhs = contract('ir,jr,kr->kij', A[dl_index][slice_].to(device), B[dl_index], D[dl_index])
				rhs = tensor_project_by_U[:, :, dl_slice].to(lhs.device)
				SVD_term[..., dl_slice].addmm_(lhs.reshape(lhs.shape[0], -1), rhs.reshape(-1, rhs.shape[-1]))
				del lhs, rhs
				
			del tensor_project_by_U

		# projection_list[block_index] = projection

	# SVD_term: dim3 * R
	if separate_meta_embedding:
		# SVD_term_nbr = mixer.transform(X=SVD_term, other=meta_embedding.T, normalize=False, weight=20.)
		SVD_term_nbr = SVD_term
		for dl_index, dl_slice in enumerate(tqdm(data_list)):
			meta_embedding[dl_slice], S = project2orthogonal(SVD_term_nbr[..., dl_slice].T, rank=rank)
			print (S.shape)
			explained_ratio = S[1:] / torch.sum(S[1:])
			ratio1 = torch.cumsum(explained_ratio, dim=-1)
			# print("explained", ratio1, torch.sum(ratio1 < 0.9), torch.sum(ratio1 < 0.95), torch.sum(ratio1 < 0.99))
			# rec_error_x_V += S.sum().item()
			rec_error_x_V += meta_embedding[dl_slice].mul(SVD_term[..., dl_slice].T).sum().item()
	else:
		SVD_term = SVD_term.T
		do_mixing = mixer.A_adj is not None and mixer_weight > 0
		if not do_mixing:
			t, S = project2orthogonal(SVD_term, rank=rank)
			explained_ratio = S / torch.sum(S)
			ratio1 = torch.cumsum(explained_ratio)
			print("explained", ratio1, torch.sum(ratio1 < 0.9), torch.sum(ratio1 < 0.95), torch.sum(ratio1 < 0.99))
			
			tqdm.write(f'meta embedding {torch.linalg.norm(t - meta_embedding).item():.2e}')
			meta_embedding[:] = t
			rec_error_x_V = S.sum().item()
		else:
			step_size = 8. if mixer.mode_nn == 'mnn' else 1.
			step_size_lb = .5
			# weight = mixer_weight * torch.linalg.norm(SVD_term, ord='fro')
			weight = mixer_weight * sum(tensor.numel() for tensor in tensor_slices_list) / meta_embedding.numel()
			def calc_coef(meta_embedding):
				# regularization on V
				feat = meta_embedding
				eigval_p = 1.
				# regularization on VD
				# P = get_embedding_projection(D_list, data_list)
				# feat = torch.empty_like(meta_embedding)
				# eigval_p = 0
				# for dl_slice, p in zip(data_list, P):
				# 	p = p @ p.T
				# 	eigval_p = max(eigval_p, torch.linalg.eigh(p)[0].max().item())
				# 	feat[dl_slice] = meta_embedding[dl_slice] @ p
				# #
				reg_nbr, eigval_Aw, loss = mixer.transform(feat=feat)
				loss = loss * weight / 2 - SVD_term.mul(meta_embedding).sum().item()
				loss_all = loss + (rec_error_tensor_norm.sum().item() + rec_error_core_norm.sum().item()) * .5
				# tqdm.write(f'eigval p = {eigval_p:.2e}, loss = {loss:.4e}')
				reg_nbr = reg_nbr.neg_()
				return reg_nbr, eigval_Aw * eigval_p, loss, loss_all
			_t = time.perf_counter()
			reg_nbr, eigval, loss, loss_all = calc_coef(meta_embedding)
			print(time.perf_counter() - _t)
			if verbose: pbar = trange(n_iter_V)
			else: pbar = range(n_iter_V)
			for i in pbar:
				while True:
					t = reg_nbr.add(meta_embedding, alpha=eigval / step_size)
					SVD_term_nbr = SVD_term.add(t, alpha=weight)
					meta_embedding_new, S = project2orthogonal(SVD_term_nbr, rank=rank)
					loss_approx = meta_embedding.mul(reg_nbr).sum().item() / 2 - \
						meta_embedding_new.mul(reg_nbr).sum().item() + \
						torch.linalg.norm(meta_embedding_new-meta_embedding).square().item() * eigval / step_size / 2
					loss_approx = loss_approx * weight - SVD_term.mul(meta_embedding).sum().item()
					reg_nbr_new, eigval_new, loss_new, loss_all_new = calc_coef(meta_embedding_new)
					# print(f'{loss:.2e}, {loss_approx:.2e}, {loss_new:.2e}')
					# wdiff = (loss_new - loss) / np.abs(loss)
					wdiff = (loss_all_new - loss_all) / loss_all
					if wdiff > 0:
						tqdm.write(
							f'At iter {i}, '
							# f'loss = {loss:.2e} {wdiff:+.2e} -> {loss_new:.2e}, '
							f'loss = {loss_all:.2e} {wdiff:+.2e} -> {loss_all_new:.2e}, '
							f'step size = {step_size:.2e}')
						if step_size == step_size_lb: break
						step_size *= .5
						step_size = max(step_size, step_size_lb)
						continue
					else:
						break
				if isinstance(pbar, tqdm):
					pbar.set_description(
						f'meta embedding {torch.linalg.norm(meta_embedding_new - meta_embedding).item():.2e}\t'
						f'loss = {loss_all:.2e} {wdiff:+.2e} -> {loss_all_new:.2e}\t'
						f'step size = {step_size:.1f}'
					)
					pbar.update(1)
				if wdiff < 0:
					reg_nbr, eigval, loss, loss_all = reg_nbr_new, eigval_new, loss_new, loss_all_new
					meta_embedding[:] = meta_embedding_new
				if wdiff > -1e-8: break
			rec_error_x_V = meta_embedding.mul(SVD_term).sum().item()

	for block_index, (tensor_slices, projection) in enumerate(zip(tqdm(tensor_slices_list), projection_list)):
		for i in range(0, len(tensor_slices), bs):
			slice_ = slice(i, i + bs)
			tensor_slice = densify(tensor_slices, i, i+bs, do_conv, do_rwr, context=dict(device=device), mixer=mixer)
			
			einsten = "ijk,km, ijl -> ilm"
			for dl_index, dl_slice in enumerate(data_list):
				projected = contract(
					einsten, tensor_slice[:, :, dl_slice], meta_embedding[dl_slice],
					projection[dl_index][slice_].to(meta_embedding.device))
				projected_tensor_list[block_index][slice_, dl_index, :, :] = projected.to(
					projected_tensor_list[block_index].device)

	# assert rec_error_x_U.sum() < rec_error_x_V * 1.0001, \
	# 	(rec_error_x_U, rec_error_x_V, rec_error_x_U.sum() - rec_error_x_V)

	# rec_error_list_U = rec_error_tensor_norm + rec_error_core_norm - 2 * rec_error_x_U
	# rec_error_V = rec_error_tensor_norm.sum() + rec_error_core_norm.sum() - 2 * rec_error_x_V.sum()
	
	# return meta_embedding, projection_list, projected_tensor_list, rec_error_list_U, rec_error_V
	return meta_embedding, projection_list, projected_tensor_list, rec_error_x_U, rec_error_x_V, rec_error_tensor_norm


@torch.no_grad()
def fast_higashi_integrative(
		tensor_slices, rank, n_iter_max=2000, l2_reg=0.,
		tol=1e-8, random_state=None, verbose=False, return_errors=False,
		n_iter_parafac=5, size_list=None, init=None, data_list=None,
		share_factors=None,
		output_dir=None,
		gamma=0.,
		nonnegative=None,
		do_conv=True,
		do_rwr=False,
		label_list=None,
		separate_meta_embedding=True,
		kwargs_mixer=None,
		share_U=True,
		mixer_weight=0.,
		i_iter_mixer=0,
		n_iter_V=1,
):
	if size_list is None: size_list = []
	if data_list is None: data_list = [slice(None)]
	if share_factors is None: share_factors = ['free']*3
	share_factors = np.array(share_factors)
	if nonnegative is None: nonnegative = [False]*3
	nonnegative = np.array(nonnegative)
	print ("Start initializing")
	sys.stdout.flush()


	weights, A_list, B_list, meta_embedding, D_list = initialize(
		tensor_slices, rank, size_list, data_list, share_factors,
		bs=32,
		separate_meta_embedding=separate_meta_embedding,
		do_conv=do_conv,
		do_rwr=do_rwr,
	)
	
	factor_count = 0
	# for factor in zip(A_list, B_list, D_list):
	# 	factor_count += 1
	# 	ABD_count = 0
	# 	for f in itertools.chain.from_iterable(factor):
	# 		ABD_count += 1
	# 		print(factor_count, ABD_count, f.shape, f.device)
	# 		mask = f < 1e-5
	# 		f = f.mul_(~mask)
	# 		f += torch.randn(f.shape, device=f.device, dtype=f.dtype).div_(10).add_(1).clip(1e-2).mul_(mask)
	#
	# factor_count=0
	# for factor in itertools.compress([A_list, B_list, D_list], nonnegative):
	# 	factor_count += 1
	# 	ABD_count = 0
	# 	for f in itertools.chain.from_iterable(factor):
	# 		ABD_count += 1
	# 		print (factor_count, ABD_count, f.shape, f.device)
	# 		mask = f < 1e-5
	# 		f = f.mul_(~mask)
	# 		f += torch.randn(f.shape, device=f.device, dtype=f.dtype).div_(10).add_(1).clip(1e-2).mul_(mask)
	gc.collect()
	common_factor_list = []
	print ("Copying factors for different dataset")
	sys.stdout.flush()
	for i in range(len(A_list)):
		common_factor_list.append(torch.zeros(A_list[i][0].shape[0], size_list[i], rank))


	meta_embedding = meta_embedding.to(device)

	rec_errors = []
	rec_errors_total = []
	# Calculate the norm, which will be used for scale the errors
	norm_tensor = get_norm(tensor_slices).cpu().numpy()
	norm_tensor_all = np.linalg.norm(norm_tensor)

	# mixer = MixNeighbor(
	# 	n_cells=len(meta_embedding),
	# 	n_batches=len(data_list),
	# 	**({} if kwargs_mixer is None else kwargs_mixer),
	# )
	# mixer = MixNeighborMNN(**kwargs_mixer)
	mixer = None

	projection_list = [
		# torch.empty(tuple(tensor_slice.shape[:2]) + (size,), dtype=torch.float32).pin_memory()
		[torch.empty(tuple(tensor_slice.shape[:2]) + (size,), dtype=torch.float32).pin_memory()] * len(data_list)
		if share_U else
		[torch.empty(tuple(tensor_slice.shape[:2]) + (size,), dtype=torch.float32).pin_memory() for _ in data_list]
		for tensor_slice, size in zip(tensor_slices, size_list)
	]
	# Note the batch_id dim is at dim 1
	projected_tensor_list = [
		torch.empty([len(tensor_slice), len(data_list), size, rank], dtype=torch.float32).pin_memory()
		for tensor_slice, size in zip(tensor_slices, size_list)
	]

	rec_error_tensor_norm = np.zeros([len(tensor_slices), 1])
	rec_error_core_norm = np.zeros([len(tensor_slices), 1])
	bs = 4
	for tensor, factors, re_t, re_c in zip(
			tqdm(tensor_slices), zip(A_list, B_list, D_list), rec_error_tensor_norm, rec_error_core_norm):
		for i in range(0, len(tensor), bs):
			slice_ = slice(i, i + bs)
			re_t[:] += torch.linalg.norm(
				densify(tensor, i, i+bs, do_conv, do_rwr, context=dict(device=device))).square_().item()
			for A, B, D in zip(*factors):
				c = contract('ir,jr,kr->kij', A[slice_].to(device), B, D)
				re_c[:] += torch.linalg.norm(c).square_().item()
				del c

	iteration = -1
	for iteration in range(n_iter_max):
		# if verbose:
		print("Starting iteration", iteration)
		sys.stdout.flush()

		start_time = time.time()

		# meta_embedding, projection_list, projected_tensor_list, rec_error_by_block_U, rec_error_V = \
		meta_embedding, projection_list, projected_tensor_list, rec_error_x_U, rec_error_x_V, rec_error_tensor_norm = \
			_compute_projection_and_project_tensor_slices_batch(
				tensor_slices, [projection_list, A_list, B_list, D_list, meta_embedding],
				size_list, rank, projected_tensor_list,
				bs=64,
				data_list=data_list,
				do_conv=do_conv,
				do_rwr=do_rwr,
				separate_meta_embedding=separate_meta_embedding,
				mixer=mixer,
				share_U=share_U,
				mixer_weight=mixer_weight,
				n_iter_V=n_iter_V,
				rec_error_core_norm=rec_error_core_norm,
			)
		rec_error_by_block_U = rec_error_tensor_norm + rec_error_core_norm - 2 * rec_error_x_U
		rec_error_V = rec_error_tensor_norm.sum() + rec_error_core_norm.sum() - 2 * rec_error_x_V
		del rec_error_x_U, rec_error_x_V

		loss_reg_by_block = np.zeros([len(tensor_slices), 1])
		rec_error_x_core = np.zeros([len(tensor_slices), 1])

		# Run parafac on projected tensors (size of (dim1, size, rank))
		for block_index in trange(len(tensor_slices)):
			temp_factors = [A_list[block_index], B_list[block_index], D_list[block_index]]

			factors, d, loss_reg, core_norm, loss_x = parafac(
				projected_tensor_list[block_index],
				rank=size_list[block_index],
				init=temp_factors,
				# n_iter_max=n_iter_parafac if iteration > 0 else 5, verbose=verbose, l2_reg=l2_reg if iteration > 0 else 0.0,
				n_iter_max=n_iter_parafac,
				verbose=False,
				# verbose=verbose,
				l2_reg=l2_reg,
				common_factor=common_factor_list[block_index].to(device),
				share_factors=share_factors,
				gamma=gamma,
				nonnegative=nonnegative,
			)

			rec_error_core_norm[block_index] = core_norm
			rec_error_x_core[block_index] = loss_x
			loss_reg_by_block[block_index] = loss_reg

			def save_to_device(model_param, temp_var):
				if model_param is None or model_param is temp_var: return
				model_param[:] = temp_var.to(model_param.device)
			for f_store, f_new, share_mode in zip(temp_factors, factors, share_factors):
				if share_mode == 'shared':
					save_to_device(f_store[0], f_new[0])
					assert len(set(map(id, f_store))) == 1
					assert len(set(map(id, f_new))) == 1
				else:
					# list(itertools.starmap(save_to_device, zip(f_store, f_new)))
					for _ in zip(f_store, f_new): save_to_device(*_)
				del f_store, f_new
			save_to_device(common_factor_list[block_index], d)
			del factors, d
		print()
		rec_error_by_block_core = rec_error_tensor_norm + rec_error_core_norm - 2 * rec_error_x_core
		assert rec_error_by_block_core.sum() < rec_error_V * 1.005, (
			rec_error_by_block_core.sum(), rec_error_V,
			rec_error_by_block_core.sum() - rec_error_V,
		)

		rec_error = np.sqrt(rec_error_V.sum()) / norm_tensor_all
		rec_errors_total.append(rec_error)
		rec_error_by_block = np.sqrt(rec_error_by_block_U.ravel()) / norm_tensor
		rec_errors.append(rec_error_by_block)
		if iteration >= 1:
			differences = (rec_errors[-2] ** 2 - rec_errors[-1] ** 2) / (rec_errors[-2] ** 2)
			# if verbose:
			print(
				f"PARAFAC2 re={rec_error:.3f} "
				f"{(rec_errors_total[-2]-rec_errors_total[-1])/rec_errors_total[-2]:.2e} "
				f"variation min{differences.min().item():.1e} at chrom {differences.argmin().item():d}, "
				f"max{differences.max().item():.1e} at chrom {differences.argmax().item():d}",
				f"takes {time.time() - start_time:.1f}s"
			)
			if iteration >= 3 and tol > 0 and differences.max() < tol:
				# if verbose:
				print('converged in {} iterations.'.format(iteration))
				break
		else:
			# if verbose:
			print(
				f"PARAFAC2 re={rec_error:.3f} "
				f"takes {time.time() - start_time:.1f}s"
			)
		sys.stdout.flush()

	parafac2_tensor = (weights, (A_list, B_list, D_list, meta_embedding), projection_list)
	
	error_list = []
	dim_list = []
	
	if return_errors:
		return parafac2_tensor, rec_errors, error_list, dim_list
	else:
		return parafac2_tensor, error_list, dim_list


