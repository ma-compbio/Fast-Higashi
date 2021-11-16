import math, itertools, sys
from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd

import torch
from opt_einsum import contract

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_factor(f, i, A, B, nonnegative):
	if not nonnegative[i]:
		t = A.diagonal(dim1=-1, dim2=-2)
		t += 1e-10
		if isinstance(f, list):
			for f_store, f_new in zip(f, torch.linalg.solve(A, B).transpose(-1, -2)):
				f_store[:] = f_new
		else:
			f[:] = torch.linalg.solve(A, B).T
	elif all(nonnegative) and B.min() >= 0:
		assert A.min() >= 0
		assert B.min() >= 0
		for _ in range(1):
			f.mul_(B).div_((f @ A).add_(1e-10))
	else:
		assert A.min() >= 0
		Apos = A.clip(min=0)
		Aneg = A.clip_(max=0).neg_()
		Bpos = B.clip(min=0)
		Bneg = B.clip_(max=0).neg_()
		for _ in range(1):
			f_last = f.clone()
			f.mul_(
				f.matmul(Aneg).add_(Bpos).div_(
					f.matmul(Apos).add_(Bneg)
				).sqrt_()
					# .clip_(min=1e-3, max=1e3)
			)
			assert (f < 1e3).all()
		del Apos, Aneg, Bpos, Bneg

	if isinstance(f, list):
		assert not any(ff.isnan().any() for ff in f)
	else:
		assert not f.isnan().any()
	if nonnegative[i]:
		assert f.min() >= 0, f.min()
		f.clip_(min=1e-5)
		assert not f.isnan().any()


def balance_norm(ndim, rank, factors, share_factors, context):
	for factor, share_mode in zip(factors, share_factors):
		assert len(set(map(id, factor))) == (1 if share_mode == 'shared' else len(factors[1]))
		# print(share_mode, len(set(map(id, factor))))

	norm_all = torch.empty((len(factors), len(factors[0]), rank), **context)
	for factor, norm in zip(factors, norm_all):
		for f, n in zip(factor, norm):
			torch.linalg.norm(f, axis=0, out=n)
	num_shared = sum(share_factors != 'free')
	if num_shared > 0:
		norm_target_shared = norm_all[:, 0].pow(1. / ndim).prod(0, keepdim=False)
		for factor, norm in itertools.compress(zip(factors, norm_all), share_factors == 'shared'):
			assert factor[0].shape[1:] == norm[0].shape == norm_target_shared.shape, \
				(factor[0].shape, norm[0].shape, norm_target_shared.shape)
			factor[0].mul_(norm_target_shared / norm[0])
			for f in factor:
				assert torch.linalg.norm(f, axis=0).sub_(norm_target_shared).abs_().max() < 1e-3
	else: norm_target_shared = None
	if num_shared < ndim:
		norm_target = norm_all.prod(0, keepdim=False)
		if num_shared > 0: norm_target.div_(norm_target_shared.pow(num_shared))
		norm_target.pow_(1. / (ndim - num_shared))
		for factor, norm in itertools.compress(zip(factors, norm_all), share_factors == 'free'):
			for f, n, t in zip(factor, norm, norm_target):
				assert f.shape[1:] == n.shape == t.shape, (f.shape, n.shape, t.shape)
				f.mul_(t / n)

	assert not any(f.isnan().any().item() for f in itertools.chain.from_iterable(factors))

	return norm_all.prod(0).div_(3).cpu()


def parafac(
		X, rank, n_iter_max=100, init=None, verbose=False, l2_reg=0.0,
		common_factor=None, first_init=False, nonnegative=None,
		niter_inner=1, share_factors=None,
		gamma=0.,
		# opt_order=(2, 1, 0),
		opt_order=(0, 1, 2),
):
	context = dict(device=device, dtype=torch.float32)
	X = X.to(**context)
	ndim = len(X.shape)-1
	if nonnegative is None: nonnegative = [False] * ndim
	nonnegative = np.array(nonnegative)
	if share_factors is None: share_factors = ['free']*ndim
	share_factors = np.array(share_factors)
	num_data = X.shape[1]
	gamma = gamma / num_data
	if num_data == 1 or all(share_factors == 'shared'): l2_reg = 0
	factors = init
	assert set(share_factors) <= {'free', 'shared'}
	for factor, share_mode in zip(factors, share_factors):
		assert len(set(map(id, factor))) == (1 if share_mode == 'shared' else num_data)
	factors = [
		[f.to(**context) for f in factor]
		if share_mode != 'shared' else
		[factor[0].to(**context)] * len(factor)
		for factor, share_mode in zip(factors, share_factors)
	]
	for factor, share_mode in zip(factors, share_factors):
		assert len(set(map(id, factor))) == (1 if share_mode == 'shared' else num_data)
	factors_prev = [
		[f.clone() for f in factor]
		if share_mode != 'shared' else
		[factor[0].clone()] * len(factor)
		for factor, share_mode in zip(factors, share_factors)
	]
	for factor in itertools.compress(zip(*factors), nonnegative):
		for f in factor:
			assert (f.min() >= 0).all().item()

	def update_common_factor():
		common_factor.zero_()
		for factor in zip(*factors):
			common_factor.add_(contract('ir,jr,kr->ijk', *factor))
		common_factor.div_(num_data)

	def transpose_B(): return not any(nonnegative)

	def calc_A(factor, i):
		# A = torch.full([rank] * 2, 1. if share_factors[i] == 'shared' else (1.+l2_reg), **context)
		A = torch.full([rank] * 2, 1.+l2_reg, **context)
		for j, f in enumerate(factor):
			if j != i: A.mul_(f.T @ f)
		return A

	formula_B = [
		','.join(['ijk'] + ['ijk'[j] + 'r' for j in range(ndim) if j != i]) +
		'->' + (('r' + 'ijk'[i]) if transpose_B() else ('ijk'[i] + 'r'))
		for i in range(ndim)
	]
	def calc_B(dl_index, factor, i, out=None):
		B = torch.zeros(factor[i].shape[::-1 if transpose_B() else 1], **context) if out is None else out
		del out
		B += contract(
			formula_B[i],
			X[:, dl_index, :, :].add(common_factor, alpha=l2_reg)
			# if l2_reg > 0 and share_factors[i] == 'free' else
			if l2_reg > 0 else
			X[:, dl_index, :, :],
			*[factor[_] for _ in range(len(factor)) if _ != i],
		)
		return B

	if first_init:
		common_factor[:] = torch.mean(X, dim=1)
	else:
		update_common_factor()
		pass

	balance_norm(ndim, rank, factors, share_factors, context=context)

	history = []
	if verbose: pbar = trange(n_iter_max)
	else: pbar = range(n_iter_max)

	iiter = None
	for iiter in pbar:
		iiter += 1
		for i in opt_order:
			if share_factors[i] == 'free':
				B = torch.zeros((num_data, ) + tuple(factors[i][0].shape[::-1 if transpose_B() else 1]), **context)
				A = torch.empty([num_data, rank, rank], **context)
				for dl_index in range(num_data):
					factor = [factors[j][dl_index] for j in range(3)]
					A[dl_index] = calc_A(factor, i)
					calc_B(dl_index, factor, i, out=B[dl_index])
				update_factor([factors[i][dl_index] for dl_index in range(num_data)], i, A, B, nonnegative)
				del A, B
				num_data == 1 or update_common_factor()
			elif share_factors[i] == 'shared':
				fi = factors[i][0]
				B = torch.zeros(fi.shape[::-1 if transpose_B() else 1], **context)
				A = torch.zeros([rank, rank], **context)
				for dl_index, factor in enumerate(zip(*factors)):
					A += calc_A(factor, i)
					calc_B(dl_index, factor, i, out=B)
				update_factor(fi, i, A, B, nonnegative)
				del A, B
				num_data == 1 or update_common_factor()
			else: raise NotImplementedError

		norm_target = balance_norm(ndim, rank, factors, share_factors, context=context)

		loss_recon, loss_reg, loss_x, loss_norm = 0, 0, 0, 0
		for dl_index, factor in zip(range(num_data), zip(*factors)):
			Xhat = contract('ir,jr,kr->ijk', *factor)
			loss_recon += torch.linalg.norm(X[:, dl_index] - Xhat).square_().item()
			loss_norm += torch.linalg.norm(Xhat).square_().item()
			if l2_reg > 0:
				loss_reg += torch.linalg.norm(common_factor - Xhat).square_().item()
			loss_x += Xhat.mul(X[:, dl_index]).sum().item()
		loss = loss_recon + l2_reg*loss_reg

		history.append({
			'loss recon': loss_recon,
			'loss reg': loss_reg,
			'loss': loss,
			'loss x': loss_x,
			'loss norm': loss_norm,
		})

		if len(history) < 2: wdiff = np.nan
		else: wdiff = (history[-2]['loss'] - history[-1]['loss']) / history[-2]['loss']

		# assert not wdiff < -1e-3, wdiff

		if wdiff < 0:
			for f, g in zip(itertools.chain.from_iterable(factors), itertools.chain.from_iterable(factors_prev)):
				f.copy_(g)
		else:
			for f, g in zip(itertools.chain.from_iterable(factors), itertools.chain.from_iterable(factors_prev)):
				g.copy_(f)

		if isinstance(pbar, tqdm):
			pbar.set_description(
				f"loss:{loss:.2e} = {loss_recon:.2e} + {loss_reg:.2e} "
				# f"norm:{norm_target.min().item():.2e} {norm_target.max().item():.2e} "
				f"%diff:{wdiff:.2e}"
			)
			pbar.update(1)

		if wdiff < 1e-5: break

	# balance_norm(ndim, rank, factors, share_factors, context=context)
	# history = pd.DataFrame(history)
	del X
	print(f"{len(history)}, {wdiff:.1e};", end=' ')
	return factors, common_factor, history[-1]['loss reg'], history[-1]['loss norm'], history[-1]['loss x']

