import math, itertools, sys
from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd

import torch
from opt_einsum import contract
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_factor(f, A, B):
	t = A.diagonal(dim1=-1, dim2=-2)
	t += 1e-10
	f[:] = torch.linalg.solve(A, B).T
	


def balance_norm(factors):
	norm = 1
	for f in factors:
		norm = norm * torch.norm(f, dim=0)
	even_split = (norm + 1e-15)#.pow(1.0 / len(factors))
	for i in range(len(factors)):
		factors[i] = factors[i] / (torch.norm(factors[i], dim=0) + 1e-15) #* even_split
	factors[-1] = factors[-1] * even_split

def parafac(
		X, rank, n_iter_max=100, init=None, verbose=False,
		common_factor=None,
):
	context = dict(device=device, dtype=torch.float32)
	X = X.to(**context)
	ndim = len(X.shape)
	
	factors = init
	
	factors = [
		factor.to(**context) for factor in factors
	]


	def calc_A(factor, i):
		# A = torch.full([rank] * 2, 1. if share_factors[i] == 'shared' else (1.+l2_reg), **context)
		A = torch.full([rank] * 2, 1., **context)
		for j, f in enumerate(factor):
			if j != i: A.mul_(f.T @ f)
		return A

	formula_B = [
		','.join(['ijk'] + ['ijk'[j] + 'r' for j in range(ndim) if j != i]) +
		'->' + (('ijk'[i] + 'r'))
		for i in range(ndim)
	]
	def calc_B(factor, i, out=None):
		B = torch.zeros(factor[i].shape, **context)
		B += contract(
			formula_B[i],
			X,
			*[factor[_] for _ in range(len(factor)) if _ != i],
		)
		return B


	balance_norm(factors)

	history = []
	if verbose: pbar = trange(n_iter_max)
	else: pbar = range(n_iter_max)

	iiter = None
	for iiter in pbar:
		iiter += 1
		for i in range(len(factors)):
			A = calc_A(factors, i)
			B = calc_B(factors, i)
			update_factor(factors[i], A, B.T)
			

		loss_recon, loss_reg, loss_x, loss_norm = 0, 0, 0, 0
		
		Xhat = contract('ir,jr,kr->ijk', *factors)
		loss_recon += torch.linalg.norm(X - Xhat).square_().item()
		loss_norm += torch.linalg.norm(Xhat).square_().item()
		
		loss_x += Xhat.mul(X).sum().item()
		loss = loss_recon

		history.append({
			'loss recon': loss_recon,
			'loss': loss,
			'loss x': loss_x,
			'loss norm': loss_norm,
		})

		if len(history) < 2: wdiff = np.nan
		else: wdiff = (history[-2]['loss'] - history[-1]['loss']) / history[-2]['loss']

		if isinstance(pbar, tqdm):
			pbar.set_description(
				f"loss:{loss:.2e} = {loss_recon:.2e} + {loss_reg:.2e} "
				# f"norm:{norm_target.min().item():.2e} {norm_target.max().item():.2e} "
				f"%diff:{wdiff:.2e}"
			)
			pbar.update(1)

		if wdiff < 1e-5: break
		balance_norm(factors)
		
	del X
	print(f"{len(history)}, {wdiff:.1e};", end=' ')
	return factors, history[-1]['loss norm'], history[-1]['loss x']

