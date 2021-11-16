import copy, time

import numpy as np
from scipy.sparse._sparsetools import coo_tocsr
from scipy.sparse import coo_matrix, csr_matrix

import torch


#TODO: use numpy.unravel_index


def calc_offset(dims):
	offset = np.ones_like(dims)
	offset[:-1] = np.cumprod(dims[:0:-1])[::-1]
	return offset


# May consider negative values in the future.
def get_minimum_dtype(max_value):
	for dtype in [np.int8, np.int16, np.int32, np.int64]:
		if max_value <= np.iinfo(dtype).max: return dtype
	raise ValueError


def unfold(indices, shape):
	offset = calc_offset(shape)
	indices = indices.astype(shape.dtype, copy=False)
	indices = offset @ indices
	# indices = offset[:-1] @ indices[:-1] + indices[-1]
	# indices = (indices * offset[:, None]).sum(0)
	assert (indices >= 0).all()
	assert (indices < np.prod(shape)).all()
	return indices


def fold(indices, shape, out=None):
	offset = calc_offset(shape)
	# indices = indices[None] // offset[:, None]
	# indices[1:] %= offset[:-1, None]
	if out is None: indices_new = np.empty([len(shape), len(indices)], dtype=indices.dtype)
	else: indices_new = out
	del out
	# out[0] = indices
	# indices = out
	# del out
	# for i in range(len(shape)-1):
	# 	indices[i], indices[i+1] = divmod(indices[i], offset[i])
	for i in range(len(shape)-1):
		indices_new[i], indices = divmod(indices, offset[i])
	indices_new[-1] = indices
	return indices_new


# Some memory are shared. Need caution.
class Sparse:
	def __init__(
			self, indices, values, shape, indptr=None, copy=True, verbose=False,
	):
		self.ndim = len(shape)
		self.indices = np.array(indices, copy=copy)
		self.values = np.array(values, copy=copy)
		self.shape = np.array(shape, copy=copy)
		self.indptr = None if indptr is None else np.array(indptr, copy=copy)
		self.verbose = verbose
		assert self.indices.shape == (self.ndim, len(self.values))
		assert (self.indices >= 0).all()
		assert (self.indices < self.shape[:, None]).all(), (self.indices.max(1), self.shape)

	def permute(self, *dims, inplace=False):
		assert tuple(sorted(dims)) == tuple(range(self.ndim))
		dims = np.array(dims)
		indices = self.indices[dims]
		shape = self.shape[dims]
		indptr = self.indptr if dims[0] == 0 else None
		if inplace:
			self.indices = indices
			self.shape = shape
			self.indptr = indptr
			return self
		else:
			return Sparse(indices, self.values, shape, indptr, verbose=self.verbose)

	def reshape(self, *dims, inplace=False):
		dims = np.array(dims)
		if (dims == -1).any():
			assert (dims == -1).sum() <= 1, dims
			assert self.shape.prod() % dims.prod() == 0
			dims[dims == -1] = self.shape.prod(dtype=self.shape.dtype) // -dims.prod(dtype=self.shape.dtype)
		assert self.shape.prod() == dims.prod()
		indices = self.indices
		indices = unfold(indices, self.shape)
		indices = fold(indices, dims)
		if inplace:
			self.indices = indices
			self.shape = dims
			self.ndim = len(dims)
			return self
		else:
			return Sparse(
				indices, self.values, dims, verbose=self.verbose,
				indptr=self.indptr if dims[0] == 0 else None,
			)

	def sort_indices(self, dim=0, force=False):
		assert 0 <= dim < self.ndim
		assert dim == 0
		if not force and self.indptr is not None: return
		d = np.diff(self.indices[dim])
		if (d >= 0).all():
			# print('indices are sorted')
			indptr = np.full(self.shape[dim]+1, len(self.values)+1, dtype=int)
			# indptr[0] = 0
			idx = np.concatenate([[0], np.nonzero(d != 0)[0]+1])
			indptr[self.indices[dim, idx]] = idx
			indptr[-1] = len(self.values)
			np.minimum.accumulate(indptr[::-1], out=indptr[::-1])
			# for i, (l, r) in enumerate(zip(indptr[:-1], indptr[1:])):
			# 	assert (self.indices[dim, l:r] == i).all()
			# assert (np.diff(indptr) >= 0).all()
			self.indptr = indptr
			return
		# print('sorting indices')
		_t = time.perf_counter()
		dim_other = np.arange(self.ndim) != dim
		row = self.indices[dim]
		col = self.indices[dim_other]
		# print(f'time elapsed = {time.perf_counter() - _t:.2e}')
		col = unfold(col, self.shape[dim_other])
		# print(f'time elapsed = {time.perf_counter() - _t:.2e}')
		M = self.shape[dim]
		N = col.max()+1
		indptr = np.empty(M + 1, dtype=col.dtype)
		indices = np.empty_like(col)
		values = np.empty_like(self.values)
		# print(f'time elapsed = {time.perf_counter() - _t:.2e}')
		coo_tocsr(
			M, N, len(self.values), row, col, self.values,
			indptr, indices, values,
		)
		# print(f'time elapsed = {time.perf_counter() - _t:.2e}')
		self.indices[dim] = np.repeat(np.arange(M), np.diff(indptr))
		# print(f'time elapsed = {time.perf_counter() - _t:.2e}')
		self.indices[dim_other] = fold(indices, self.shape[dim_other])
		# print(f'time elapsed = {time.perf_counter() - _t:.2e}')
		self.values[:] = values
		# self.values = values
		self.indptr = indptr
		# print(f'time elapsed = {time.perf_counter() - _t:.2e}')
		# for i, (l, r) in enumerate(zip(indptr[:-1], indptr[1:])):
		# 	assert (self.indices[dim, l:r] == i).all()
		if self.verbose:
			print(f'time used in sorting indices = {time.perf_counter() - _t:.2e}')

	def slicing(self, idx, dim=0):
		assert dim == 0
		assert isinstance(idx, slice)
		assert idx.step is None or idx.step == 1
		if self.indptr is None: self.sort_indices(dim)
		start = idx.start if idx.start is not None else 0
		stop = idx.stop if idx.stop is not None else self.shape[dim]
		stop = min(stop, self.shape[dim])
		indptr = self.indptr[start: stop+1]
		idx = slice(indptr[0], indptr[-1])
		indices = self.indices[:, idx].copy()
		# assert (indices[dim] >= start).all()
		# assert (indices[dim] < stop).all()
		indices[dim] -= start
		return Sparse(indices, self.values[idx], (stop-start,) + tuple(self.shape[1:]))
		# indptr = indptr - indptr[0]
		# return Sparse(indices, self.values[idx], (stop-start,) + tuple(self.shape[1:]), indptr=indptr)

	def indexing(self, idx, dim=0):
		assert dim == 0
		assert isinstance(idx, int)
		assert 0 <= idx < self.shape[dim], (self.shape, dim, idx)
		if self.indptr is None: self.sort_indices(dim)
		idx = slice(self.indptr[idx], self.indptr[idx+1])
		return Sparse(self.indices[1:, idx], self.values[idx], tuple(self.shape[1:]))

	def __getitem__(self, item):
		if isinstance(item, int): return self.indexing(item)
		elif isinstance(item, slice): return self.slicing(item)
		else: raise NotImplementedError
	
	def to_dense(self):
		v = np.zeros(tuple(self.shape))
		indices = tuple(self.indices)
		values = self.values
		v[indices] = values
		return v
	
	def to_scipy(self):
		return coo_matrix((self.values, tuple(self.indices)), tuple(self.shape))

	def to_csr(self):
		return csr_matrix((self.values, tuple(self.indices)), tuple(self.shape))

	def to_pytorch(self, **context):
		return torch.sparse_coo_tensor(
			# np.ascontiguousarray(self.indices),
			# np.ascontiguousarray(self.values),
			self.indices,
			self.values,
			self.shape.tolist(),
			**context,
		)

	def __len__(self):
		return self.shape[0]

	def split(self, bins, dim):
		bins = list(bins)
		chunk = np.digitize(self.indices[dim], bins)
		order = np.argsort(chunk, kind='stable')
		chunk = chunk[order]
		boundaries = [0] + (np.nonzero(chunk[:-1] != chunk[1:])[0]+1).tolist() + [len(order)]
		slices = [slice(*_) for _ in zip(boundaries[:-1], boundaries[1:])]
		# for i, slc in enumerate(slices):
		# 	assert i == 0 or (bins[i-1] <= self.indices[dim, order[slc]]).all()
		# 	assert i == len(slices)-1 or (self.indices[dim, order[slc]] < bins[i]).all()
		def f(indices, offset, dim):
			indices = indices.copy()
			indices[dim] -= offset
			return indices
		return (Sparse(
			f(self.indices[:, order[slc]], start, dim),
			self.values[order[slc]],
			tuple(self.shape[:dim]) + (stop-start,) + tuple(self.shape[dim+1:]),
		) for slc, start, stop in zip(slices, [0] + bins, bins + [self.shape[dim]]))

	def numel(self):
		return int(np.prod(self.shape))


def test():
	def new_obj():
		return Sparse([[0, 2, 1, 0], [0, 3, 2, 1]], [1, 2, 3, 4.], (3, 4))
	base = new_obj().to_scipy().todense()
	print(base)
	o = new_obj()
	o.sort_indices()
	assert (o.to_scipy().todense() == base).all()
	o = new_obj()
	o = o.permute(0, 1)
	assert (o.to_scipy().todense() == base).all()
	o = new_obj()
	o = o.permute(1, 0)
	assert (o.to_scipy().todense() == base.T).all()
	o = new_obj()
	o = o.reshape(1, 12)
	assert (o.to_scipy().todense() == base.reshape(1, 12)).all()
	o = new_obj()
	o = o.reshape(4, 3)
	assert (o.to_scipy().todense() == base.reshape(4, 3)).all()
	for s in [slice(2), slice(10), slice(0, None), slice(2, None), slice(1, 3)]:
		o = new_obj()
		o = o.slicing(s)
		assert (o.to_scipy().todense() == base[s]).all()

	o = new_obj()
	o = o.permute(1, 0).reshape(6, 2).slicing(slice(1, 3)).permute(1, 0)
	assert (o.to_scipy().todense() == base.T.reshape(6, 2)[1: 3].T).all()


if __name__ == '__main__':
	test()
