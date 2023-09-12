import gc
import math
import time
from typing import List

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse._sparsetools import coo_tocsr

#TODO: use numpy.unravel_index

gpu_flag = torch.cuda.is_available()
# gpu_flag = False

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
	
	def scale(self, factor=1):
		self.shape = [int(math.ceil(self.shape[0] / factor)),
		             int(math.ceil(self.shape[1] / factor)),
		             self.shape[2]
		             ]
		self.sort_indices()
		self.indices[0] = np.floor(self.indices[0] / factor)
		self.indices[1] = np.floor(self.indices[1] / factor)
		self.indptr = None
		
		#sum_duplicates
		unique, inv, unique_counts = np.unique(self.indices.T, axis=0, return_inverse=True, return_counts=True)
		new_count = np.zeros_like(unique_counts, dtype='float32')
		for i, iv in enumerate(inv):
			new_count[iv] += self.values[i]
		self.indices = unique.T
		self.values = new_count
		return
	
	def filter_max_distance(self, max_distance=100):
		distance = np.abs(self.indices[1] - self.indices[0])
		mask = distance <= max_distance
		self.indices = np.asarray([a[mask] for a in self.indices])
		self.values = self.values[mask]
		self.indptr = None
	
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
	
	@torch.no_grad()
	def get_slice_idx_value(self, idx, dim=0, device='cpu'):
		assert dim == 0
		assert isinstance(idx, slice)
		assert idx.step is None or idx.step == 1
		if self.indptr is None: self.sort_indices(dim)
		start = idx.start if idx.start is not None else 0
		stop = idx.stop if idx.stop is not None else self.shape[dim]
		stop = min(stop, self.shape[dim])
		indptr = self.indptr[start: stop + 1]
		idx = slice(indptr[0], indptr[-1])
		indices = self.indices[:, idx]
		v = self.values[idx]
		return indices, v, (stop-start,) + tuple(self.shape[1:]), start
	
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
	
# import torch.jit as jit
# @jit.script
def densify_jit(shape:torch.Tensor,
                indices:List[torch.Tensor],
                values:torch.Tensor,
                device:torch.device,
                transpose:bool=False,
                do_conv:bool=False):
	
	if transpose:
		shape_local = [int(shape[2]), int(shape[0]), int(shape[1])]
	else:
		shape_local = [int(shape[0]), int(shape[1]), int(shape[2])]
	
	dense_tensor = torch.zeros(shape_local, device=device, dtype=values.dtype)
	values = values.to(device, non_blocking=True)

	cell_indices = indices[2].to(device, non_blocking=True).long()
	indices_0 = indices[0].to(device, non_blocking=True).long()
	indices_1 = indices[1].to(device, non_blocking=True).long()
	
	
	if do_conv:
		count = 0
		for id_0 in [-1, 0, 1]:
			for id_1 in [-1, 0, 1]:
				if transpose:
					dense_tensor[cell_indices, indices_0 + id_0, indices_1 + id_1] += values
				else:
					dense_tensor[indices_0 + id_0, indices_1 + id_1, cell_indices] += values
				count += 1
		dense_tensor /= count

	else:
		if transpose:
			dense_tensor[cell_indices, indices_0, indices_1] = values
		else:
			dense_tensor[indices_0, indices_1, cell_indices] = values
	
	if transpose:
		dense_tensor = dense_tensor[:, 1:-1, 1:-1]
	else:
		dense_tensor = dense_tensor[1:-1, 1:-1, :]
	return dense_tensor.clamp_(min=1e-8)

class Fake_Sparse:
	def __init__(self, slice_, indices, values, shape):
		self.slice_ = slice_
		
			
		self.shape = [shape[0]+2, shape[1]+2, shape[2]]
		
		if gpu_flag:
			self.indices = [torch.tensor(indices[0]+1, dtype=torch.short).pin_memory(),
			                torch.tensor(indices[1]+1, dtype=torch.short).pin_memory(),
			                torch.tensor(indices[2], dtype=torch.int).pin_memory()]
			self.values = torch.tensor(values, dtype=torch.float32).pin_memory()#[mask]

		else:
			self.indices = [torch.tensor(indices[0]+1, dtype=torch.int).contiguous(),
			                torch.tensor(indices[1]+1, dtype=torch.int).contiguous(),
			                torch.tensor(indices[2], dtype=torch.int).contiguous()]
			self.values = torch.tensor(values, dtype=torch.float32).contiguous()  # [mask]
		
		
	def compress(self, flank):
		print ("compressing")
		self.shape = [self.shape[0], self.shape[0] + 2 * flank, self.shape[2]]
		self.indices[1] = self.indices[1] - self.slice_.start + flank
		
	def pin_memory(self):
		self.values = self.values.pin_memory()
		self.indices = [_.pin_memory() for _ in self.indices]
		return self
	
	def densify(self, save_context, transpose=False, do_conv=False, out=None):
		return densify_jit(torch.as_tensor(self.shape), self.indices, self.values, save_context['device'], transpose, do_conv)


class Chrom_Dataset:
	def __init__(self, tensor, bs_bin, bs_cell, good_qc_num=-1, kind='hic',
	             upper_sim=False, compact=False, flank=0, chrom='chr1', resolution=10000):
		# tensor: big sparse or dense tensor
		# bs_bin: batch_size for bin
		# bs_cell: batch_size for cell
		# good_qc_num: the first good_qc_num cells are good cells
		# kind: is it hic or 1d signals
		# upper_sim: stores only the upper triangle signals or
		# compact: if compact if True: return matrix is size of (bs_bin, bs_bin + 2*flank, bs_cell) otherwise (bs_bin, all bins, bs_cell)
		# flank: flanking region size, also equivalent to the max distance.
		# chrom: the chromosome of this dataset (which can associate it with other dataset of same chrom but different resolution)
		# resolution: the resolution of this dataset
		
		self.resolution = resolution
		self.chrom = chrom
		self.length = tensor.shape[0]
		if good_qc_num == -1:
			good_qc_num = tensor.shape[-1]
		self.num_cell = good_qc_num
		self.total_cell_num = tensor.shape[-1]
		self.num_bin = tensor.shape[0]
		self.shape = [self.num_bin, tensor.shape[1], self.num_cell]
		self.bs_cell = bs_cell
		self.bs_bin = bs_bin
		
		self.tensor_list = []
		self.bad_tensor_list = []
		self.bin_slice_list = [] # this is the global one, indicating for a contact map of (n_bin, n_bin) where this small tensor correspond to
		self.local_bin_slice_list = [] # this is the local one, for compact map, it indicates which col slice in the small compact map correspond to
		self.cell_slice_list = []
		self.bad_cell_slice_list = []
		self.col_bin_slice_list = []
		
		self.kind_list = []
		self.bad_kind_list = []
		self.upper_sim = upper_sim
		self.compact = compact
		self.flank = flank
		
		# the tensor list is ordered by:
		# - n_batch_bin
		# - - n_batch_cell
		
		count = 0
		for i in range(0, self.num_bin, bs_bin):
			# Fetch and densify the X
			slice_ = slice(i, i + bs_bin)
			self.tensor_list.append([])
			self.bad_tensor_list.append([])
			self.kind_list.append([])
			self.bad_kind_list.append([])
			if not (type(tensor) is torch.Tensor):
				indices, values, shape, start = tensor.get_slice_idx_value(slice_, device='cpu')
				if self.upper_sim:
					# For a rectangular data slice: [x1:x2, :], the duplicated part is the lower triangular of [x1:x2, x1:x2]
					# Thus store upper triangle of [x1:x2, :] or , the left part of [x1:x2, :x1]
					mask = (indices[1, :] < start) | (indices[1, :] >= indices[0, :])
					indices = indices[:, mask]
					values = values[mask]
					# Because we'll do a = a + a.T, diag needs to be divided by 2
					mask2 = (indices[1, :] == indices[0, :])
					values[mask2] *=  0.5
				indices[0, :] -= start
			
			
			cell_start_point = list(np.arange(0, self.num_cell, self.bs_cell)) + \
			                   list(np.arange(self.num_cell, tensor.shape[-1], self.bs_cell))
			
			for cell_index in cell_start_point:
				if cell_index < self.num_cell:
					rhs = self.num_cell
					storage = self.tensor_list
					storage_kind = self.kind_list
				else:
					rhs = tensor.shape[-1]
					storage = self.bad_tensor_list
					storage_kind = self.bad_kind_list
				
				cell_start = cell_index
				cell_end = min(cell_index + self.bs_cell, rhs)
				
				if i == 0:
					self.cell_slice_list.append(slice(cell_start, cell_end))
				
				if (type(tensor) is torch.Tensor):
					t = tensor[slice_, :, slice(cell_index, cell_end)]
					storage[count].append(t)
					storage_kind[count].append(kind)
					if cell_index == 0:
						self.bin_slice_list.append(slice(i, i + t.shape[0]))
						self.local_bin_slice_list.append(slice(i, i + t.shape[0]))
						self.col_bin_slice_list.append(slice(None))
					continue
					
					
				mask = (indices[2] >= cell_index) & (indices[2] < cell_end)
				local_shape = (shape[0], shape[1], min(cell_end, shape[-1]) - cell_start)
				local_indices = [indices[0][mask], indices[1][mask], indices[2][mask] - cell_start]
				local_values = values[mask]
				
				if cell_index == 0:
					self.bin_slice_list.append(slice(i, i+shape[0]))
					if self.compact:
						if i > self.flank:
							self.local_bin_slice_list.append(slice(self.flank, self.flank + shape[0]))
							extend_shape_right = 0
							if self.num_bin - i - shape[0] - flank > 0:
								extend_shape_right += flank
							else:
								extend_shape_right += self.num_bin - i - shape[0]
							
							self.col_bin_slice_list.append(slice(i-self.flank, i+shape[0]+extend_shape_right))
						else:
							self.local_bin_slice_list.append(slice(i, i + shape[0]))
							extend_shape_right = 0
							if self.num_bin - i - shape[0] - flank > 0:
								extend_shape_right += flank
							else:
								extend_shape_right += self.num_bin - i - shape[0]
							self.col_bin_slice_list.append(slice(0, i + shape[0] + extend_shape_right))
					else:
						self.local_bin_slice_list.append(slice(i, i + shape[0]))
						self.col_bin_slice_list.append(slice(None))
				
				if compact:
					extend_shape = 0
					if i > self.flank:
						local_indices[1] = local_indices[1] - i + flank
						extend_shape += self.flank
					else:
						extend_shape += i
					if self.num_bin - i - shape[0] - flank > 0:
						extend_shape += flank
					else:
						extend_shape += self.num_bin - i - shape[0]
					local_shape = [local_shape[0], shape[0] + extend_shape, local_shape[2]]
					# print("esr", extend_shape_right, self.num_bin, i, shape[0], local_shape)
					# if np.sum(np.max(local_indices, axis=1) > np.asarray(local_shape)) > 0:
					# 	print (shape, i, extend_shape, self.num_bin)
					# 	print (chrom, local_shape, np.min(local_indices, axis=-1), np.max(local_indices, axis=1),
					# 	       np.min(local_values), np.max(local_values))
					# 	raise EOFError
				storage[count].append(Fake_Sparse(slice(i, i+shape[0]), local_indices, local_values, local_shape))
				storage_kind[count].append(kind)
				
				
			count += 1
		
		self.num_bin_batch = len(self.tensor_list)
		self.num_cell_batch = len(self.tensor_list[0])
		self.num_cell_batch_bad = len(self.bad_tensor_list[0])
		self.uniq_kind = np.unique(self.kind_list)
		if compact:
			self.shape = [self.num_bin,  bs_bin + 2 * flank, self.num_cell]
		
	def __len__(self):
		return self.length
	
	# hasn't adapted to multires
	def append_dim0(self, tensor, good_qc_num=-1, kind='hic'):
		if good_qc_num == -1:
			good_qc_num = tensor.shape[-1]
		bs_bin = self.bs_bin
		count = self.num_bin_batch
		
		for i in range(0, tensor.shape[0], bs_bin):
			# Fetch and densify the X
			slice_ = slice(i, i + bs_bin)
			self.tensor_list.append([])
			self.bad_tensor_list.append([])
			self.kind_list.append([])
			self.bad_kind_list.append([])
			if not (type(tensor) is torch.Tensor):
				indices, values, shape, start = tensor.get_slice_idx_value(slice_, device='cpu')
				indices[0, :] -= start
			for cell_index in range(0, self.num_cell, self.bs_cell):
				cell_start = cell_index
				cell_end = min(cell_index + self.bs_cell, self.num_cell)
				if (type(tensor) is torch.Tensor):
					t = tensor[slice_, :, slice(cell_index, cell_end)]
					self.tensor_list[count].append(t)
					self.kind_list[count].append(kind)
					if cell_index == 0:
						self.bin_slice_list.append(slice(i+self.num_bin, i+self.num_bin+t.shape[0]))
					continue
				mask = (indices[2] >= cell_index) & (indices[2] < cell_end)
				local_shape = (shape[0], shape[1], min(cell_end, shape[-1]) - cell_start)
				local_indices = [indices[0][mask], indices[1][mask], indices[2][mask] - cell_start]
				local_values = values[mask]
				if cell_index == 0:
					self.bin_slice_list.append(slice(i + self.num_bin, i + self.num_bin + shape[0]))
				self.tensor_list[count].append(Fake_Sparse(local_indices, local_values, local_shape))
				self.kind_list[count].append(kind)
			for cell_index in range(self.num_cell, tensor.shape[-1], self.bs_cell):
				cell_start = cell_index
				cell_end = min(cell_index + self.bs_cell, tensor.shape[-1])
				if (type(tensor) is torch.Tensor):
					self.bad_tensor_list[count].append(tensor[slice_, :, slice(cell_index, cell_end)])
					self.bad_kind_list[count].append(kind)
					continue
				mask = (indices[2] >= cell_index) & (indices[2] < cell_end)
				local_shape = (shape[0], shape[1], min(cell_end, shape[-1]) - cell_start)
				local_indices = [indices[0][mask], indices[1][mask], indices[2][mask] - cell_start]
				local_values = values[mask]
				self.bad_tensor_list[count].append(Fake_Sparse(local_indices, local_values, local_shape))
				self.bad_kind_list[count].append(kind)
			
			count += 1
		
		self.num_bin += tensor.shape[0]
		self.shape[0] += tensor.shape[0]
		
		
		
		self.num_bin_batch = len(self.tensor_list)
		self.num_cell_batch = len(self.tensor_list[0])
		self.num_cell_batch_bad = len(self.bad_tensor_list[0])
		self.uniq_kind = np.unique(self.kind_list)
		
	def pin_memory(self):
		for i in range(len(self.tensor_list)):
			for j in range(len(self.tensor_list[i])):
				self.tensor_list[i][j] = self.tensor_list[i][j].pin_memory()
		
		for i in range(len(self.bad_tensor_list)):
			for j in range(len(self.bad_tensor_list[i])):
				self.bad_tensor_list[i][j] = self.bad_tensor_list[i][j].pin_memory()
	
	def fetch_bad(self, bin_id, cell_id, **kwargs):
		return self.fetch(bin_id, cell_id, good_qc=False, **kwargs)
		
	def fetch(self, bin_id, cell_id, save_context, transpose=False, good_qc=True, **kwargs):
		if good_qc:
			temp = self.tensor_list[bin_id][cell_id]
			kind = self.kind_list[bin_id][cell_id]
		else:
			temp = self.bad_tensor_list[bin_id][cell_id]
			kind = self.bad_kind_list[bin_id][cell_id]
			
		transpose = False if kind != 'hic' else transpose
		if type(temp) is torch.Tensor:
			if transpose:
				temp = temp.permute(2, 0, 1).to(save_context['device'])
			return (temp.to(save_context['device']), [0]), kind
		else:
			_t = time.perf_counter()
			a = temp.densify(save_context, transpose, **kwargs)
			if self.upper_sim and kind == 'hic':
				# Why local? because local indicates the "diag" in the local compact map
				slice_ = self.local_bin_slice_list[bin_id]
				
				if transpose:
					a[:, :, slice_] = a[:, :, slice_] + a[:, :, slice_].permute(0, 2, 1)
				else:
					a[:, slice_, :] = a[:, slice_, :] + a[:, slice_, :].permute(1, 0, 2)
			b = [time.perf_counter() - _t]
			return (a, b), kind
		
	def norm(self):
		total_norm = 0
		for i in range(len(self.tensor_list)):
			for j in range(len(self.tensor_list[i])):
				total_norm += self.tensor_list[i][j].values.square().sum()
		return torch.sqrt(total_norm).item()
	
	def replace(self, bin_id, cell_id, dense_tensor, sparse_ratio):
		# self.tensor_list[bin_id][cell_id] = dense_tensor
		print (self.tensor_list[bin_id][cell_id].indices[0].shape)
		cutoff = torch.quantile(dense_tensor.permute(2, 0, 1).reshape(dense_tensor.shape[2], -1), 1 - sparse_ratio, dim=1)
		local_indices = (dense_tensor > cutoff[None, None, :]).nonzero().T
		local_values = dense_tensor[local_indices[0], local_indices[1], local_indices[2]]
		print (local_indices.shape, local_values.shape, torch.prod(torch.tensor(dense_tensor.shape)))
		# del self.tensor_list[bin_id][cell_id]
		self.tensor_list[bin_id][cell_id] = Fake_Sparse(local_indices,
		                                                local_values, dense_tensor.shape)
		gc.collect()

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
