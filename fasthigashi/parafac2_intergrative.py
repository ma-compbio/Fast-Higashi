import gc
import time
import torch.cuda
import torch.jit as jit
from opt_einsum import contract
from sklearn.decomposition import TruncatedSVD
try:
	from .parafac_integrative import parafac
	from .project2orthogonal import *
	from .partial_rwr import partial_rwr, tilte2rec
	from .util import *
except:
	from parafac_integrative import parafac
	from project2orthogonal import *
	from partial_rwr import partial_rwr, tilte2rec
	from util import *

import torch.nn.functional as F

## factors are named as : U (dim1, dim2, r), A(dim1, r), B(r, r), D(R, r), meta_embedding(dim3, R)
## matmul(meta_embedding, D) are also referred to as C
## size_list: list of r
## rank: R
def pass_(x, **kwargs):
	return x

gpu_flag = torch.cuda.is_available()
progressbar = pass_
def save_to_device(model_param, temp_var):
	if model_param is None or model_param is temp_var: return
	model_param[:] = temp_var.to(model_param.device)
	
def summarize_a_tensor(ts, name):
	a = torch.sum(torch.isnan(ts))
	if a > 0:
		print("name:", name)
		print ("shape:", ts.shape)
		print ("min/max:", torch.min(ts), torch.max(ts))
		print ("nan num:", torch.sum(torch.isnan(ts)))
		raise EOFError
	else:
		return
	
class Fast_Higashi_core():
	
	def __init__(self, rank, off_diag, res_list):
		self.rank = rank

		self.sparse_conv = False
		self.off_diag = off_diag
		self.res_list = res_list
		# print ("sparse conv", self.sparse_conv)
		self.device = torch.device("cpu")
		
	def to(self, device):
		self.device = device
		return self
	
	@torch.no_grad()
	def init_params(self, schic, do_conv, do_rwr, do_col):
		rank = self.rank
		uniq_size_list = list(self.chrom2size.values()) * len(self.res_list)
		_t = time.perf_counter()
		context = dict(device=self.device, dtype=torch.float32)
		context_cpu = dict(device='cpu', dtype=torch.float32)
		cum_size_list = np.concatenate([[0], np.cumsum(uniq_size_list)])
		# A_list, one for each matrix
		# A of size (#bin, r)
		# A could be large in cpu
		A_list = [
			torch.randn([chrom_data.shape[0], self.chrom2size[chrom_data.chrom]], **context_cpu) * 1e-2 + 1
			for chrom_data in schic]
		if gpu_flag: A_list = [a.pin_memory() for a in A_list]
		
		# B one for each chromosome (multi-resolution)
		# B is definitely small in gpu
		# B of size (size)
		B_dict = {chrom: torch.eye(size, **context).add_(torch.randn(size, **context), alpha=1e-2)
			for chrom, size in self.chrom2size.items()}
		
		C = None
		
		print(f'time elapsed: {time.perf_counter() - _t:.2f}')
		sys.stdout.flush()
		bin_cov_list = []
		bad_bin_cov_list = []
		n_i_all = []
		feats_for_SVD = {chrom: [] for chrom in self.chrom2size}
		
		for chrom_data in tqdm(schic, desc="initializing params"):
			if C is None:
				C = np.empty((chrom_data.num_cell, cum_size_list[-1]))
				C_start = 0
			n_i_list = []
			bin_cov = torch.ones([chrom_data.num_cell, chrom_data.num_bin],
			                     dtype=torch.float32, device='cpu') * 1e-4
			bad_bin_cov = torch.ones([chrom_data.total_cell_num - chrom_data.num_cell, chrom_data.num_bin],
			                         dtype=torch.float32, device='cpu') * 1e-4
			# for bin_index in range(0, chrom_data.shape[0], chrom_data.bs_bin):
			
			num_bin_1m = int(math.ceil(chrom_data.num_bin * chrom_data.resolution / 1000000))
			size1 = min(int(math.ceil(chrom_data.num_bin / chrom_data.num_bin_batch *
			                          chrom_data.resolution / 1000000)) + 2 * self.off_diag + 1,
			            num_bin_1m)
			feats_dim = int(math.ceil(num_bin_1m * size1))
			
			feats = np.empty((chrom_data.num_cell, feats_dim))
			
			ll = int(math.ceil(1000000 / chrom_data.resolution))
			if ll > 1:
				conv_filter = torch.ones(1, 1, ll, ll).to(self.device) / (ll * ll)
				conv_flag = True
			else:
				conv_flag = False
			feats_start = 0
			for bin_batch_id in range(0, chrom_data.num_bin_batch):
				slice_ = chrom_data.bin_slice_list[bin_batch_id]
				slice_local = chrom_data.local_bin_slice_list[bin_batch_id]
				slice_col = chrom_data.col_bin_slice_list[bin_batch_id]
				# feat = []
				for cell_batch_id in range(0, chrom_data.num_cell_batch):
					slice_cell = chrom_data.cell_slice_list[cell_batch_id]
					chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
					                                          save_context=dict(device=self.device),
					                                          transpose=True,
					                                          do_conv=do_conv if self.sparse_conv else False)
					chrom_batch_cell_batch, t = chrom_batch_cell_batch
					# here it's always, cell, row, col
					if kind == 'hic':
						chrom_batch_cell_batch, n_i = partial_rwr(chrom_batch_cell_batch,
						                                      slice_start=slice_local.start,
						                                      slice_end=slice_local.stop,
						                                      do_conv=False if self.sparse_conv else do_conv,
						                                      do_rwr=do_rwr,
						                                      do_col=False,
						                                      bin_cov=torch.ones(1),
						                                      return_rwr_iter=True,
						                                      force_rwr_epochs=-1,
						                                      final_transpose=False)
						b_c = chrom_batch_cell_batch.sum(1)
						bin_cov[slice_cell, slice_col] += b_c.detach().cpu()
						n_i_list.append(n_i)
						
						if not do_col:
						
							if conv_flag:
								B = F.conv2d(chrom_batch_cell_batch[:, None, :, :], conv_filter,
								             stride=ll)[:, 0, :, :]
							else:
								B = chrom_batch_cell_batch
							
							# feat.append(
							# 	B.cpu().numpy())
							B = B.cpu().numpy().reshape((len(B), -1))
							try:
								feats[slice_cell, feats_start:feats_start+B.shape[-1]] = B
							except Exception as e:
								print (B.shape, feats_start, feats_dim, num_bin_1m, chrom_data.resolution, chrom_data.num_bin, chrom_data.chrom)
								print (e)
								raise e
					# else:
					# 	feat.append(chrom_batch_cell_batch.permute(2, 0, 1).cpu().numpy().reshape(chrom_batch_cell_batch.shape[2], -1))
					del chrom_batch_cell_batch
				if not do_col:
					feats_start += B.shape[-1]
				gc.collect()
				try:
					torch.cuda.empty_cache()
				except:
					pass
				
				# if len(feat) > 0:
				# 	feat = np.concatenate(feat, axis=0)
				# 	feat = feat.reshape((len(feat), -1))
				# 	feats_for_SVD[chrom_data.chrom].append(feat)
				
				for cell_batch_id in range(0, chrom_data.num_cell_batch_bad):
					slice_cell = chrom_data.cell_slice_list[cell_batch_id + chrom_data.num_cell_batch]
					slice_cell = slice(slice_cell.start-chrom_data.num_cell, slice_cell.stop-chrom_data.num_cell)
					chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
					                                                    save_context=dict(device=self.device),
					                                                    transpose=True,
					                                                    do_conv=do_conv if self.sparse_conv else False,
					                                                    good_qc=False)
					chrom_batch_cell_batch, t = chrom_batch_cell_batch
					if kind == 'hic':
						chrom_batch_cell_batch, _ = partial_rwr(chrom_batch_cell_batch,
						                                        slice_start=slice_local.start,
						                                        slice_end=slice_local.stop,
						                                        do_conv=False if self.sparse_conv else do_conv,
						                                        do_rwr=do_rwr,
						                                        do_col=False,
						                                        bin_cov=torch.ones(1),
						                                        return_rwr_iter=True,
						                                        force_rwr_epochs=-1,
						                                        final_transpose=False)
						b_c = chrom_batch_cell_batch.sum(1)
						bad_bin_cov[slice_cell, slice_col] += b_c.detach().cpu()
			if do_col:
				for bin_batch_id in range(0, chrom_data.num_bin_batch):
					slice_ = chrom_data.bin_slice_list[bin_batch_id]
					slice_local = chrom_data.local_bin_slice_list[bin_batch_id]
					slice_col = chrom_data.col_bin_slice_list[bin_batch_id]
					# feat = []
					for cell_batch_id in range(0, chrom_data.num_cell_batch):
						slice_cell = chrom_data.cell_slice_list[cell_batch_id]
						chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
						                                                save_context=dict(device=self.device),
						                                                transpose=True,
						                                                do_conv=do_conv if self.sparse_conv else False)
						chrom_batch_cell_batch, t = chrom_batch_cell_batch
						# here it's always, cell, row, col
						if kind == 'hic':
							chrom_batch_cell_batch, n_i = partial_rwr(chrom_batch_cell_batch,
							                                          slice_start=slice_local.start,
							                                          slice_end=slice_local.stop,
							                                          do_conv=False if self.sparse_conv else do_conv,
							                                          do_rwr=do_rwr,
							                                          do_col=do_col,
							                                          bin_cov=bin_cov[slice_cell, slice_col],
							                                          return_rwr_iter=True,
							                                          force_rwr_epochs=-1,
							                                          final_transpose=False)
							
							
							if conv_flag:
								B = F.conv2d(chrom_batch_cell_batch[:, None, :, :], conv_filter,
								             stride=ll)[:, 0, :, :]
							else:
								B = chrom_batch_cell_batch
							
							# feat.append(
							# 	B.cpu().numpy())
							B = B.cpu().numpy().reshape((len(B), -1))
							feats[slice_cell, feats_start:feats_start + B.shape[-1]] = B
						
						# else:
						# 	feat.append(chrom_batch_cell_batch.permute(2, 0, 1).cpu().numpy().reshape(chrom_batch_cell_batch.shape[2], -1))
						del chrom_batch_cell_batch
					feats_start += B.shape[-1]
					gc.collect()
					try:
						torch.cuda.empty_cache()
					except:
						pass
					
					
			# feats_for_SVD[chrom_data.chrom] = np.concatenate(feats_for_SVD[chrom_data.chrom], axis=1)
			size = self.chrom2size[chrom_data.chrom]
			svd = TruncatedSVD(n_components=size, n_iter=2)
			temp = svd.fit_transform(feats[:, :feats_start])
			# del feats
			C[:, C_start:C_start+temp.shape[-1]] = temp
			C_start += temp.shape[-1]
			
			n_i_all.append(np.max(n_i_list) if len(n_i_list) > 0 else 0)
			if type(bin_cov) is not float:
				bin_cov[bin_cov <= 1e-4] = float('inf')
			if chrom_data.num_cell_batch_bad > 0:
				if type(bin_cov) is not float:
					bad_bin_cov[bad_bin_cov <= 1e-4] = float('inf')
					bad_bin_cov_list.append(bad_bin_cov)
				else:
					bad_bin_cov_list.append(0)
			else:
				bad_bin_cov_list.append(0)
			bin_cov_list.append(bin_cov)
		
		n_i_all = np.array(n_i_all)
		self.n_i = np.array(n_i_all)
		
		
		print("rwr iters:", self.n_i)
		C = torch.from_numpy(C).float()
		U, S, Vh = torch.linalg.svd(C.to(self.device), full_matrices=False)
		meta_embedding = U[:, :rank]
		SVh = Vh[:rank].mul_(S[:rank, None])
		D_dict = {
			chrom: SVh[:, start: stop].clone()
			for chrom, start, stop in zip(list(self.chrom2size.keys()), cum_size_list[:-1], cum_size_list[1:])
		}
		del C
		
		print(f'time elapsed: {time.perf_counter() - _t:.2f}')
		sys.stdout.flush()
		self.A_list = A_list
		self.B_dict = B_dict
		self.meta_embedding = meta_embedding
		self.D_dict = D_dict
		self.bin_cov_list = bin_cov_list
		self.bad_bin_cov_list = bad_bin_cov_list
		print ("finish init")
	
	@torch.no_grad()
	def update_meta_embedding_interactions(self,
			schic, projection_list,
			projected_tensor_list=None,
			do_conv=True, do_rwr=True, do_col=False,
			first_iter=False
	):
		if first_iter:
			rec_error_tensor_norm = np.zeros([len(schic), 1])
		device = self.device
		bin_cov_list = self.bin_cov_list
		rank = self.rank
		A_list, B_dict, D_dict, meta_embedding = self.A_list, self.B_dict, self.D_dict, self.meta_embedding
		
		SVD_term = torch.zeros(meta_embedding.shape[::-1], dtype=torch.float32, device=device)
		rec_error_x_U = np.zeros([len(schic), 1])
		rec_error_x_V = 0
		
		densify_time = 0
		partial_rwr_time = 0
		svd_time = 0
		contract_time = 0
		
						
		for chrom_index, (chrom_data, A, projection, bin_cov) in enumerate(zip(
				progressbar(schic), A_list, projection_list,
				bin_cov_list
		)):
			B = B_dict[chrom_data.chrom]
			D = D_dict[chrom_data.chrom]
			size = self.chrom2size[chrom_data.chrom]
			
			# chromosome specific embedding
			C = torch.matmul(meta_embedding, D)
			
			for bin_batch_id in range(0, chrom_data.num_bin_batch):
				slice_ = chrom_data.bin_slice_list[bin_batch_id]
				slice_local = chrom_data.local_bin_slice_list[bin_batch_id]
				slice_col = chrom_data.col_bin_slice_list[bin_batch_id]
				# Fetch and densify the X
				temp = None
				
				for cell_batch_id in range(0, chrom_data.num_cell_batch):
					slice_cell = slice(cell_batch_id * chrom_data.bs_cell,
					                   min((cell_batch_id + 1) * chrom_data.bs_cell, chrom_data.num_cell))
					_t1 = time.perf_counter()
					chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
					                                          save_context=dict(device=device),
					                                          transpose=(do_conv and not self.sparse_conv) or do_rwr,
					                                          do_conv=do_conv if self.sparse_conv else False)
					chrom_batch_cell_batch, t = chrom_batch_cell_batch
					densify_time += np.array(list(t) + [time.perf_counter() - _t1])
					_t = time.perf_counter()
					if kind == 'hic':
						chrom_batch_cell_batch, t1 = partial_rwr(chrom_batch_cell_batch,
						                                     slice_start=slice_local.start,
						                                     slice_end=slice_local.stop,
						                                     do_conv=False if self.sparse_conv else do_conv,
						                                     do_rwr=do_rwr,
						                                     do_col=do_col,
						                                     bin_cov=bin_cov[slice_cell, slice_col],
						                                     bin_cov_row=bin_cov[slice_cell, slice_],
						                                     force_rwr_epochs=self.n_i[chrom_index])
					
						
					if first_iter:
						rec_error_tensor_norm[chrom_index] += torch.linalg.norm(chrom_batch_cell_batch).square_().item()
					
					partial_rwr_time += time.perf_counter() - _t
					_t = time.perf_counter()
					# lhs: bs_bin, cell, size
					lhs = contract('ir,jr,kr->ikj', A[slice_].to(device), B, C[slice_cell].to(device))
					contract_time += time.perf_counter() - _t
					_t = time.perf_counter()
					# rhs: bs_bin, # bin2, cell
					rhs = chrom_batch_cell_batch.to(device)
					
					if temp is None:
						temp = torch.bmm(rhs, lhs)
					else:
						temp.baddbmm_(rhs, lhs)
					if cell_batch_id != chrom_data.num_cell_batch-1:
						del chrom_batch_cell_batch
					
				
				_t = time.perf_counter()
				# For smaller batch size or small matrix dimension, cpu is much faster
				# GPU has advantages when dealing with large batch size or super large matrix
				# Here, temp is shape of (bs_bin, total_bin, r)
				# bs_bin < 200, r ~ 100, total_bin can goes up to 2280, so...
				svd_device = 'cpu' if temp.shape[1] <= 700 else device
				try:
					U, S = project2orthogonal(temp.to(svd_device), temp.shape[-1], compute_device=device)
				except:
					U, S = project2orthogonal_ill(temp.to(svd_device), temp.shape[-1], compute_device=device)
					
				svd_time += time.perf_counter() - _t
				_t = time.perf_counter()
			
				# store projections
				projection[bin_batch_id] = U.to(projection[bin_batch_id].device)
				U = U.to(device)
				
				
				# calc error
				if S is None:
					rec_error_x_U[chrom_index] += temp.view(-1).inner(U.view(-1)).item()
				else:
					# assert (S.sum().item() - temp.view(-1).inner(U.view(-1)).item()) / S.sum().item() < 1e-5, (
					# 	S.sum().item(), temp.view(-1).inner(U.view(-1)).item(),
					# 	S.sum().item() - temp.view(-1).inner(U.view(-1)).item(),
					# 	(S.sum().item() - temp.view(-1).inner(U.view(-1)).item()) / S.sum().item()
					# )
					rec_error_x_U[chrom_index] += S.sum().item()
				
				
				
				# lhs: rank, # bin1, size
				lhs = contract('ir,jr,kr->kij', A[slice_].to(device), B, D)
				lhs = lhs.reshape(lhs.shape[0], -1)
				_t = time.perf_counter()
				
				# First use the last densified one
				# bin1, size, bin2 * bin1, bin2, bs_cell -> bin1, size, bs_cell
				_t = time.perf_counter()
				projected = torch.bmm(U.transpose(-1, -2), chrom_batch_cell_batch)
				SVD_term[:, slice_cell] += torch.matmul(lhs, projected.reshape(-1, projected.shape[-1]))
				contract_time += time.perf_counter() - _t
				_t = time.perf_counter()
				
				# All but the last one (which has been reused )
				for cell_batch_id in range(0, chrom_data.num_cell_batch - 1):
					slice_cell = slice(cell_batch_id * chrom_data.bs_cell,
					                   min((cell_batch_id + 1) * chrom_data.bs_cell, chrom_data.num_cell))
					
					if chrom_data.bs_cell < chrom_data.num_cell:
						_t = time.perf_counter()
						# torch.cuda.synchronize()
						chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
						                                          save_context=dict(device=device),
						                                          transpose=(do_conv and not self.sparse_conv) or do_rwr,
						                                          do_conv=do_conv if self.sparse_conv else False)
						chrom_batch_cell_batch, t = chrom_batch_cell_batch
						# torch.cuda.synchronize()
						densify_time += np.array(list(t)+[time.perf_counter() - _t])
						_t = time.perf_counter()
						if kind == 'hic':
							chrom_batch_cell_batch, t1 = partial_rwr(chrom_batch_cell_batch.clamp_(1e-8),
							                                         slice_start=slice_local.start,
							                                         slice_end=slice_local.stop,
							                                         do_conv=False if self.sparse_conv else do_conv,
							                                         do_rwr=do_rwr,
							                                         do_col=do_col,
							                                         bin_cov=bin_cov[slice_cell, slice_col],
							                                         bin_cov_row=bin_cov[slice_cell, slice_],
							                                         force_rwr_epochs=self.n_i[chrom_index]
							                                         )
						else:
							t1 = 0
						
						partial_rwr_time += t1
						partial_rwr_time += time.perf_counter() - _t
						_t = time.perf_counter()
						
					# bin1, size, bin2 * bin1, bin2, bs_cell -> bin1, size, bs_cell
					_t = time.perf_counter()
					projected = torch.bmm(U.transpose(-1, -2), chrom_batch_cell_batch)
					SVD_term[:, slice_cell] += torch.matmul(lhs, projected.reshape(-1, projected.shape[-1]))
					# if kind == 'hic':
					# 	SVD_term_1[:, slice_cell] += torch.matmul(lhs, projected.reshape(-1, projected.shape[-1]))
					# else:
					# 	SVD_term_2[:, slice_cell] += torch.matmul(lhs, projected.reshape(-1, projected.shape[-1]))
					contract_time += time.perf_counter() - _t
					del chrom_batch_cell_batch
					_t = time.perf_counter()
			
			del C
		# SVD_term: dim3 * R
		_t = time.perf_counter()
		meta_embedding, S = project2orthogonal(SVD_term.T, rank=rank, compute_device=device)
		svd_time += time.perf_counter() - _t
		_t = time.perf_counter()
		rec_error_x_V += meta_embedding.mul(SVD_term.T).sum().item()
		
		for chrom_index, (chrom_data, projection, bin_cov) in enumerate(
				zip(progressbar(schic), projection_list, bin_cov_list)):
			
			for bin_batch_id in range(0, chrom_data.num_bin_batch):
				gather_project = 0
				slice_ = chrom_data.bin_slice_list[bin_batch_id]
				slice_local = chrom_data.local_bin_slice_list[bin_batch_id]
				slice_col = chrom_data.col_bin_slice_list[bin_batch_id]
				for cell_batch_id in range(0, chrom_data.num_cell_batch):
					slice_cell = slice(cell_batch_id * chrom_data.bs_cell,
					                   min((cell_batch_id + 1) * chrom_data.bs_cell, chrom_data.num_cell))
					_t = time.perf_counter()
					chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
					                                          save_context=dict(device=device),
					                                          transpose=(do_conv and not self.sparse_conv) or do_rwr,
					                                          do_conv=do_conv if self.sparse_conv else False)
					chrom_batch_cell_batch, t = chrom_batch_cell_batch
					densify_time += np.array(list(t) + [time.perf_counter() - _t])
					_t = time.perf_counter()
					if kind == 'hic':
						chrom_batch_cell_batch, t1 = partial_rwr(chrom_batch_cell_batch.clamp_(1e-8),
						                                      slice_start=slice_local.start,
						                                      slice_end=slice_local.stop,
						                                      do_conv=False if self.sparse_conv else do_conv,
						                                      do_rwr=do_rwr,
						                                      do_col=do_col,
						                                      bin_cov=bin_cov[slice_cell, slice_col],
						                                      bin_cov_row=bin_cov[slice_cell, slice_],
						                                      force_rwr_epochs=self.n_i[chrom_index])
						partial_rwr_time += t1
					
					
					_t = time.perf_counter()
					
					projected = contract(
						"ijk,km, ijl -> ilm", chrom_batch_cell_batch, meta_embedding[slice_cell],
						projection[bin_batch_id].to(meta_embedding.device))
					del chrom_batch_cell_batch
					gather_project += projected
					contract_time += time.perf_counter() - _t
					_t = time.perf_counter()
				projected_tensor_list[chrom_data.chrom][chrom_data.global_slice_bin][slice_] = gather_project.to(projected_tensor_list[chrom_data.chrom].device)
				
		self.meta_embedding = meta_embedding.to(self.meta_embedding.device)
		
		gc.collect()
		try:
			torch.cuda.empty_cache()
		except:
			pass
		if first_iter:
			return projection_list, projected_tensor_list, rec_error_x_U, rec_error_x_V, rec_error_tensor_norm
		return projection_list, projected_tensor_list, rec_error_x_U, rec_error_x_V
	
	
	
	def fit(self, schic, size_ratio=0.3,
	                  n_iter_max=2000, n_iter_parafac=5,
	                  do_conv=True, do_rwr=False, do_col=False, tol=1e-8,
	                  size_list = None, gpu_id=None,
	                  verbose=True):
		
		self.gpu_id = gpu_id
		self.all_in_gpu = False
		self.benchmark_speed = False
		rank = self.rank
		device = self.device
		# Calculating sizes, the size would be forced to be the same for matrix from the same chromosomes
		
		if size_list is None:
			size_list = [min(int(chrom_data.shape[0] * size_ratio * chrom_data.resolution / 1000000), rank)
			             for chrom_data in schic]
			chrom2size = {}
			for chrom_data, size in zip(schic, size_list):
				chrom = chrom_data.chrom
				if chrom in chrom2size:
					chrom2size[chrom] = min(chrom2size[chrom], size)
				else:
					chrom2size[chrom] = size
		else:
			chrom2size = {}
			for chrom_data, size in zip(schic, size_list):
				chrom = chrom_data.chrom
				if chrom in chrom2size:
					if size != chrom2size[chrom]:
						print ("size of the same chromosome must be same!", size, chrom2size[chrom], chrom)
						raise EOFError
				else:
					chrom2size[chrom] = size
		
		self.chrom2size = chrom2size
		self.chrom2num_bin = {}
		self.chrom2id = {chrom:[] for chrom in self.chrom2size}
		for chrom_index, chrom_data in enumerate(schic):
			self.chrom2id[chrom_data.chrom].append(chrom_index)
			if chrom_data.chrom not in self.chrom2num_bin:
				self.chrom2num_bin[chrom_data.chrom] = chrom_data.num_bin
				chrom_data.global_slice_bin = slice(0, chrom_data.num_bin)
			else:
				chrom_data.global_slice_bin = slice(self.chrom2num_bin[chrom_data.chrom],
				                                    self.chrom2num_bin[chrom_data.chrom]+chrom_data.num_bin)
				self.chrom2num_bin[chrom_data.chrom] += chrom_data.num_bin
				
		
		del size_list
		
		self.init_params(schic, do_conv, do_rwr, do_col)
		rec_errors = []
		rec_errors_total = []
		
		
		# create_projection_list:
		projection_list = []
		for chrom_data in schic:
			temp1 = []
			for bin_batch in chrom_data.tensor_list:
				for cell_batch_bin_batch in bin_batch:
					a = torch.empty((cell_batch_bin_batch.shape[0] - 2,
					                          cell_batch_bin_batch.shape[1] - 2,
					                          self.chrom2size[chrom_data.chrom]
					), dtype=torch.float32)
					if gpu_flag: a = a.pin_memory()
					temp1.append(a)
					break
			projection_list.append(temp1)
		# Note the batch_id dim is at dim 1
		projected_tensor_list = {chrom:
			torch.empty([self.chrom2num_bin[chrom], self.chrom2size[chrom], rank], dtype=torch.float32)
			for chrom in self.chrom2size
		                         }
		if gpu_flag:
			for a in projected_tensor_list: projected_tensor_list[a] = projected_tensor_list[a].pin_memory()
		rec_error_core_norm = np.zeros([len(schic), 1])
		
		for chrom_index, (chrom_data, A, re_c, bin_cov) in enumerate(zip(
				progressbar(schic), self.A_list, rec_error_core_norm, self.bin_cov_list)):
			B = self.B_dict[chrom_data.chrom]
			D = self.D_dict[chrom_data.chrom]
			for i in range(0, chrom_data.num_bin, chrom_data.bs_bin):
				slice_ = slice(i, i + chrom_data.bs_bin)
				c = contract('ir,jr,kr->kij', A[slice_].to(device), B, D)
				
				re_c[:] += torch.linalg.norm(c).square_().item()
				del c
		
		rec_error_tensor_norm = None
		for iteration in range(n_iter_max):
			if (iteration % 10) == 0 and iteration > 0 and n_iter_parafac < 5:
				n_iter_parafac += 1
				# print ("n_iter_para", n_iter_parafac)
			print("Starting iteration", iteration)
			sys.stdout.flush()
			
			start_time = time.time()

			if rec_error_tensor_norm is None:
				projection_list, projected_tensor_list, rec_error_x_U, rec_error_x_V, rec_error_tensor_norm = \
					self.update_meta_embedding_interactions(
						schic, projection_list,
						projected_tensor_list,
						do_conv=do_conv,
						do_rwr=do_rwr,
						do_col=do_col,
						first_iter=True
					)
				norm_tensor = np.sqrt(rec_error_tensor_norm).reshape((-1))
				norm_tensor_all = float(np.linalg.norm(norm_tensor))
			else:
				projection_list, projected_tensor_list, rec_error_x_U, rec_error_x_V = \
					self.update_meta_embedding_interactions(
						schic, projection_list,
						projected_tensor_list,
						do_conv=do_conv,
						do_rwr=do_rwr,
						do_col=do_col)
				

			
			rec_error_by_block_U = rec_error_tensor_norm + rec_error_core_norm - 2 * rec_error_x_U
			rec_error_V = rec_error_tensor_norm.sum() + rec_error_core_norm.sum() - 2 * rec_error_x_V
			del rec_error_x_U, rec_error_x_V
			
			rec_error_x_core = np.zeros([len(schic), 1])
			
			# Run parafac on projected tensors (size of (dim1, size, rank))
			for chrom in self.chrom2id:
				ids = self.chrom2id[chrom]
				temp_A = torch.cat([self.A_list[i] for i in ids], dim=0)
				temp_B = self.B_dict[chrom]
				temp_D = self.D_dict[chrom]
				temp_factors = [temp_A, temp_B, temp_D]
				factors, core_norm, loss_x = parafac(
					projected_tensor_list[chrom],
					rank=self.chrom2size[chrom],
					init=temp_factors,
					n_iter_max=n_iter_parafac,
					verbose=False,
				)
				
				# rec_error_core_norm[chrom_index] = core_norm
				# rec_error_x_core[chrom_index] = loss_x
				
				for i in ids:
					self.A_list[i][:] = factors[0][schic[i].global_slice_bin].to(self.A_list[i].device)
				
				self.B_dict[chrom][:] = factors[1].to(self.B_dict[chrom].device)
				self.D_dict[chrom][:] = factors[2].to(self.D_dict[chrom].device)
			
			rec_error_core_norm = np.zeros([len(schic), 1])
			for chrom_index, (chrom_data, A, re_c, bin_cov) in enumerate(zip(
					progressbar(schic), self.A_list, rec_error_core_norm, self.bin_cov_list)):
				B = self.B_dict[chrom_data.chrom]
				D = self.D_dict[chrom_data.chrom]
				for i in range(0, chrom_data.num_bin, chrom_data.bs_bin):
					slice_ = slice(i, i + chrom_data.bs_bin)
					c = contract('ir,jr,kr->kij', A[slice_].to(device), B, D)
					re_c[:] += torch.linalg.norm(c).square_().item()
					del c
					
			print()
			
			rec_error = np.sqrt(rec_error_V.sum()) / norm_tensor_all
			rec_errors_total.append(rec_error)
			rec_error_by_block = np.sqrt(rec_error_by_block_U.ravel()) / norm_tensor
			rec_errors.append(rec_error_by_block)
			
			if iteration >= 1:
				differences = (rec_errors[-2] ** 2 - rec_errors[-1] ** 2) / (rec_errors[-2] ** 2)
				total_differences = (
							(rec_errors_total[-2] ** 2 - rec_errors_total[-1] ** 2) / rec_errors_total[-2] ** 2)
				
				print(
					f"PARAFAC2 re={rec_error:.3f} "
					f"{total_differences:.2e} "
					f"variation min{differences.min().item():.1e} at chrom {differences.argmin().item():d}, "
					f"max{differences.max().item():.1e} at chrom {differences.argmax().item():d}",
					f"takes {time.time() - start_time:.1f}s"
				)
				# if iteration >= 3 and tol > 0 and (total_differences < tol or differences.max() < tol * 2):
				if iteration >= 3 and tol > 0 and (total_differences < tol or differences.max() < tol * 2):
					print('converged in {} iterations.'.format(iteration))
					break
			else:
				
				print(
					f"PARAFAC2 re={rec_error:.3f} "
					f"takes {time.time() - start_time:.1f}s"
				)
			sys.stdout.flush()
		self.projection_list = projection_list
		self.projected_tensor_list = projected_tensor_list
		return self
		
	def transform(self, schic, do_conv, do_rwr, do_col):
		if self.device == 'cpu':
			do_conv = False
			do_rwr = False
			do_col = False
		# final update of meta-embeddings:
		device = self.device
		projection_list = self.projection_list
		projected_tensor_list = self.projected_tensor_list
		bin_cov_list = self.bin_cov_list
		bad_bin_cov_list = self.bad_bin_cov_list
		rank = self.rank
		print ("start transform")
		SVD_term = torch.zeros([self.meta_embedding.shape[-1], schic[0].total_cell_num], dtype=torch.float32, device=device)
		
		lhs_all = 0
		
		
		for chrom_index, (chrom_data, A, projection, bin_cov, bad_bin_cov) in enumerate(zip(
				progressbar(schic), self.A_list, projection_list,
				bin_cov_list, bad_bin_cov_list
		)):
			B = self.B_dict[chrom_data.chrom]
			D = self.D_dict[chrom_data.chrom]
			for bin_batch_id in range(0, chrom_data.num_bin_batch):
				# slice_ = slice(bin_batch_id * chrom_data.bs_bin, bin_batch_id * chrom_data.bs_bin + chrom_data.bs_bin)
				slice_ = chrom_data.bin_slice_list[bin_batch_id]
				slice_local = chrom_data.local_bin_slice_list[bin_batch_id]
				slice_col = chrom_data.col_bin_slice_list[bin_batch_id]
				# lhs: rank, # bin1, size
				lhs = contract('ir,jr,kr->kij', A[slice_].to(device), B, D)
				lhs = lhs.reshape(lhs.shape[0], -1)
				lhs_all += lhs @ lhs.T
				U = projection[bin_batch_id].to(device)
				
				# Fetch and densify the X
				
				for cell_batch_id in range(0, chrom_data.num_cell_batch):
					slice_cell = slice(cell_batch_id * chrom_data.bs_cell,
					                   min((cell_batch_id + 1) * chrom_data.bs_cell, chrom_data.num_cell))
					chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
					                                          save_context=dict(device=device),
					                                          transpose=(do_conv and not self.sparse_conv) or do_rwr,
					                                          do_conv=do_conv if self.sparse_conv else False)
					chrom_batch_cell_batch, t = chrom_batch_cell_batch
					
					if kind == 'hic':
						chrom_batch_cell_batch, t1 = partial_rwr(chrom_batch_cell_batch,
						                                         slice_start=slice_local.start,
						                                         slice_end=slice_local.stop,
						                                         do_conv=False if self.sparse_conv else do_conv,
						                                         do_rwr=do_rwr,
						                                         do_col=do_col,
						                                         bin_cov=bin_cov[slice_cell, slice_col],
						                                         bin_cov_row=bin_cov[slice_cell, slice_],
						                                         force_rwr_epochs=self.n_i[chrom_index])
						
					projected = torch.bmm(U.transpose(-1, -2), chrom_batch_cell_batch)
	
					SVD_term[:, slice_cell] += torch.matmul(lhs, projected.reshape(-1, projected.shape[-1]))
				
				for cell_batch_id in range(0, chrom_data.num_cell_batch_bad):
					slice_cell = slice(cell_batch_id * chrom_data.bs_cell,
					                   (cell_batch_id + 1) * chrom_data.bs_cell)
					chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
					                                          save_context=dict(device=device),
					                                          transpose=(do_conv and not self.sparse_conv) or do_rwr,
					                                          do_conv=do_conv if self.sparse_conv else False,
					                                          good_qc=False)
					chrom_batch_cell_batch, t = chrom_batch_cell_batch
					if kind == 'hic':
						chrom_batch_cell_batch, t1 = partial_rwr(chrom_batch_cell_batch,
						                                         slice_start=slice_local.start,
						                                         slice_end=slice_local.stop,
						                                         do_conv=False if self.sparse_conv else do_conv,
						                                         do_rwr=do_rwr,
						                                         do_col=do_col,
						                                         bin_cov=bad_bin_cov[slice_cell, slice_col],
						                                         bin_cov_row=bad_bin_cov[slice_cell, slice_],
						                                         force_rwr_epochs=self.n_i[chrom_index])
					
					projected = torch.bmm(U.transpose(-1, -2), chrom_batch_cell_batch)
					slice_cell2 = slice(chrom_data.num_cell + cell_batch_id * chrom_data.bs_cell,
					                    chrom_data.num_cell + (cell_batch_id + 1) * chrom_data.bs_cell)
					SVD_term[:, slice_cell2] += torch.matmul(lhs, projected.reshape(-1, projected.shape[-1]))
				
	
	
		# SVD_term: dim3 * R
		meta_embedding, S = project2orthogonal(SVD_term.T, rank=rank, compute_device=device)
		parafac2_tensor = (None, (self.A_list, self.B_dict.values(), self.D_dict.values(), meta_embedding), self.projection_list)
		
		return parafac2_tensor
	
	def fit_transform(self, schic, size_ratio=0.3,
	                  n_iter_max=2000, n_iter_parafac=5,
	                  do_conv=True, do_rwr=False, do_col=False, tol=1e-8,
	                  size_list = None, gpu_id=None,
	                  verbose=True):
		print ("n_iter_parafac", n_iter_parafac)
		self.fit(schic, size_ratio,
	                  n_iter_max, n_iter_parafac,
	                  do_conv, do_rwr, do_col, tol,
	                  size_list, gpu_id,
	                  verbose)
			
			
		
		return self.transform(schic, do_conv, do_rwr, do_col)