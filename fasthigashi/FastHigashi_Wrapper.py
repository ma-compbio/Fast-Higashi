import torch
import argparse, os, gc, pickle, sys
from pathlib import Path
from tqdm.auto import tqdm
import math, time, h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize

try:
	from .parafac2_intergrative import Fast_Higashi_core
	from .preprocessing import calc_bulk, filter_bin, normalize_per_cell, normalize_by_coverage, Clip, normalize_per_batch
	from .sparse_for_schic import Sparse, Chrom_Dataset
except:
	try:
		from parafac2_intergrative import Fast_Higashi_core
		from preprocessing import calc_bulk, filter_bin, normalize_per_cell, normalize_by_coverage, Clip, normalize_per_batch
		from sparse_for_schic import Sparse, Chrom_Dataset
	except:
		raise EOFError


def parse_args():
	parser = argparse.ArgumentParser(description="Higashi main program")
	parser.add_argument('-c', '--config', type=Path, default=Path("../config_dir/config_ramani.JSON"))
	parser.add_argument('--path2input_cache', type=Path, default=None)
	parser.add_argument('--path2result_dir', type=Path, default=None)
	parser.add_argument('--rank', type=int, default=256)
	parser.add_argument('--size', type=int, default=15)
	parser.add_argument('--size_func', type=str, default='scale')
	parser.add_argument('--off_diag', type=int, default=100)
	parser.add_argument('--fac_size', type=eval, default=1)
	parser.add_argument('--share_factors', type=eval, default=['shared', 'shared', 'shared'])
	parser.add_argument('--l2reg', type=float, default=10)
	parser.add_argument('--do_conv', action='store_true', default=False)
	parser.add_argument('--do_rwr', action='store_true', default=False)
	parser.add_argument('--do_col', action='store_true', default=False)
	parser.add_argument('--no_col', action='store_true', default=False)
	parser.add_argument('--extra', type=str, default="")
	parser.add_argument('--cache_extra', type=str, default="")
	parser.add_argument('--filter', action='store_true', default=False)
	parser.add_argument('--batch_norm', action='store_true', default=False)

	return parser.parse_args()


def get_config(config_path = "./config.jSON"):
	import json
	c = open(config_path,"r")
	return json.load(c)


def get_free_gpu(num=1):
	# os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Total > ./tmp1')
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used > ./tmp2')
	memory_all = [int(x.split()[2]) for x in open('tmp1', 'r').readlines()]
	memory_used = [int(x.split()[2]) for x in open('tmp2', 'r').readlines()]
	memory_available = [m1-m2 for m1,m2 in zip(memory_all, memory_used)]
	if len(memory_available) > 0:
		max_mem = np.max(memory_available)
		ids = np.where(memory_available >= max_mem-1000)[0]
		if num == 1:
			chosen_id = int(np.random.choice(ids, 1)[0])
			print("setting to gpu:%d" % chosen_id)
			sys.stdout.flush()
			torch.cuda.set_device(chosen_id)
			return torch.device("cuda:%d" % chosen_id), chosen_id
		else:
			return None, np.random.choice(ids, num, replace=False)
	else:
		return None, None


def get_memory_free(gpu_index):
	try:
		from py3nvml import py3nvml
		print ("gpu mem")
		py3nvml.nvmlInit()
		handle = py3nvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
		mem_info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
		print (mem_info)
		return mem_info.free
	except:
		print ("cpu device?")
		import psutil
		return psutil.virtual_memory().available


def parse_embedding(project_list, fac, dim=None):
	if dim is None:
		dim = fac.shape[1]
	else:
		dim = min(dim, fac.shape[1])
	embedding_list = []
	for p in project_list:
		if type(p).__name__ == 'Tensor': p = p.detach().cpu().numpy()
		p = p / np.linalg.norm(p, axis=0, keepdims=True)
		embed = fac @ p
		embedding_list.append(embed)

	embedding = np.concatenate(embedding_list, axis=1)
	from sklearn.decomposition import TruncatedSVD
	model = TruncatedSVD(n_components=dim)
	embedding = model.fit_transform(embedding)
	return embedding






class FastHigashi():
	def __init__(self, config_path,
				 path2input_cache,
				 path2result_dir,
				 off_diag,
				 filter,
				 do_conv,
				 do_rwr,
				 do_col,
				 no_col):
		super().__init__()
		self.off_diag = off_diag
		self.filter = filter
		self.do_conv = do_conv
		self.do_rwr = do_rwr
		self.do_col = do_col
		self.no_col = no_col
		
		self.config_path = config_path
		self.config = get_config(config_path)
		
		self.chrom_list = self.config['chrom_list']
		self.temp_dir = self.config['temp_dir']
		self.data_dir = self.config['data_dir']
		self.fh_resolutions = self.config['resolution_fh']
		
		
		if path2input_cache is None:
			path2input_cache = self.temp_dir
		
		if path2result_dir is None:
			path2result_dir = self.temp_dir
		
		self.path2input_cache = path2input_cache
		if not os.path.exists(path2input_cache):
			os.mkdir(path2input_cache)
		
		self.path2result_dir = path2result_dir
		if not os.path.exists(path2result_dir):
			os.mkdir(path2result_dir)
			
		_, self.gpu_id = get_free_gpu()
		if torch.cuda.is_available():
			self.device = 'cuda'
			# torch.set_num_threads(4)
		else:
			self.device = 'cpu'
	
	def eval_batch_mix(self, embed):
		from pynndescent import PyNNDescentTransformer
		import warnings
		from sklearn.decomposition import TruncatedSVD
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			if "batch_id" in self.config:
				batch_id = self.batch_id
				knn_graph = PyNNDescentTransformer(n_neighbors=50).fit_transform(embed)
				# print ("first knn")
				knn_graph.data = np.ones_like(knn_graph.data)
				knn_graph2nd = knn_graph @ knn_graph.T
				# print ("sparse dot")
				knn_graph2nd_embed = TruncatedSVD(n_components=embed.shape[-1]).fit_transform(knn_graph2nd)
				# print ("svd")
				knn_graph2nd = PyNNDescentTransformer(n_neighbors=10).fit_transform(knn_graph2nd_embed)
				# print ("second knn")
				knn_graph = PyNNDescentTransformer(n_neighbors=10).fit_transform(embed)
				# print ("third knn")
				knn_graph = knn_graph.tocoo()
				knn_graph2nd = knn_graph2nd.tocoo()
				mix_score = (np.sum(batch_id[knn_graph.row] == batch_id[knn_graph.col]) - len(embed)) / (
							len(knn_graph.row) - len(embed))
				mix_score2 = (np.sum(batch_id[knn_graph2nd.row] == batch_id[knn_graph2nd.col]) - len(embed)) / (
							len(knn_graph2nd.row) - len(embed))
			
			return np.array([mix_score, mix_score2])
	
	def preprocess_meta(self):
		if os.path.isfile(os.path.join(self.path2input_cache, "qc.npy")):
			qc = np.load(os.path.join(self.path2input_cache, "qc.npy"))
			readcount = np.load(os.path.join(self.path2input_cache, "read_count_all.npy"))
		else:
			qc, readcount = self.get_qc()
		good_qc = np.where(qc > 0)[0]
		bad_qc = np.where(qc <= 0)[0]
		# Always put good quality cells before bad quality cells
		reorder = np.concatenate([np.sort(good_qc),
								  np.sort(bad_qc)], axis=0)
		np.save(os.path.join(self.path2input_cache, "reorder.npy"), reorder)
		
		data_dir = self.config['data_dir']
		with open(os.path.join(data_dir, "label_info.pickle"), "rb") as f:
			label_info = pd.DataFrame(pickle.load(f))
		try:
			plot_label = self.config['plot_label']
		except:
			plot_label = []
		
		label_info = label_info.iloc[reorder].reset_index()
		sig_list = label_info[plot_label]
		
		gc.collect()
		
		if "batch_id" in self.config:
			self.batch_id = np.asarray(label_info[self.config['batch_id']])
		
		return label_info, reorder, sig_list, readcount, qc
	
	def pack_training_data_one_process(self,
			raw_dir, chrom, reorder,
			off_diag=None,
			fac_size=None,
			merge_fac_row=1, merge_fac_col=1,
			is_sym=True,
			filename_pattern='%s_sparse_adj.npy',
			force_shift=None,
	        batch_norm=True):
		
		filename = filename_pattern % chrom
		a = np.load(os.path.join(raw_dir, filename), allow_pickle=True)[reorder]
		
		# For data with blacklist, block those regions
		try:
			blacklist = np.load(os.path.join(self.temp_dir, "raw", "blacklist.npy"), allow_pickle=True).item()
		except:
			print("no black list")
			blacklist = None
		
		if blacklist is not None:
			bl = blacklist[chrom]
			bl = bl[bl < a[0].shape[0]]
			print("num of bl", bl.shape)
			new_sparse_list = []
			for m in a:
				m = m.astype('float32').toarray()
				m[bl, :] = 0.0
				m[:, bl] = 0.0
				new_sparse_list.append(csr_matrix(m))
			a = new_sparse_list
		
		if fac_size is None: fac_size = 1
		matrix_list = [m.tocoo() for m in a]
		
		if merge_fac_row > 1 or merge_fac_col > 1:
			for m in matrix_list:
				m.col //= merge_fac_col
				m.row //= merge_fac_row
				m.sum_duplicates()
				m.resize(list(np.ceil(np.array(m.shape) / np.array([merge_fac_col, merge_fac_row])).astype('int')))
		
		bulk = calc_bulk(matrix_list)
		bin_id_mapping_row, num_bins_row, bin_id_mapping_col, num_bins_col, v_row, v_col = filter_bin(bulk=bulk,
		                                                                                              is_sym=is_sym)
		
		if "batch_id" in self.config:
			if batch_norm:
				print ("per batch norm")
				matrix_list = normalize_per_batch(
					matrix_list=matrix_list,
					batch_id=self.batch_id
				)
			
		matrix_list = normalize_per_cell(
			matrix_list, matrix_list_intra=matrix_list, bulk=None,
			per_cell_normalize_func=[
				normalize_by_coverage,
			],
		)
		
		assert not any(np.isnan(m.data).any() for m in matrix_list)
		
		nnz = sum(m.nnz for m in matrix_list)
		indices = np.empty([3, nnz], dtype=np.int32)
		values = np.empty([nnz], dtype=np.float32)
		del nnz
		
		shape = (num_bins_row, num_bins_col)
		col_offset = None
		do_shift = False
		
		idx_nnz = 0
		for i, m in enumerate(matrix_list):
			if m.nnz == 0: continue
			row, col, data = bin_id_mapping_row[m.row], bin_id_mapping_col[m.col], m.data
			col_new = col - row
			idx = (row != -1) & (col != -1) & (col_new >= -off_diag) & (col_new <= off_diag)
			# assert idx.sum() > 0
			row, col, data = row[idx], col_new[idx], data[idx]
			if isinstance(fac_size, int) and fac_size == 1:
				pass
			elif isinstance(fac_size, int) and fac_size > 1:
				col += fac_size // 2
				if fac_size % 2 == 0: col[col <= 0] -= 1
				col //= fac_size
				offset = col.min()
				col -= offset
				tmp = coo_matrix((data, (row, col)), shape=(row.max() + 1, col.max() + 1))
				tmp.sum_duplicates()
				row, col, data = tmp.row, tmp.col + offset, tmp.data
				del tmp, offset
			else:
				col_sgn = np.sign(col)
				col = col_sgn * fac_size[np.abs(col)]
			
			if do_shift:
				col += col_offset
			else:
				col += row
			
			nnz = len(data)
			ii = slice(idx_nnz, idx_nnz + nnz)
			indices[0, ii] = row
			indices[1, ii] = col
			indices[2, ii] = i
			values[ii] = data
			idx_nnz += nnz
			del nnz, ii
		
		indices = np.ascontiguousarray(indices[:, :idx_nnz])
		values = np.ascontiguousarray(values[:idx_nnz])
		shape = shape + (len(matrix_list),)
		# print(shape, do_shift)
		assert indices.min() >= 0
		assert (indices.max(1) < shape).all()
		gc.collect()
		
		values = np.log1p(values)
		if is_sym:
			s = 15
		else:
			s = 15
		mean_, std_ = np.mean(values), np.std(values)
		values = np.clip(values, a_min=None, a_max=mean_ + s * std_)
		return indices, values, shape
	
	def preprocess_contact_map(self, config, reorder, path2input_cache, batch_norm, key_fn=lambda c: c, **kwargs):
		print(f'cache file = {path2input_cache}')
		do_cache = path2input_cache is not None
		
		if do_cache and os.path.exists(path2input_cache):
			print(f'loading cached input from {path2input_cache}')
			with open(path2input_cache, 'rb') as f: all_matrix = pickle.load(f)
			return all_matrix
		
		chrom_list = config['chrom_list']
		
		all_matrix = [self.pack_training_data_one_process(
			raw_dir=Path(config['temp_dir']) / 'raw', chrom=chrom, reorder=reorder,
			batch_norm=batch_norm,
			**kwargs,
		) for chrom in chrom_list]
		
		all_matrix = [Sparse(indices, values, shape, copy=False) for indices, values, shape in tqdm(all_matrix)]
		size_list = []
		for obj in tqdm(all_matrix):
			obj.sort_indices()
			size_list.append(obj.shape[0])
		
		if do_cache:
			print(f'saving cached input to {path2input_cache}')
			sys.stdout.flush()
			with open(path2input_cache, 'wb') as f: pickle.dump(all_matrix, f, protocol=4)
		
		sys.stdout.flush()
		return all_matrix
	
	@staticmethod
	def sum_sparse(m):
		x = np.zeros(m[0].shape)
		for a in tqdm(m):
			ri = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
			x[ri, a.indices] += a.data
		return x
	
	def get_qc(self):
		temp_dir = self.config['temp_dir']
		raw_dir = os.path.join(temp_dir, "raw")
		chrom_list = self.config['chrom_list']
		mask = []
		scale = int(1000000 / self.config['resolution'])
		read_count_all = 0
		for chrom in chrom_list:
			read_count = []
			a = np.load(os.path.join(raw_dir, "%s_sparse_adj.npy" % chrom), allow_pickle=True)
			bulk = self.sum_sparse(a)
			cov = np.sum(bulk, axis=-1)
			n_bin = np.sum(cov > 0.1 * cov.shape[0] * scale)
			
			mask_chrom = []
			for m in tqdm(a):
				off_diag = (np.sum(m > 0) + np.sum(m.diagonal() > 0)) / 2
				mask_chrom.append(off_diag)
				read_count.append(np.sum(m))
			# print(n_bin, m.shape[0])
			read_count = np.asarray(read_count)
			mask_chrom = np.array(mask_chrom).astype('float')
			mask.append(mask_chrom > n_bin)
			read_count_all += np.asarray(read_count)
			# print("pass", np.sum(mask[-1]))
		kept = (np.sum(np.array(mask).astype('float'), axis=0) >= (len(chrom_list))).astype('float32')
		print("total number of cells that pass qc check", np.sum(kept))
		read_count_all = np.log1p(read_count_all)
		np.save(os.path.join(self.path2input_cache, "qc.npy"), kept)
		np.save(os.path.join(self.path2input_cache, "read_count_all.npy"), read_count_all)
		return kept, read_count_all
		
	def prep_dataset(self, meta_only=False, batch_norm=True):
		self.label_info, reorder, self.sig_list, readcount, qc = self.preprocess_meta()
		self.reorder = reorder
		if meta_only:
			return
		good_qc_num = np.sum(qc > 0)
		# print("good", good_qc_num, "bad", len(qc) - good_qc_num)
		tensor_list = [None] * len(self.fh_resolutions) * len(self.chrom_list)
		recommend_bs_cell = []
		ct = 0
		for res in self.fh_resolutions:
			all_matrix = []
			path2input_cache_intra = os.path.join(self.path2input_cache, 'cache_intra_%d%s.pkl' % (res, args.cache_extra))
			# print("merge_fac_row", int(res / self.config['resolution']))
			all_matrix += self.preprocess_contact_map(
				self.config, reorder=reorder, path2input_cache=path2input_cache_intra,
				batch_norm=batch_norm,
				is_sym=True,
				off_diag=self.off_diag,
				fac_size=1,
				merge_fac_row=int(res / self.config['resolution']), merge_fac_col=int(res / self.config['resolution']),
				filename_pattern='%s_sparse_adj.npy',
				force_shift=False,
			)
			
			size_list = [m.shape[0] for m in all_matrix]
			num_cell = all_matrix[-1].shape[-1]
			avail_mem = get_memory_free(self.gpu_id)
			# 4 because of float32 -> bytes,
			# 10 because of overhead & cache
			max_tensor_size = avail_mem / (4 * 12)
			recommend_bs_bin = min(max(int(15000000 / res), 128), 256)
			total_cell_num = all_matrix[-1].shape[-1]
			print(total_cell_num)
			
			total_reads, total_possible = 0, 0
			
			for i, size in enumerate(size_list):
				n_batch = max(math.ceil(size / recommend_bs_bin), 1)
				if self.device == 'cpu':
					bs_bin_local = size
					bs_cell = num_cell
				else:
					bs_bin_local = math.ceil(size / n_batch)
					bs_cell = int(max_tensor_size / (bs_bin_local * (bs_bin_local + 2 * self.off_diag)))
					# print ("bs_cell", bs_cell)
					n_batch = int(math.ceil(num_cell / bs_cell))
					bs_cell = int(math.ceil(num_cell / n_batch))
					bs_cell = min(bs_cell, num_cell)
				# print ("bs_bin_local", bs_bin_local, size)
				recommend_bs_cell.append(bs_cell)
				try:
					total_reads += len(all_matrix[i].values)
				except:
					total_reads += torch.sum(all_matrix[i] > 0)
				total_possible += np.prod(all_matrix[i].shape)
				tensor_list[ct] = Chrom_Dataset(
					tensor=all_matrix[i],
					bs_bin=bs_bin_local,
					bs_cell=bs_cell,
					good_qc_num=good_qc_num if self.filter else -1,
					kind='hic',
					upper_sim=False,
					compact=True,
					flank=self.off_diag,
					chrom=self.chrom_list[i],
					resolution=res)
				ct += 1
				
			
			sparsity = total_reads / total_possible
			print("sparsity", sparsity)
			del all_matrix
			gc.collect()
			
			if sparsity * (500000 / res) ** 2 <= 0.03:
				print("sparsity below threshold, automatically col_normalize")
			do_col = sparsity * (500000 / res) ** 2 <= 0.03 or self.do_col
			if self.no_col:
				do_col = False
			print("do_conv", self.do_conv, "do_rwr", self.do_rwr, "do_col", do_col)
			self.final_do_col = do_col
			if self.no_col and self.do_col:
				print("choose one between do col or no col!")
				raise EOFError
			
		self.coverage_feats = readcount[reorder].reshape((-1, 1))
		print("recommend_bs_cell", recommend_bs_cell, "pinning memory")
		all_matrix = tensor_list
		for i in range(len(all_matrix)):
			all_matrix[i].pin_memory()
		
		shape_list = np.stack([mtx.shape[:-1] for mtx in all_matrix])
		print(shape_list, np.sum(shape_list[:, 0]))
		
		self.all_matrix = all_matrix
	
	def only_partial_rwr(self):
		try:
			from .partial_rwr import partial_rwr
		except:
			from partial_rwr import partial_rwr
		chrom_count = 0
		impute_result = h5py.File(os.path.join(self.path2result_dir, "impute_prwr.hdf5"), "w")
		for chrom_data in tqdm(self.all_matrix, desc="imputing"):
			group = impute_result.create_group(self.chrom_list[chrom_count])
			group.create_dataset("shape", data=np.asarray([chrom_data.num_bin, chrom_data.num_bin]))
			for cell_batch_id in range(0, chrom_data.num_cell_batch):
				slice_cell = chrom_data.cell_slice_list[cell_batch_id]
				
				imputed_map = None
				
				
				for bin_batch_id in range(0, chrom_data.num_bin_batch):
					slice_local = chrom_data.local_bin_slice_list[bin_batch_id]
					slice_col = chrom_data.col_bin_slice_list[bin_batch_id]
					slice_row = chrom_data.bin_slice_list[bin_batch_id]
					
					chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
															  save_context=dict(device=self.device),
															  transpose=True,
															  do_conv=False)
					
					chrom_batch_cell_batch, t = chrom_batch_cell_batch
					if imputed_map is None:
						imputed_map = np.zeros((int(chrom_batch_cell_batch.shape[0]), chrom_data.num_bin, chrom_data.num_bin))
					if kind == 'hic':
						chrom_batch_cell_batch, n_i = partial_rwr(chrom_batch_cell_batch,
															  slice_start=slice_local.start,
															  slice_end=slice_local.stop,
															  do_conv=self.do_conv,
															  do_rwr=self.do_rwr,
															  do_col=False,
															  bin_cov=torch.ones(1),
															  return_rwr_iter=True,
															  force_rwr_epochs=-1,
															  final_transpose=False)
						
						imputed_map[:, slice_row, slice_col] = chrom_batch_cell_batch.detach().cpu().numpy()
				imputed_map = imputed_map + imputed_map.transpose(0, 2, 1)
				for i in range(len(imputed_map)):
					m = imputed_map[i]
					m = m - np.diag(np.diag(m) / 2)
					group.create_dataset(str(self.reorder[slice_cell][i]), data=m.astype('float32'))
			i
			for cell_batch_id in range(0, chrom_data.num_cell_batch_bad):
				slice_cell = chrom_data.cell_slice_list[cell_batch_id + chrom_data.num_cell_batch]
				imputed_map = None
				
				for bin_batch_id in range(0, chrom_data.num_bin_batch):
					slice_local = chrom_data.local_bin_slice_list[bin_batch_id]
					slice_col = chrom_data.col_bin_slice_list[bin_batch_id]
					slice_row = chrom_data.bin_slice_list[bin_batch_id]
					
					chrom_batch_cell_batch, kind = chrom_data.fetch(bin_batch_id, cell_batch_id,
																	save_context=dict(device=self.device),
																	transpose=True,
																	do_conv=False,
																	good_qc=False)
					
					chrom_batch_cell_batch, t = chrom_batch_cell_batch
					if imputed_map is None:
						imputed_map = np.zeros((int(chrom_batch_cell_batch.shape[0]), chrom_data.num_bin, chrom_data.num_bin))
					if kind == 'hic':
						chrom_batch_cell_batch, _ = partial_rwr(chrom_batch_cell_batch,
																slice_start=slice_local.start,
																slice_end=slice_local.stop,
																do_conv=self.do_conv,
																do_rwr=self.do_rwr,
																do_col=False,
																bin_cov=torch.ones(1),
																return_rwr_iter=True,
																force_rwr_epochs=-1,
																final_transpose=False)
						imputed_map[:, slice_row, slice_col] = chrom_batch_cell_batch.detach().cpu().numpy()
				imputed_map = imputed_map + imputed_map.transpose(0, 2, 1)
				for i in range(len(imputed_map)):
					m = imputed_map[i]
					m = m - np.diag(np.diag(m) / 2)
					group.create_dataset(str(self.reorder[slice_cell][i]), data=m.astype('float32'))
					
						
			chrom_count += 1
		impute_result.close()
	
	def run_model(self, dim1=.6,
				  rank=256,
				  n_iter_parafac=1,
				  extra=""):
		self.rank = rank
		save_str = "dim1_%.1f_rank_%d_niterp_%d_%s" % (dim1, rank, n_iter_parafac, extra)
		self.save_str = save_str
		print(save_str)
		start = time.time()
		model = Fast_Higashi_core(rank=rank, off_diag=self.off_diag).to(self.device)
		result = model.fit_transform(
			self.all_matrix,
			size_ratio=dim1,
			n_iter_max=100,
			n_iter_parafac=1,
			do_conv=self.do_conv,
			do_rwr=self.do_rwr,
			do_col=self.final_do_col,
			tol=3.0e-5,
			gpu_id=self.gpu_id
		)
		print("takes: %.2f s" % (time.time() - start))
		weights_all, factors_all, p_list = result
		
		A_list, B_list, D_list, meta_embedding = factors_all
		
		self.meta_embedding = meta_embedding.detach().cpu().numpy()
		
		self.A_list = [A.detach().cpu().numpy() for A in A_list]
		self.B_list = [B.detach().cpu().numpy() for B in B_list]
		self.D_list = [D.detach().cpu().numpy() for D in D_list]
		self.p_list = [[p.detach().cpu().numpy() for p in temp] for temp in p_list]
		del factors_all, p_list, weights_all
		pickle.dump([self.A_list, self.B_list, self.D_list, self.meta_embedding, self.p_list],
					open(os.path.join(self.path2result_dir, "results_all%s.pkl" % save_str), "wb"), protocol=4)
		
		pickle.dump([self.meta_embedding, self.D_list], open(os.path.join(self.path2result_dir, "results%s.pkl" % save_str), "wb"), protocol=4)
		
	
	def load_model(self, dim1=.6,
				  rank=256,
				  n_iter_parafac=1,
				  extra=""):
		save_str = "dim1_%.1f_rank_%d_niterp_%d_%s" % (dim1, rank, n_iter_parafac, extra)
		data = pickle.load(open(os.path.join(self.path2result_dir, "results_all%s.pkl" % save_str), "rb"))
		self.A_list, self.B_list, self.D_list, self.meta_embedding, self.p_list = data
		
	
	def fetch_cell_embedding(self, final_dim=None, auto_correct=True, restore_order=False):
		embed = parse_embedding(self.D_list, self.meta_embedding, dim=final_dim if final_dim is not None else self.rank)
		if restore_order:
			reorder_embed = np.zeros_like(embed)
			reorder_embed[self.reorder] = embed
			embed = reorder_embed
			
		embed_l2 = normalize(embed, axis=1)
		embed_prefer = embed_l2
		if auto_correct:
			if "batch_id" in self.config:
				score_ori = self.eval_batch_mix(embed_l2)
				embed_correct = self.correct_batch_linear(embed, self.coverage_feats)
				embed_correct_l2 = normalize(embed_correct)
				score_correct = self.eval_batch_mix(embed_correct_l2)
				if np.max(score_ori - score_correct) > 0.01:
					print ("linear regression of coverage seems to improve the batch mixing")
					embed_prefer = embed_correct_l2
					
		
		return {'prefer': embed_prefer, 'raw':embed, 'l2_norm_raw': embed_l2}
			
		
	def correct_batch_linear(self, embedding, var_to_regress):
		if len(var_to_regress.shape) == 1:
			var_to_regress = var_to_regress.reshape((-1, 1))
		from sklearn.linear_model import LinearRegression
		embedding = embedding - LinearRegression().fit(var_to_regress, embedding).predict(var_to_regress)
		return embedding
	
if __name__ == '__main__':
	print(time.ctime())
	# parse all arguments
	args = parse_args()
	
	# initialize the model
	wrapper = FastHigashi(config_path=args.config,
				 path2input_cache=args.path2input_cache,
				 path2result_dir=args.path2result_dir,
				 off_diag=args.off_diag,
				 filter=args.filter,
				 do_conv=args.do_conv,
				 do_rwr=args.do_rwr,
				 do_col=args.do_col,
				 no_col=args.no_col)
	
	# packing data from sparse matrices to
	wrapper.prep_dataset(batch_norm=args.batch_norm)
	wrapper.run_model(extra=args.extra, rank=args.rank)
	
	
	# loading existing trained models
	# wrapper.load_model(extra=args.extra, rank=args.rank)
	
	# only do partial_rwr for analysis purpose
	# wrapper.only_partial_rwr()
	
	# getting embedding
	embed = wrapper.fetch_cell_embedding(final_dim=args.rank,
	                                     auto_correct=True,
	                                     restore_order=False)
	# prefer stands for the embeddings that the algorithm think might perform the best
	embed = embed['prefer']
	
	
	# internal uses... code not uploaded
	try:
		from .evaluation import evaluate_combine
	except:
		from evaluation import evaluate_combine
	evaluate_combine(wrapper.config, wrapper.sig_list, [slice(None)], embed, project=None, extra=args.cache_extra+"_"+args.extra+"", save_dir=wrapper.path2result_dir, with_CCA=False, label_info=wrapper.label_info,
	                 coverage_feats=None, log=None, number_only=False, save_fmt='png', linear_corr=False)
