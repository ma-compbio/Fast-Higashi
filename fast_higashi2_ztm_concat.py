import torch.cuda
import argparse, os, pickle, sparse, gc, pickle, sys, itertools
from pathlib import Path
from tqdm.auto import tqdm, trange

import numpy as np
import pandas as pd

from torch_parafac2_integrative import fast_higashi_integrative
from util import get_config
from preprocessing import calc_bulk, filter_bin, normalize_per_cell, normalize_by_coverage, Clip
from sparse import Sparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.sparse import coo_matrix, csr_matrix


def parse_args():
	parser = argparse.ArgumentParser(description="Fast-Higashi main program")
	parser.add_argument('-c', '--config', type=Path, default=Path("../config_dir/config_ramani.JSON"))
	parser.add_argument('--path2input_cache', type=Path, default=None)
	parser.add_argument('--path2result_dir', type=Path, default=None)
	parser.add_argument('--use_intra', action='store_true', default=False)
	parser.add_argument('--use_inter', action='store_true', default=False)
	parser.add_argument('--rank', type=int, default=200)
	parser.add_argument('--size', type=int, default=15)
	parser.add_argument('--size_func', type=str, default='scale')
	parser.add_argument('--off_diag', type=int, default=100)
	parser.add_argument('--fac_size', type=eval, default=1)
	parser.add_argument('--share_factors', type=eval, default=['shared', 'shared', 'shared'])
	parser.add_argument('--l2reg', type=float, default=10)
	parser.add_argument('--do_conv', action='store_true', default=False)
	parser.add_argument('--do_rwr', action='store_true', default=False)
	parser.add_argument('--eval', action='store_true', default=False)
	parser.add_argument('--label_list', type=eval, default=None)
	parser.add_argument('--extra', type=str, default="")
	parser.add_argument('--method', type=str, default="HOOI")

	return parser.parse_args()


def get_free_gpu(num=1):
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > ./tmp')
	memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
	if len(memory_available) > 0:
		max_mem = np.max(memory_available)
		ids = np.where(memory_available >= max_mem-1000)[0]
		if num == 1:
			chosen_id = int(np.random.choice(ids, 1)[0])
			print("setting to gpu:%d" % chosen_id)
			sys.stdout.flush()
			torch.cuda.set_device(chosen_id)
			return torch.device("cuda:%d" % chosen_id)
		else:
			return np.random.choice(ids, num, replace=False)
	else:
		return


def pack_training_data_one_process(
		config, raw_dir, chrom, reorder,
		mask_row=None,
		off_diag=None,
		fac_size=None,
		merge_fac_row=1, merge_fac_col=1,
		is_sym=True,
		filename_pattern='%s_sparse_adj.npy',
		force_shift=None,
):
	
	filename = filename_pattern % chrom
	a = np.load(os.path.join(raw_dir, filename), allow_pickle=True)[reorder]
	try:
		blacklist = np.load(os.path.join(temp_dir, "raw", "blacklist.npy"), allow_pickle=True).item()
	except:
		print("no black list")
		blacklist = None
	
	if blacklist is not None:
		bl = blacklist[chrom]
		bl = bl[bl < a[0].shape[0]]
		print("num of bl", bl.shape)
		new_sparse_list = []
		for m in tqdm(a):
			m = m.astype('float32').toarray()
			m[bl, :] = 0.0
			m[:, bl] = 0.0
			new_sparse_list.append(csr_matrix(m))
		a = new_sparse_list
		
	coverage = np.fromiter((m.nnz for m in a), dtype=int)
	if fac_size is None: fac_size = max(1, int(300000 / config['resolution']))
	
	matrix_list = [m.tocoo() for m in a]
	if filename != f'{chrom}_sparse_adj.npy':
		# matrix_list_intra = np.load(os.path.join(raw_dir, f'{chrom}_sparse_adj.npy'), allow_pickle=True)[reorder]
		# matrix_list_intra = [m.tocoo() for m in matrix_list_intra]
		matrix_list_intra = a
	else:
		matrix_list_intra = a
	# if not is_sym: matrix_list = [m.transpose() for m in matrix_list]
	del a

	if merge_fac_row > 1 or merge_fac_col > 1:
		for m in matrix_list:
			m.col //= merge_fac_col
			m.row //= merge_fac_row
			m.sum_duplicates()
			m.resize(list((np.array(m.shape) + (merge_fac_row, merge_fac_col) - 1) // (merge_fac_row, merge_fac_col)))

	bulk = calc_bulk(matrix_list)
	bin_id_mapping_row, num_bins_row, bin_id_mapping_col, num_bins_col = filter_bin(bulk=bulk, is_sym=is_sym)

	if is_sym:
		num_bins = num_bins_row
		bin_id_mapping = bin_id_mapping_row
	else: num_bins, bin_id_mapping = None, None
	if off_diag is None: off_diag = int(50000000 / config['resolution'])
	if is_sym:
		matrix_list = normalize_per_cell(
			matrix_list, matrix_list_intra=matrix_list_intra, bulk=None,
			per_cell_normalize_func=[
				normalize_by_coverage,
			],
		)
	if not is_sym:
		matrix_list = normalize_per_cell(
			matrix_list, matrix_list_intra=matrix_list_intra, bulk=None,
			per_cell_normalize_func=[
				normalize_by_coverage,
				lambda *x: normalize_by_coverage(*x, scale=1),
			],
		)
		matrix_list = normalize_per_cell(
			matrix_list, matrix_list_intra=matrix_list_intra, bulk=None,
			normalizers=[
				# Clip(axis='entry', s=5.),
				Clip(axis='row', s=5.),
			],
		)

	assert not any(np.isnan(m.data).any() for m in matrix_list)

	nnz = sum(m.nnz for m in matrix_list)
	indices = np.empty([3, nnz], dtype=np.int16)
	values = np.empty([nnz], dtype=np.float32)
	del nnz

	# if off_diag <= num_bins-1:
	if is_sym and force_shift is not False and (
			off_diag * 2 < num_bins - 1 or
			(isinstance(fac_size, int) and fac_size > 1) or
			force_shift is True):
	# if isinstance(fac_size, int) and fac_size > 1 and is_sym:
		# 2: od - 0 + 1 // 2
		# 3: od - 1 + 2 // 3
		# 4: od - 1 + 3 // 4
		off_diag = min(off_diag, num_bins-1)
		col_offset = (off_diag + fac_size//2) // fac_size
		shape = (num_bins, col_offset * 2 + 1)
		do_shift = True
	elif (isinstance(fac_size, int) and fac_size == 1) or is_sym:
		shape = (num_bins_row, num_bins_col)
		col_offset = None
		do_shift = False
	else:
		assert (np.diff(fac_size) >= 0).all()
		assert (np.diff(fac_size) <= 1).all()
		assert len(fac_size) >= num_bins
		assert fac_size[0] == 0
		col_offset = fac_size[num_bins-1]
		shape = (num_bins, col_offset * 2 + 1)
		do_shift = True

	idx_nnz = 0
	for i, m in enumerate(tqdm(matrix_list)):
		if m.nnz == 0: continue
		row, col, data = bin_id_mapping_row[m.row], bin_id_mapping_col[m.col], m.data
		col_new = col - row
		idx = (row != -1) & (col != -1) & (col_new >= -off_diag) & (col_new <= off_diag)
		# assert idx.sum() > 0
		row, col, data = row[idx], col_new[idx], data[idx]
		if isinstance(fac_size, int) and fac_size == 1: pass
		elif isinstance(fac_size, int) and fac_size > 1:
			col += fac_size // 2
			if fac_size % 2 == 0: col[col <= 0] -= 1
			col //= fac_size
			offset = col.min()
			col -= offset
			tmp = coo_matrix((data, (row, col)), shape=(row.max()+1, col.max()+1))
			tmp.sum_duplicates()
			row, col, data = tmp.row, tmp.col + offset, tmp.data
			del tmp, offset
		else:
			col_sgn = np.sign(col)
			col = col_sgn * fac_size[np.abs(col)]

		if do_shift: col += col_offset
		else: col += row

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
	print (shape, do_shift)
	assert indices.min() >= 0
	assert (indices.max(1) < shape).all()
	gc.collect()

	# if is_sym or False:
	# 	values = np.log1p(values)
	#
	if do_shift:
		for i in trange(shape[1]):
			mask = indices[1] == i
			temp = values[mask]
			if is_sym: s = 15
			else: s = 15
			if s is not None:
				mean_, std_ = np.mean(temp), np.std(temp)
				# print("mean", "std", "max", "0.9q", mean_, std_, np.max(temp), np.quantile(temp, 0.9))
				# print("# of outliers:", np.sum(temp > (mean_ + s * std_)), "# of nonzero terms", len(temp))
				values[mask] = np.clip(temp, a_min=None, a_max=mean_ + s * std_)
	else:
		for i in trange(shape[1]):
			mask = np.abs(indices[1] - indices[0]) == i
			temp = values[mask]
			if is_sym: s = 15
			else: s = 15
			if s is not None:
				mean_, std_ = np.mean(temp), np.std(temp)
				# print("mean", "std", "max", "0.9q", mean_, std_, np.max(temp), np.quantile(temp, 0.9))
				# print("# of outliers:", np.sum(temp > (mean_ + s * std_)), "# of nonzero terms", len(temp))
				values[mask] = np.clip(temp, a_min=None, a_max=mean_ + s * std_)

	return indices, values, shape


def preprocess_meta(config, kept=None):
	data_dir = Path(config['data_dir'])
	temp_dir = Path(config['temp_dir'])

	if kept is None:
		try:
			kept = np.load(temp_dir / 'raw' / 'kept.npy', allow_pickle=True)
			print(f'{sum(kept)} out of {len(kept)} cells are kept')
		except Exception as e:
			print(e)
			kept = slice(None)

	with open(os.path.join(data_dir, "label_info.pickle"), "rb") as f: label_info = pd.DataFrame(pickle.load(f))
	try: plot_label = config['plot_label']
	except: plot_label = []
	try:
		print(f"batch key = {config['batch_id']}")
		batch_id = label_info[config['batch_id']].values
	except Exception as e:
		print ("no batch", e)
		batch_id = np.zeros(len(label_info), dtype=int)
	# batch_id = np.zeros(len(label_info), dtype=int)
	
	unique_data_ids = np.unique(batch_id)
	print('batches =', unique_data_ids)
	reorder = np.concatenate([np.where(batch_id == d)[0] for d in unique_data_ids], axis=0)
	
	if not isinstance(kept, slice): reorder = reorder[kept[reorder]]
	print (reorder)
	batch_id = batch_id[reorder]
	label_info = label_info.iloc[reorder].reset_index()
	sig_list = label_info[plot_label]

	index = [0] + list(np.nonzero(batch_id[:-1] != batch_id[1:])[0] + 1) + [len(batch_id)]
	data_list = [slice(*_) for _ in zip(index[:-1], index[1:])]

	print(data_list)
	gc.collect()

	return label_info, reorder, data_list, sig_list, batch_id


def preprocess_contact_map(config, reorder, path2input_cache, key_fn=lambda c: c, **kwargs):
	print(f'cache file = {path2input_cache}')
	do_cache = path2input_cache is not None

	if do_cache and os.path.exists(path2input_cache):
		print(f'loading cached input from {path2input_cache}')
		with open(path2input_cache, 'rb') as f: all_matrix = pickle.load(f)
		return all_matrix

	chrom_list = config['chrom_list']

	all_matrix = {key_fn(chrom): pack_training_data_one_process(
		config=config, raw_dir=Path(config['temp_dir']) / 'raw', chrom=chrom, reorder=reorder,
		# mask_row = mask1[chrom_list1 == chrom],
		**kwargs,
	) for chrom in chrom_list}
	all_matrix = {chrom: Sparse(indices, values, shape) for chrom, (indices, values, shape) in all_matrix.items()}
	for obj in tqdm(all_matrix.values()): obj.sort_indices()

	if do_cache:
		print(f'saving cached input to {path2input_cache}')
		sys.stdout.flush()
		with open(path2input_cache, 'wb') as f: pickle.dump(all_matrix, f)

	sys.stdout.flush()
	return all_matrix


if __name__ == '__main__':
	import time
	print(time.ctime())
	args = parse_args()
	config = get_config(args.config)
	chrom_list = config['chrom_list']
	temp_dir = config['temp_dir']
	import h5py
	
	with h5py.File(os.path.join(temp_dir, "node_feats.hdf5"), "r") as input_f:
		cell_feats1 = np.array(input_f['cell2weight'])
		
	gpu_id = get_free_gpu()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	label_info, reorder, data_list, sig_list, batch_id = preprocess_meta(config, kept=None)
	cell_feats1 = cell_feats1[reorder]
	all_matrix = dict()
	path2input_cache = args.path2input_cache
	if not os.path.exists(path2input_cache):
		os.mkdir(path2input_cache)
	path2input_cache_intra = os.path.join(path2input_cache, 'cache_intra.pkl')
	path2input_cache_inter = os.path.join(path2input_cache, 'cache_inter.pkl')

	method = args.method
	path2result_dir = args.path2result_dir
	if not os.path.exists(path2result_dir):
		os.mkdir(path2result_dir)
	if method == 'PARAFAC2':
		if args.use_intra:
			all_matrix.update(preprocess_contact_map(
				config, reorder=reorder, path2input_cache=path2input_cache_intra, is_sym=True,
				off_diag=50,
				fac_size=1,
				merge_fac_row=1, merge_fac_col=1,
				filename_pattern='%s_sparse_adj.npy',
				force_shift=False,
			))
		if args.use_inter:
			all_matrix.update(preprocess_contact_map(
				config, reorder=reorder, path2input_cache=path2input_cache_inter, is_sym=False,
				off_diag=np.inf,
				fac_size=1,
				merge_fac_row=2, merge_fac_col=40,
				filename_pattern='%s_sparse_inter_adj.npy',
				force_shift=None,
				key_fn=lambda c: f'{c}_inter',
			))
		
		shape_list = np.stack([all_matrix[obj].shape[:-1] for obj in all_matrix])
		print (shape_list, np.sum(shape_list[:, 0]))
	else: raise NotImplementedError
	start = time.time()
	
	if method == 'PARAFAC2':
		analysis_pool = ProcessPoolExecutor(max_workers=32)
		p_list = []
		for dim1, rank, n_iter_parafac, trial in itertools.product(
				[.3],
				[64],
				[1],
				range(1),
		):
			if isinstance(dim1, float):
				size_list = (shape_list[:, 0] * dim1).astype(int)
			else:
				size_list = np.ones(len(shape_list), dtype='int') * dim1
			print('size_list', size_list)
			print('total size', np.sum(size_list))

			save_str = "dim1_%.1f_rank_%d_niterp_%d_trial_%d_%s" % (dim1, rank, n_iter_parafac, trial, args.extra)
			print(save_str)
			start = time.time()

			result, error_rate, dim_list = fast_higashi_integrative(
				list(all_matrix.values()), rank=rank, n_iter_max=100,
				tol=5e-4, random_state=None, verbose=False,
				return_errors=False, n_iter_parafac=n_iter_parafac,
				l2_reg=args.l2reg,
				size_list=size_list,
				data_list=data_list,
				share_factors=args.share_factors,
				output_dir=path2result_dir,
				do_conv=args.do_conv,
				do_rwr=args.do_rwr,
				# label_list=None if args.label_list is None else label_info[args.label_list]
			)
			weights_all, factors_all, projection_matrices_all = result
			cell_embeddings = factors_all[-1].detach().cpu().numpy()
			embedding_projectors = [[f.detach().cpu().numpy() for f in factor] for factor in factors_all[-2]]
			A_list = [[f.detach().cpu().numpy() for f in factor] for factor in factors_all[0]]
			B_list = [[f.detach().cpu().numpy() for f in factor] for factor in factors_all[1]]
			D_list = [[f.detach().cpu().numpy() for f in factor] for factor in factors_all[2]]
			p_list = [[f.detach().cpu().numpy() for f in factor] for factor in projection_matrices_all]

			with open(os.path.join(path2result_dir, "results_all%s.pkl" % save_str), "wb") as f:
				pickle.dump([A_list, B_list, D_list, cell_embeddings, p_list], f)

			fac = cell_embeddings
			fac2 = embedding_projectors
			with open(os.path.join(path2result_dir, "results%s.pkl" % save_str), "wb") as f:
				pickle.dump([fac, fac2], f)
			print("takes: %.2f s" % (time.time() - start))

			with open(os.path.join(path2result_dir, "results%s.pkl" % save_str), "rb") as f:
				result = pickle.load(f)
			fac, fac2 = result[0], result[1]
	else:
		raise NotImplementedError(method)
	print(time.ctime())
