import math, time, itertools, gc, os, pickle, sys, copy, json
import numpy as np, pandas as pd


from tqdm.auto import tqdm, trange

from sklearn.neighbors import NearestNeighbors
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix


def get_config(config_path = "./config.jSON"):
	c = open(config_path,"r")
	return json.load(c)


def shift_csr(a, u, v, m, n):
	a = a.tocsr()
	indptr = np.full(m + 1, a.indptr[-1])
	indptr[u] = a.indptr[:-1]
	np.minimum.accumulate(indptr[::-1], out=indptr[::-1])
	indices = v[a.indices]
	return csr_matrix((a.data, indices, indptr), shape=(m, n))


def shift_coo(a, u, v, m, n):
	a = a.tocoo()
	return coo_matrix((a.data, (u[a.row], v[a.col])), shape=(m, n))


def trim_sparse(a, lb=-np.inf, ub=np.inf):
	a.data[(a.data < lb) | (a.data > ub)] = 0.
	a.sum_duplicates()
	a.prune()
	return a


def load_data_frame(path2file, open_fn=open, delimiter=',', dtype=float):
	decode = lambda s: s if isinstance(s, str) else s.decode()
	with open_fn(path2file) as f:
		cols = decode(f.readline()).strip().split(delimiter)
		rows = []
		values = []
		for line in tqdm(f):
			line = decode(line).strip().split(delimiter)
			rows.append(line[0])
			values.append(list(map(dtype, line[1:])))
	return pd.DataFrame(
		data=np.array(values),
		index=pd.Series(data=rows, name=cols[0]),
		columns=cols[1:]
	)
