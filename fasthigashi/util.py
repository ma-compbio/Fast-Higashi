import math, time, itertools, gc, os, pickle, sys, copy
import numpy as np, pandas as pd

import torch
from tqdm.auto import tqdm, trange

from sklearn.neighbors import NearestNeighbors
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix


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


def is_oom_error(exception: BaseException) -> bool:
    print (isinstance(exception, RuntimeError), len(exception.args), "CUDA" in exception.args[0], "out of memory" in exception.args[0])
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        # This is the last thing that should cause an OOM error, but seemingly it can.
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        if not is_oom_error(exception):
            # Only handle OOM errors
            raise