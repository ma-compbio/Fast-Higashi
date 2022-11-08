import torch
import torch.nn.functional as F
import time
import numpy as np
torch.backends.cudnn.benchmark = True
import torch.jit as jit

def slice_arrange_func(x, slice_,strips=30):
	num_bin = int(x.shape[1])
	index_dim0 = torch.tile(torch.arange(num_bin), (strips,))
	index_dim1 = (torch.arange(strips).reshape((-1, 1)) +
	              torch.arange(slice_.start, num_bin + slice_.start).reshape((1, -1))).reshape((-1))
	index_dim1_ = torch.repeat_interleave(torch.arange(strips), num_bin)
	filter = index_dim1 < x.shape[2]
	
	a = torch.zeros(x.shape[0], x.shape[1], strips).float().to(x.device)
	a[:, index_dim0[filter], index_dim1_[filter]] = x[:, index_dim0[filter], index_dim1[filter]]
	return a


def rec2tilte(x, strips):
	num_bin = int(x.shape[1])
	index_dim0 = torch.tile(torch.arange(num_bin), (strips,))
	index_dim1 = (torch.arange(strips).reshape((-1, 1)) +
	              torch.arange(num_bin).reshape((1, -1))).reshape((-1))
	index_dim1_ = torch.repeat_interleave(torch.arange(strips), num_bin)
	filter = index_dim1 < x.shape[2]
	
	a = torch.zeros(x.shape[0], x.shape[1], x.shape[1]+strips).float().to(x.device)
	a[:, index_dim0[filter], index_dim1[filter]] = x[:, index_dim0[filter], index_dim1_[filter]]
	return a

def tilte2rec(x, strips):
	num_bin = int(x.shape[1])
	index_dim0 = torch.tile(torch.arange(num_bin), (strips,))
	index_dim1 = (torch.arange(strips).reshape((-1, 1)) +
	              torch.arange(num_bin).reshape((1, -1))).reshape((-1))
	index_dim1_ = torch.repeat_interleave(torch.arange(strips), num_bin)
	filter = index_dim1 < x.shape[2]
	
	a = torch.zeros(x.shape[0], x.shape[1], strips).float().to(x.device)
	a[:, index_dim0[filter], index_dim1_[filter]] = x[:, index_dim0[filter], index_dim1[filter]]
	return a

@torch.no_grad()
# @jit.script
def partial_rwr(x: torch.Tensor,
                slice_start: int,
                slice_end: int,
                do_conv:bool,
                do_rwr:bool,
                do_col:bool,
                bin_cov:torch.Tensor=torch.ones(1),
                bin_cov_row:torch.Tensor=torch.ones(1),
                return_rwr_iter:bool=False,
                force_rwr_epochs:int=-1,
                final_transpose:bool=True,
                slice_arrange:bool=False,
				slice_arrange_size:int=100,
                **kw
                # compact=False,
                # flank:int=10,
                # max_dis:int=10
                ):
	
	# The slice_start / end has to be from the local_bin_slice_list not the global one.
	# slice_arrange: when true, after rwr, only store the elements that are within max_dis
	# The returned element would be of size: (bs_cell, bs_bin, 2*max_dis + 1)
	# compact: input tensor is in compact form or dense form.
	
	# if max_dis > flank and compact:
	# 	print ("max_dis has to be smaller than flank for compact matrix.")
	# 	raise EOFError
	
	n_iter = 0
	if do_conv or do_rwr:
		if do_conv and x.shape[1] > 1:
			pad = 1
			ll = pad * 2 + 1
			x = F.avg_pool2d(x[:, None], ll, 1, padding=pad, ceil_mode=True).clamp_(min=1e-8)
			x = x[:, 0, :, :]
		
		if do_rwr:
			A = x
			local_sim_2nd = (torch.bmm(A, A.permute(0, 2, 1)))  # size of (bs, #bin1, #bin1)
			t = torch.diagonal(local_sim_2nd, dim1=-2, dim2=-1)
			t.zero_()
			local_sim_1st = A[:, :, slice_start:slice_end].clone()

			local_sim_2nd = (local_sim_2nd.div_(local_sim_2nd.sum(1, keepdim=True).add_(1e-15))) * 0.25
			local_sim_1st = (local_sim_1st.div_(local_sim_1st.sum(1, keepdim=True).add_(1e-15))) * 0.75
			
			local_sim = local_sim_1st.add_(local_sim_2nd)
			
			# # fill entries with zero coverage with 1
			t = torch.diagonal(local_sim, dim1=-2, dim2=-1)
			t[:] += local_sim.sum(1) == 0
			
			if force_rwr_epochs < 0:
				auto_stop = True
				rwr_epochs = 60
			else:
				rwr_epochs = force_rwr_epochs
				auto_stop = False
			
			# Copy paste code here for speed optimization
			# For input A of size x
			# P, Q, Q_new: 3x
			rp = 0.5
			ngene = local_sim.shape[1]
			P = local_sim.div_(local_sim.sum(1, keepdim=True).add_(1e-15))
			Q = torch.eye(ngene, device=P.device)[None]
			Q = Q.repeat(local_sim.shape[0], 1, 1)
			epoch_count = 0
			for i in range(rwr_epochs):
				Q_new = rp * torch.bmm(Q, P)
				t = torch.diagonal(Q_new, dim1=-2, dim2=-1)
				t.add_(1 - rp)
				if auto_stop:
					delta = ((Q - Q_new).square_()).sum(dim=1).sum(dim=1).sqrt_()
					Q = Q_new
					if torch.max(delta) < 0.01:
						break
				else:
					Q = Q_new
				epoch_count += 1
			
			local_sim = Q
			n_iter = epoch_count
			
			if do_col:
				local_sim = (local_sim + local_sim.permute(0, 2, 1)) * 0.5
				local_sim = local_sim.clamp_(min=0.0)
				local_sim = local_sim.div_(local_sim.sum(2, keepdim=True).add_(1e-15))
				A = A.div_(bin_cov[:, None, :].to(A.device))
				
			
			x = torch.bmm(local_sim, A)

		# if slice_arrange:
		# 	# currect x: (bs_cell, bs_bin, all_bin or flank...)
		# 	num_bin = int(x.shape[1])
		#
		#
		#
		# 	if compact:
		# 		# When compact, it's the easier, because the main diag is always flank~flank+bs_bin, flank~flank+bs_bin
		# 		# plus we require that max_dis < flank, we don't need to worry negative index or outof boundary index
		# 		row_index = torch.repeat_interleave(torch.arange(num_bin, device=x.device), 2*max_dis+1)
		# 		# It'll be [0,0,0,...0,1,1,...,1,2,2,...,2...]
		#
		# 		# for col, slices or (2*max_dis+1), needs to add flank such that starts at the true main diag, minus max_dis, such center is main_diag
		# 		# Then for each row, increase index by one...
		# 		col_index = (torch.arange(flank-max_dis, flank-max_dis+2*max_dis+1, device=x.device)).reshape((1, -1)) + \
		# 		            torch.arange(num_bin, device=x.device).reshape((-1, 1))
		# 		col_index = col_index.reshape((-1))
		# 		# Should be [0,1,2,3..., 1,2,3..,2,3,...] correspond to strips of first, second...
		#
		# 	else:
		# 		# When not compact things is a little more complicated. We will have cases where the value is out of index(too left or too right)
		# 		row_index = torch.repeat_interleave(torch.arange(num_bin, device=x.device), 2 * max_dis + 1)
		# 		# It'll be [0,0,0,...0,1,1,...,1,2,2,...,2...]
		# 		# For col, it should start at slice_start - max_dis ends at slice_start+max_dis+1
		# 		col_index = (torch.arange(slice_start - max_dis, slice_start+max_dis+1).reshape((1, -1)) +
		# 		                   torch.arange(num_bin, device=x.device).reshape((-1, 1)))
		# 		col_index = col_index.reshape((-1)).clamp_(min=0, max=x.shape[2]-1)
		# 		# Main different is that, we add clamp to address out of index problem.
		# 		# One Caveat, it will now copy out of index values instead of putting it as 0
		# 	x = x[:, row_index, col_index]
			
		if final_transpose:
			x = x.permute(1, 2, 0)
	if return_rwr_iter:
		return x, n_iter
	return x, 0


