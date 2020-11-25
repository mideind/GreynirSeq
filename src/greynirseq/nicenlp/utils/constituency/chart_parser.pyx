#!python
#cython: language_level=3
import cython
from cython.parallel import parallel, prange
from cython.view cimport array as cvarray
# from cython cimport integral, floating, numeric
from libc.stdlib cimport malloc, free

cimport openmp

import numpy as np
cimport numpy as np

import torch

from collections import namedtuple
try:
    from icecream import ic
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


NP_BOOLTYPE = np.uint8

ctypedef float float32
ctypedef double float64
ctypedef int int32
ctypedef long int64

cdef struct max1d_f32_res_t:
    float32 value
    int32 index

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline max1d_f32_res_t max_argmax_1d_f32(float[:] arr, int nitems) nogil:
    cdef:
        int ii = 0
        float val = arr[0]
        int idx = 0
    for ii in range(1, nitems):
        if arr[ii] > val:
            val = arr[ii]
            idx = ii
    cdef max1d_f32_res_t result
    result.value = val
    result.index = idx
    return result


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef parse_many(
    scores,
    nwords,
    allow_null_root=False,
):
    cdef bsz, max_nrows, max_ncols, nlabels = 0
    bsz, max_nrows, max_ncols, nlabels = list(scores.shape)[:4]
    cdef:
        float[:, :, :, :] c_scores
        long[:] c_nwords

    if isinstance(scores, torch.Tensor) and scores.dtype == torch.float32:
        c_scores = scores.numpy()
        # c_scores = np.ascontiguousarray(scores.float(), order='C')
    elif isinstance(scores, torch.Tensor):
        # c_scores = np.array(scores.float(), order='C')
        c_scores = scores.float().numpy()
    else:
        c_scores = scores

    if isinstance(nwords, torch.Tensor):
        c_nwords = np.array(nwords.long(), order='C')
    else:
        c_nwords = nwords

    cdef:
        int idx, nrows, ncols = 0
        float tree_score = 0
        int[:, :] mask
        float[:] res = np.zeros(bsz, dtype=np.float32)
        int[:, :, :] masks = np.zeros((bsz, max_nrows, max_ncols), dtype=np.int32)
        unsigned char[:, :, :, :] lmasks = np.zeros((bsz, max_nrows, max_ncols, nlabels), dtype=NP_BOOLTYPE)
        unsigned char[:, :, :] lmask

        float[:] tree_scores = np.zeros(bsz, dtype=np.float32)
        list lspans = []

    for idx in range(bsz):
        nrows = c_nwords[idx]
        ncols = nrows + 1
        tree_score, seq_lspans, mask, lmask = parse_single(
            c_scores[idx, :nrows, :ncols, :]
        )
        tree_scores[idx] = tree_score
        masks[idx, :nrows, :ncols] = mask
        lmasks[idx, :nrows, :ncols, :] = lmask
        lspans.append(torch.from_numpy(np.asarray(seq_lspans)))

    return (
        torch.from_numpy(np.asarray(tree_scores)),
        lspans,
        torch.from_numpy(np.asarray(masks)),
        torch.from_numpy(np.asarray(lmasks))
    )


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef parse_single(
    float[:, :, :] scores,
    allow_null_root=False,
):
    # walk diagonal line starting at position (0, start)
    cdef:
        int nrows = scores.shape[0]
        int ncols = scores.shape[1]  # 1 more than nwords
        int nlabels = scores.shape[2]
        float[:, :] chart = np.zeros((nrows, ncols), dtype=np.float32)
        float[:, :] best_chart = chart.copy()
        int[:, :] lchart = np.zeros((nrows, ncols), dtype=np.int32)
        int[:, :] kchart = lchart.copy()
        int idx, ii, jj, kk = 0
        float max_root_lscore, min_root_lscore, root_delta = 0.0
        max1d_f32_res_t max_res

    if not allow_null_root:
        # root position in tree cannot have label index 0
        max_root_lscore = max(scores[nrows - 1, ncols - 1, :])
        min_root_lscore = min(scores[nrows - 1, ncols - 1, :])
        root_delta = max(max_root_lscore, 0.0) - min_root_lscore + 1
        scores[0, ncols - 1, 0] = -root_delta
    with nogil:
        for ii in prange(ncols):
            for jj in range(ii + 1, ncols):
                max_res = max_argmax_1d_f32(scores[ii, jj, :], nlabels)
                lchart[ii, jj] = max_res.index
                chart[ii, jj] = max_res.value

    cdef:
        int start, end, span_length, best_split_idx
        float label_score, best_split_score
        float[:] best_split_scores = np.zeros(ncols, dtype=np.float32)

    for span_length in range(1, ncols):
        for start in range(ncols - span_length):
            end = start + span_length
            label_score = chart[start, end]
            if span_length == 1:
                best_chart[start, end] = label_score
                continue

            for kk in range(start + 1, end):
                best_split_scores[kk - start - 1] = best_chart[start, kk] + best_chart[kk, end]

            max_res = max_argmax_1d_f32(best_split_scores, end - start)
            best_split_score, best_split_idx  = max_res.value, max_res.index
            kchart[start, end] = best_split_idx
            best_chart[start, end] = label_score + best_split_score

    cdef:
        float tree_score = best_chart[0, ncols - 1]
        int max_nspans = ncols * (ncols) // 2
        int[:] stack_ii = np.zeros(max_nspans, dtype=np.int32)
        int[:] stack_jj = np.zeros(max_nspans, dtype=np.int32)
        int[:, :] lspans = np.zeros((max_nspans, 4), dtype=np.int32)
        int[:, :] mask = np.zeros((nrows, ncols), dtype=np.int32)
        int sortkey = 0
        int nspans = 0

    idx = 0
    stack_ii[idx] =  0
    stack_jj[idx] =  ncols - 1
    idx = idx + 1
    while idx > 0:
        idx = idx - 1
        ii, jj = stack_ii[idx], stack_jj[idx]
        if ii + 1 == jj:
            sortkey = ncols * ii + (ncols - jj)
            lspans[nspans, 0] = sortkey
            lspans[nspans, 1] = ii
            lspans[nspans, 2] = jj
            nspans = nspans + 1
            continue
        kk = kchart[ii, jj] + 1
        sortkey = ncols * ii + (ncols - jj)
        lspans[nspans, 0] = sortkey
        lspans[nspans, 1] = ii
        lspans[nspans, 2] = jj
        nspans = nspans + 1

        stack_ii[idx] = ii
        stack_jj[idx] = ii + kk
        stack_ii[idx + 1] = ii + kk
        stack_jj[idx + 1] = jj
        idx = idx + 2

    cdef unsigned char[:, :, :] lmask = np.zeros((nrows, ncols, nlabels), dtype=NP_BOOLTYPE)
    # insert labels
    for idx in range(nspans):
        ii, jj = lspans[idx, 1], lspans[idx, 2]
        lspans[idx, 3] = lchart[ii, jj]
        mask[ii, jj] = 1

        lmask[ii, jj, lchart[ii, jj]] = 1

    in_order_permute = np.argsort(lspans[:nspans, 0])
    return (
        tree_score,
        np.asarray(lspans[:nspans])[in_order_permute, 1:],
        np.asarray(mask),
        np.asarray(lmask, dtype=NP_BOOLTYPE)
    )
