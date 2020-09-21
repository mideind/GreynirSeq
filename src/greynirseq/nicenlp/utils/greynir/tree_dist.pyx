#!python
#cython: language_level=3
import cython
from cython.parallel import parallel, prange
from cython.view cimport array as cvarray

from libc.stdlib cimport malloc, free

cimport openmp

import numpy as np
cimport numpy as np

try:
    from icecream import ic
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

ctypedef np.float32_t DTYPE_t

def tree_dist(tree1, tree2, ignore):
    t1_postorder, t2_postorder = tree1.to_postfix(), tree2.to_postfix()
    strings = list(set(t1_postorder + t2_postorder + ([] if ignore is None else [ignore])))
    ignore = strings.index(ignore) if ignore is not None else -1
    # determine necessary dtype at runtime (uint16 or uint8 for cache efficieny)
    # we need a separate dist function for uint8 then
    t1_postorder = [strings.index(it) for it in t1_postorder]
    t2_postorder = [strings.index(it) for it in t2_postorder]
    lr_kr_1, l_1 = LR_keyroots(tree1)
    lr_kr_2, l_2 = LR_keyroots(tree2)
    buffer_width = max(len(t1_postorder), len(t2_postorder)) + 1
    return tree_distance_cyx(
        np.zeros((buffer_width, buffer_width), dtype=np.float32),  # tree dist buffer
        np.zeros((buffer_width, buffer_width), dtype=np.float32),  # forest dist buffer
        np.array(t1_postorder, dtype=np.short),
        np.array(lr_kr_1, dtype=np.short),
        np.array(l_1, dtype=np.short),
        np.array(t2_postorder, dtype=np.short),
        np.array(lr_kr_2, dtype=np.short),
        np.array(l_2, dtype=np.short),
        ignore,
    )


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef tree_dist_multi(trees1, trees2, verbose=False, ignore=None):
    cdef size_t num_seqs = len(trees1)

    t1s_py = []
    t2s_py = []
    kr1s_py = []
    kr2s_py = []
    l1s_py = []
    l2s_py = []

    cdef:
        size_t[:] t1_idxs = np.zeros(2 * num_seqs, dtype=np.uintp)
        size_t[:] t2_idxs = np.zeros(2 * num_seqs, dtype=np.uintp)
        size_t[:] kr1_idxs = np.zeros(2 * num_seqs, dtype=np.uintp)
        size_t[:] kr2_idxs = np.zeros(2 * num_seqs, dtype=np.uintp)
        size_t[:] l1_idxs = np.zeros(2 * num_seqs, dtype=np.uintp)
        size_t[:] l2_idxs = np.zeros(2 * num_seqs, dtype=np.uintp)

        int buffer_width = -1

    cdef size_t seq_idx = 0
    for seq_idx in range(num_seqs):
        t1, kr1, l1, t2, kr2, l2 = _convert(trees1[seq_idx], trees2[seq_idx], ignore=ignore)
        buffer_width = max(buffer_width, len(t1))
        buffer_width = max(buffer_width, len(t2))

        t1_idxs[2 * seq_idx] = len(t1s_py)
        kr1_idxs[2 * seq_idx] = len(kr1s_py)
        l1_idxs[2 * seq_idx] = len(l1s_py)
        t2_idxs[2 * seq_idx] = len(t2s_py)
        kr2_idxs[2 * seq_idx] = len(kr2s_py)
        l2_idxs[2 * seq_idx] = len(l2s_py)

        t1s_py.extend(t1)
        kr1s_py.extend(kr1)
        l1s_py.extend(l1)
        t2s_py.extend(t2)
        kr2s_py.extend(kr2)
        l2s_py.extend(l2)

        l1_idxs[2 * seq_idx + 1] = len(l1s_py)
        t1_idxs[2 * seq_idx + 1] = len(t1s_py)
        kr1_idxs[2 * seq_idx + 1] = len(kr1s_py)
        t2_idxs[2 * seq_idx + 1] = len(t2s_py)
        kr2_idxs[2 * seq_idx + 1] = len(kr2s_py)
        l2_idxs[2 * seq_idx + 1] = len(l2s_py)

    buffer_width = buffer_width + 1
    # print("buffer_width", buffer_width)

    cdef short[:] t1s = np.array(t1s_py, dtype=np.short)
    cdef short[:] t2s = np.array(t2s_py, dtype=np.short)
    cdef short[:] kr1s = np.array(kr1s_py, dtype=np.short)
    cdef short[:] kr2s = np.array(kr2s_py, dtype=np.short)
    cdef short[:] l1s = np.array(l1s_py, dtype=np.short)
    cdef short[:] l2s = np.array(l2s_py, dtype=np.short)

    cdef float[:] accum_tree_dists = np.zeros(num_seqs, dtype=np.float32)

    cdef float[:, :, :] tree_dist_buffer = np.zeros((num_seqs, buffer_width, buffer_width), dtype=np.float32)
    cdef float[:, :, :] forest_dist_buffer = np.zeros((num_seqs, buffer_width, buffer_width), dtype=np.float32)

    cdef short ignore_ = -1 if ignore is None else ignore

    # mean 0.0944 std 0.011 min 0.064 at 1000 iters,  24 trees
    # this means 25% reduction over using single tree_dist with uncythonized iteration over trees
    for seq_idx in prange(num_seqs, nogil=True):
        accum_tree_dists[seq_idx] = tree_distance_cyx(
            tree_dist_buffer[seq_idx],
            forest_dist_buffer[seq_idx],
            t1s[t1_idxs[2 * seq_idx] : t1_idxs[2 * seq_idx + 1]],
            kr1s[kr1_idxs[2 * seq_idx] : kr1_idxs[2 * seq_idx + 1]],
            l1s[l1_idxs[2 * seq_idx] : l1_idxs[2 * seq_idx + 1]],
            t2s[t2_idxs[2 * seq_idx] : t2_idxs[2 * seq_idx + 1]],
            kr2s[kr2_idxs[2 * seq_idx] : kr2_idxs[2 * seq_idx + 1]],
            l2s[l2_idxs[2 * seq_idx] : l2_idxs[2 * seq_idx + 1]],
            ignore_,
        )

    return np.asarray(accum_tree_dists)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline float tree_distance_cyx(
    float[:, :] tree_dist_buffer,
    float[:, :] forest_dist_buffer,
    short[:] t1,
    short[:] t1_lr_kr,
    short[:] t1_l,
    short[:] t2,
    short[:] t2_lr_kr,
    short[:] t2_l,
    short ignore,
) nogil:

    # t1 and t2 are node labels (as int) of the trees in post-order format
    cdef size_t n = len(t1)
    cdef size_t m = len(t2)
    tree_dist_buffer[:n + 2, :m + 2] = 0.0
    forest_dist_buffer[:n + 2, :m + 2] = 0.0
    # if verbose:
    #     print(type(n))

    cdef size_t ii, jj
    for i_outer in range(len(t1_lr_kr)):
        for j_outer in range(len(t2_lr_kr)):
            ii = t1_lr_kr[i_outer]
            jj = t2_lr_kr[j_outer]
            forest_dist(tree_dist_buffer, forest_dist_buffer, t1, t1_l, t2, t2_l, ii, jj, ignore)

    # if verbose:
    #     ic(treedist)
    #     ic("end")
    return tree_dist_buffer[n, m]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void forest_dist(
    float[:, :] treedist,
    float[:, :] forestdist,
    short[:] t1,
    short[:] t1_l,
    short[:] t2,
    short[:] t2_l,
    size_t ii,
    size_t jj,
    short ignore,
) nogil:
    # Operates inplace on treedist and forestdist buffers
    cdef size_t ni, nj, li, lj, i1, j1
    li, lj = t1_l[ii - 1], t2_l[jj - 1]
    ni, nj = ii - li + 2, jj - lj + 2
    for i1 in range(li, ii + 1):
        forestdist[<size_t> (i1 - li + 1), 0] = forestdist[<size_t>(i1 - li), 0] + cost(t1, t2, i1, 0, ignore)
    for j1 in range(lj, jj + 1):
        forestdist[0, <size_t>(j1 - lj + 1)] = forestdist[0, <size_t>(j1 - lj)] + cost(t1, t2, 0, j1, ignore)

    for i1 in range(li, ii + 1):
        for j1 in range(lj, jj + 1):
            if t1_l[<size_t>(i1 - 1)] == t1_l[<size_t>(ii - 1)] and t2_l[<size_t>(j1 - 1)] == t2_l[<size_t>(jj - 1)]:
                forestdist[<size_t>((i1 - li + 1)), <size_t>(j1 - lj + 1)] = min(
                    forestdist[<size_t>(i1 - li), <size_t>(j1 - lj + 1)] + cost(t1, t2, i1, 0, ignore),  # remove
                    forestdist[<size_t>(i1 - li + 1), <size_t>(j1 - lj)] + cost(t1, t2, 0, j1, ignore),  # insert
                    forestdist[<size_t>(i1 - li), <size_t>(j1 - lj)] + cost(t1, t2, i1, j1, ignore),  # substitute
                )
                treedist[i1, j1] = forestdist[<size_t>(i1 - li + 1), <size_t>(j1 - lj + 1)]
            else:
                forestdist[<size_t>(i1 - li + 1), <size_t>(j1 - lj + 1)] = min(
                    forestdist[<size_t>(i1 - li), <size_t>(j1 - lj + 1)] + cost(t1, t2, i1, 0, ignore),  # remove
                    forestdist[<size_t>(i1 - li + 1), <size_t>(j1 - lj)] + cost(t1, t2, 0, j1, ignore),  # insert
                    forestdist[<size_t>(i1 - li), <size_t>(j1 - lj)] + treedist[i1, j1],  # substitute
                )


def LR_keyroots(tree):
    """Returns a tuple of:
    keyroots: LR_keyroots as defined in Zhang & Sasha,
            {k | there exists no k'>k such that l(k) = l(k')}
                where l(k) is the leftmost descendant of k
    leftmost_child: a list where the ith entry is the
        postfix ordering of its leftmost descendant"""
    cdef list keyroots = []
    cdef dict l_dict = {}
    cdef list l_list = []
    # assert isinstance(tree, Node)

    # pf means postfix
    def LR_keyroots_inner(tree, is_left_child=False, pf=0):
        l_desc = -1
        # for idx, child in enumerate(tree.children):
        # TODO: refactor so we can also choose to compare terminals
        for idx, child in enumerate(
            [child for child in tree.children if child.nonterminal]
        ):
            pf, ld = LR_keyroots_inner(child, idx == 0, pf)
            if idx == 0:
                l_desc = ld
        pf += 1
        if not is_left_child:
            keyroots.append(pf)
        if l_desc == -1:
            l_desc = pf
        l_list.append(l_desc)
        return pf, l_desc

    _, _ = LR_keyroots_inner(tree)
    return keyroots, l_list


def _convert(tree1, tree2, ignore=None):
    t1_postorder, t2_postorder = tree1.to_postfix(), tree2.to_postfix()
    strings = list(set(t1_postorder + t2_postorder + ([] if ignore is None else [ignore])))
    ignore = strings.index(ignore) if ignore is not None else ignore
    # determine necessary dtype at runtime (uint16 or uint8 for cache efficieny)
    # we need a separate dist function for uint8 then
    t1_postorder = [strings.index(it) for it in t1_postorder]
    t2_postorder = [strings.index(it) for it in t2_postorder]
    lr_kr_1, l_1 = LR_keyroots(tree1)
    lr_kr_2, l_2 = LR_keyroots(tree2)
    return (
        # np.array(t1_postorder, dtype=np.short),
        # np.array(lr_kr_1, dtype=np.short),
        # np.array(l_1, dtype=np.short),
        # np.array(t2_postorder, dtype=np.short),
        # np.array(lr_kr_2, dtype=np.short),
        # np.array(l_2, dtype=np.short),
        t1_postorder,
        lr_kr_1,
        l_1,
        t2_postorder,
        lr_kr_2,
        l_2,
    )


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline float cost(
    short[:] t1,
    short[:] t2,
    size_t a,
    size_t b,
    short ignore
) nogil:
    if ignore >= 0:
        if (t1[a - 1] == ignore) or (t2[b - 1] == ignore):
            return 0
    if (a == 0) or (b == 0):
        return int(a != b)
    return int(t1[a - 1] != t2[b - 1])
