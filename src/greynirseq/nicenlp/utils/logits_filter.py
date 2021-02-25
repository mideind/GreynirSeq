import torch
import torch.nn.functional as F

from greynirseq.utils.ifd_utils import CAT_GROUPS, LABEL_GROUPS


def word_classes_to_mask(word_class_tensor, mask_groups=CAT_GROUPS, n_labels_grps=len(LABEL_GROUPS)):
    """
    word_classes: tensor [bsz, word_class]
    returns tensor [bsz, word, label_groups]
    """
    # Handle padding
    mask_groups.append([])
    n_word_clasess = len(mask_groups)
    word_class_tensor_cp = word_class_tensor.clone()

    train = False
    if word_class_tensor_cp[word_class_tensor == -1].int().sum():
        train = True

    word_class_tensor_cp[word_class_tensor == -1] = n_word_clasess
    if train:
        word_class_tensor_cp = word_class_tensor_cp - 1

    word_class_mask = word_class_tensor_cp.new_zeros(n_word_clasess, n_labels_grps)
    for i in range(n_word_clasess):
        word_class_mask[i, :] = F.one_hot(word_class_tensor_cp.new_tensor(mask_groups[i]), n_labels_grps).sum(dim=0)
    wc_one_hot = F.one_hot(word_class_tensor_cp, n_word_clasess).type(word_class_tensor.dtype)

    # todo: check if needed
    del word_class_tensor_cp
    return torch.mm(wc_one_hot.float(), word_class_mask.float()).type(word_class_tensor.dtype)


def filter_logits(logits, unique_together):
    _, _, label_c = logits.shape
    n_hot = F.one_hot(unique_together, label_c).sum(dim=0)
    mutex_logits = logits * n_hot
    return mutex_logits


def filter_max_logits(logits, unique_together):
    _, _, label_c = logits.shape
    n_hot = F.one_hot(unique_together, label_c).sum(dim=0)
    mutex_logits = logits * n_hot
    _, max_idx = mutex_logits.max(dim=-1)
    max_hot = F.one_hot(max_idx, label_c)
    indexes = torch.ones(label_c, dtype=int)
    logits_sans_non_max_hits = logits * (indexes - n_hot + max_hot)
    return logits_sans_non_max_hits


def max_tensor_by_bins(tensor, bins, softmax_by_bin=False):
    """
    Expects a tensor and a list of bin start and end tuples, returns a new tensor
    with index of max value in each bin.
    """
    n_bins = len(bins)
    bsz, labels = tensor.shape
    max_tensor = tensor.new_zeros(bsz, labels)
    for i in range(n_bins):
        bin_start, bin_end = bins[i][0], bins[i][-1]
        bin_tensors = tensor[:, bin_start : bin_end + 1]  # noqa
        if softmax_by_bin:
            if bin_end - bin_start > 0:
                bin_tensors = F.softmax(bin_tensors)
            else:
                bin_tensors = torch.sigmoid(bin_tensors)

        max_in_bin = bin_tensors.max(dim=-1)
        for j in range(len(max_in_bin[0])):
            max_tensor[j, bin_start + max_in_bin[1][j]] = max_in_bin[0][j]
    return max_tensor
