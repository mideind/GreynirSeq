#
# Based on fairseq/tasks/translation.py that has the following license
#
#    Copyright (c) Facebook, Inc. and its affiliates.
#
#    This source code is licensed under the MIT license found in the
#    LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace
from functools import lru_cache

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    BaseWrapperDataset,
    ConcatDataset,
    Dictionary,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.data.noising import NoisingDataset, UnsupervisedMTNoising, WordDropout, WordNoising, WordShuffle
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


class DynamicGPT2BPEDropoutResampling(BaseWrapperDataset):
    """Reencode input dataset encoded GPT2-BPE, with bpe-dropout.
    Note that currently Huggingface tokenizers library does not support reproducible pseudo-rng."""

    def __init__(
        self,
        args,
        dataset,
        source_dictionary,
        dropout=0.1,
        seed=1,
    ):
        super().__init__(dataset)
        self.source_dictionary = source_dictionary
        self.epoch = 0
        self.seed = seed
        self.dropout = dropout

        import tokenizers

        self.hf_tokenizer = tokenizers.ByteLevelBPETokenizer(
            args.gpt2_encoder_json,
            args.gpt2_vocab_bpe,
            add_prefix_space=True,
            dropout=self.dropout,
        )

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item_old = self.dataset[index]
            unk_sym = self.source_dictionary.symbols[self.source_dictionary.unk()]
            parts = self.source_dictionary.string(item_old).split(unk_sym)
            hf_bpe_tokens_str = []
            for part in parts:
                part_bpe_str_old = [int(elem) for elem in part.split(" ") if elem]  # filter trailing whitespace
                part_bpe_str_new = " ".join(
                    map(
                        str,
                        self.hf_tokenizer.encode(self.hf_tokenizer.decode(part_bpe_str_old)).ids,
                    )
                )
                hf_bpe_tokens_str.append(part_bpe_str_new)
            item = self.source_dictionary.encode_line(unk_sym.join(hf_bpe_tokens_str), add_if_not_exist=False).long()

            if item_old[0] == self.source_dictionary.bos():
                item = torch.cat([torch.tensor([self.source_dictionary.bos()]), item])

            return item

    def set_epoch(self, epoch, **_unused):
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)


class DynamicNoisingDataset(NoisingDataset):
    def __init__(
        self, src_dataset, src_dict, seed, noiser=None, noising_class=UnsupervisedMTNoising, epoch=1, **kwargs
    ):
        super().__init__(src_dataset, src_dict, seed, noiser=noiser, noising_class=noising_class, **kwargs)
        self.epoch = 1

    def __getitem__(self, index):
        """
        Same as super method, except includes epoch for RNG
        """
        src_tokens = self.src_dataset[index]
        has_bos = src_tokens[0] == self.src_dict.bos()
        has_eos = src_tokens[-1] == self.src_dict.eos()
        src_lengths = torch.tensor([len(src_tokens)]).long()
        src_tokens = src_tokens.unsqueeze(0)

        # Transpose src tokens to fit expected shape of x in noising function
        # (batch size, sequence length) -> (sequence length, batch size)
        src_tokens_t = torch.t(src_tokens)

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            noisy_src_tokens = self.noiser.noising(src_tokens_t, src_lengths)

        # Transpose back to expected src_tokens format
        # (sequence length, 1) -> (1, sequence length)
        noisy_src_tokens = torch.t(noisy_src_tokens)
        noisy_src_tokens = noisy_src_tokens[0]
        if has_bos and noisy_src_tokens[0] != self.src_dict.bos():
            noisy_src_tokens = torch.cat([torch.tensor([self.src_dict.bos()]), noisy_src_tokens])
        if has_eos and noisy_src_tokens[-1] != self.src_dict.eos():
            noisy_src_tokens = torch.cat([noisy_src_tokens, torch.tensor([self.src_dict.eos()])])
        return noisy_src_tokens

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        logger.debug("DynamicNoisingDataset.set_epoch: {}".format(epoch))
        self.epoch = epoch

    @property
    def sizes(self):
        return self.src_dataset.sizes


class GPT2WordNoising(UnsupervisedMTNoising):
    """Same as UnsupervisedMTNoising class, except handles GPT2 byte-level bpe
    instead of standard bpe. This assumes lookup-table is for dictionary bpes.
     I.e. not raw gpt2 bpe itself"""

    def __init__(
        self,
        dictionary,
        mask_is_beginning_of_word,
        max_word_shuffle_distance,
        word_dropout_prob,
        word_blanking_prob,
    ):
        super(GPT2WordNoising, self).__init__(
            dictionary,
            max_word_shuffle_distance,
            word_dropout_prob,
            word_blanking_prob,
        )
        self.dictionary = dictionary
        self.word_dropout = WordDropout(
            dictionary=dictionary,
            bpe_cont_marker=None,
            bpe_end_marker="$",
        )
        self.word_shuffle = WordShuffle(
            dictionary=dictionary,
            bpe_cont_marker=None,
            bpe_end_marker="$",
        )
        self.bpe_end = mask_is_beginning_of_word
        self.word_dropout.bpe_end = self.bpe_end
        self.word_shuffle.bpe_end = self.bpe_end
        self.get_word_idx = self._get_gpt2_bpe_word_idx

    def _get_gpt2_bpe_word_idx(self, x):
        if x.dim() == 2:
            x = x.flip(0)
        else:
            x = x.flip(0)

        raise NotImplementedError()
        return self._get_bpe_word_idx(x)

    def noising(self, x, lengths):
        length_no_eos = len(x) - 1 if x[-1] == self.dictionary.eos() else len(x)
        has_bos = x[0] == self.dictionary.bos()
        if has_bos:
            y = x.clone()
            y[1:length_no_eos] = y[1:length_no_eos].flip(0)
            x = y
        else:
            x[:length_no_eos] = x[:length_no_eos].flip(0)
        noisy_src_tokens = super(GPT2WordNoising, self).noising(x, lengths)

        if has_bos:
            noisy_src_tokens[1:length_no_eos] = noisy_src_tokens[1:length_no_eos].flip(0)
        else:
            noisy_src_tokens[:length_no_eos] = noisy_src_tokens[:length_no_eos].flip(0)
        return noisy_src_tokens


class GPT2Noising(WordNoising):
    """Same as WordNoising root class, except handles GPT2 byte-level bpe
    instead of standard bpe. This assumes lookup-table is for dictionary bpes.
     I.e. not raw gpt2 bpe itself"""

    def __init__(
        self,
        dictionary,
        mask_is_beginning_of_word,
    ):
        super(GPT2Noising, self).__init__(dictionary, bpe_cont_marker=None, bpe_end_marker=",")
        self.dictionary = dictionary
        self.bpe_end = mask_is_beginning_of_word


class GPT2WordDropout(GPT2Noising):
    """Similar to WordDropout except for GPT2BPE and uses single blanks for
    dropped 'words'"""

    def __init__(self, dictionary, mask_is_beginning_of_word, default_dropout_prob=0.1):
        super().__init__(dictionary, mask_is_beginning_of_word)
        self.default_dropout_prob = default_dropout_prob

    def noising(self, x, lengths, dropout_prob=None, blank_idx=None):
        if dropout_prob is None:
            dropout_prob = self.default_dropout_prob
        # x: (T x B), lengths: B
        if dropout_prob == 0:
            return x, lengths

        assert 0 < dropout_prob < 1

        # be sure to drop entire words
        word_idx = self.get_word_idx(x)
        sentences = []
        modified_lengths = []
        for i in range(lengths.size(0)):
            # Since dropout probabilities need to apply over non-pad tokens,
            # it is not trivial to generate the keep mask without consider
            # input lengths; otherwise, this could be done outside the loop

            # We want to drop whole words based on word_idx grouping
            num_words = max(word_idx[:, i]) + 1

            # ith example: [x0, x1, ..., eos, pad, ..., pad]
            # We should only generate keep probs for non-EOS tokens. Thus if the
            # input sentence ends in EOS, the last word idx is not included in
            # the dropout mask generation and we append True to always keep EOS.
            # Otherwise, just generate the dropout mask for all word idx
            # positions.
            has_eos = x[lengths[i] - 1, i] == self.dictionary.eos()
            if has_eos:  # has eos?
                keep = np.random.rand(num_words - 1) >= dropout_prob
                keep = np.append(keep, [True])  # keep EOS symbol
            else:
                keep = np.random.rand(num_words) >= dropout_prob

            words = x[: lengths[i], i].tolist()

            # TODO: speed up the following loop
            # drop words from the input according to keep
            # new_s = [
            #     w if keep[word_idx[j, i]] else blank_idx
            #     for j, w in enumerate(words)
            # ]
            # By Haukur; use single blank token per word instead of per bpe token
            new_s = []
            last_word_idx = None
            for j, w in enumerate(words):  # bpe tokens
                this_word_idx = word_idx[j, i]
                if keep[this_word_idx]:
                    new_s.append(w)
                elif last_word_idx != this_word_idx:
                    new_s.append(blank_idx)
                last_word_idx = this_word_idx
            new_s = [w for w in new_s if w is not None]
            # we need to have at least one word in the sentence (more than the
            # start / end sentence symbols)
            if len(new_s) <= 1:
                # insert at beginning in case the only token left is EOS
                # EOS should be at end of list.
                new_s.insert(0, words[np.random.randint(0, len(words))])
            assert len(new_s) >= 1 and (
                not has_eos  # Either don't have EOS at end or last token is EOS
                or (len(new_s) >= 2 and new_s[-1] == self.dictionary.eos())
            ), "New sentence is invalid."
            sentences.append(new_s)
            modified_lengths.append(len(new_s))
        # re-construct input
        modified_lengths = torch.tensor(modified_lengths).long()
        modified_x = torch.full((int(modified_lengths.max().item()), modified_lengths.size(0)), self.dictionary.pad())
        for i in range(modified_lengths.size(0)):
            modified_x[: modified_lengths[i], i].copy_(torch.tensor(sentences[i]).long())

        return modified_x, modified_lengths


class GPT2WordShuffle(GPT2Noising):
    """Same as WordShuffle except for GPT2BPE"""

    def __init__(self, dictionary, mask_is_beginning_of_word, default_max_shuffle_distance=3):
        super().__init__(dictionary, mask_is_beginning_of_word)
        self.default_max_shuffle_distance = 3

    def noising(self, x, lengths, max_shuffle_distance=None):
        if max_shuffle_distance is None:
            max_shuffle_distance = self.default_max_shuffle_distance
        # x: (T x B), lengths: B
        if max_shuffle_distance == 0:
            return x, lengths

        # max_shuffle_distance < 1 will return the same sequence
        assert max_shuffle_distance > 1

        # define noise word scores
        noise = np.random.uniform(
            0,
            max_shuffle_distance,
            size=(x.size(0), x.size(1)),
        )
        if x[0, 0] == self.dictionary.bos():
            noise[0] = -1  # do not move start sentence symbol
        # be sure to shuffle entire words
        word_idx = self.get_word_idx(x)
        x2 = x.clone()
        for i in range(lengths.size(0)):
            length_no_eos = lengths[i]
            if x[lengths[i] - 1, i] == self.dictionary.eos():
                length_no_eos = lengths[i] - 1
            # generate a random permutation
            scores = word_idx[:length_no_eos, i] + noise[word_idx[:length_no_eos, i], i]
            # ensure no reordering inside a word
            scores += 1e-6 * np.arange(length_no_eos.item())
            permutation = scores.argsort()
            # shuffle words
            x2[:length_no_eos, i].copy_(x2[:length_no_eos, i][torch.from_numpy(permutation)])
        return x2, lengths


class GPT2WordNoisingV2(GPT2Noising):
    """Same as UnsupervisedMTNoising class, except handles GPT2 byte-level bpe
    instead of standard bpe. This assumes lookup-table is for dictionary bpes.
     I.e. not raw gpt2 bpe itself"""

    def __init__(
        self,
        dictionary,
        mask_is_beginning_of_word,
        max_word_shuffle_distance,
        word_dropout_prob,
        word_blanking_prob,
    ):
        super(GPT2WordNoisingV2, self).__init__(
            dictionary,
            mask_is_beginning_of_word,
        )
        self.max_word_shuffle_distance = max_word_shuffle_distance
        self.word_dropout_prob = word_dropout_prob
        self.word_blanking_prob = word_blanking_prob
        self.dictionary = dictionary
        self.word_dropout = GPT2WordDropout(
            dictionary,
            mask_is_beginning_of_word,
        )
        self.word_shuffle = GPT2WordShuffle(
            dictionary,
            mask_is_beginning_of_word,
        )
        self.bpe_end = mask_is_beginning_of_word

    def noising(self, x, lengths):
        has_eos = x[-1] == self.dictionary.eos()
        length_no_eos = len(x) - 1 if has_eos else len(x)
        has_bos = x[0] == self.dictionary.bos()
        start_idx = 0 if not has_bos else 1
        y = x.clone()
        y[start_idx:length_no_eos] = y[start_idx:length_no_eos].flip(0)
        x = y

        # 1. Word Shuffle
        noisy_src_tokens, noisy_src_lengths = self.word_shuffle.noising(
            x=x,
            lengths=lengths,
            max_shuffle_distance=self.max_word_shuffle_distance,
        )
        # 2. Word Dropout
        noisy_src_tokens, noisy_src_lengths = self.word_dropout.noising(
            x=noisy_src_tokens,
            lengths=noisy_src_lengths,
            dropout_prob=self.word_dropout_prob,
        )
        # 3. Word Blanking
        noisy_src_tokens, noisy_src_lengths = self.word_dropout.noising(
            x=noisy_src_tokens,
            lengths=noisy_src_lengths,
            dropout_prob=self.word_blanking_prob,
            blank_idx=self.dictionary.unk(),
        )

        noisy_src_length = noisy_src_lengths[0]
        noisy_src_length = noisy_src_length - 1 if has_eos else noisy_src_length
        noisy_src_tokens[start_idx:noisy_src_length] = noisy_src_tokens[start_idx:noisy_src_length].flip(0)
        return noisy_src_tokens


def load_unpaired_langpair(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    truncate_source=False,
    append_source_id=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError("Dataset not found: {} ({})".format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info("{} {} {}-{} {} examples".format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index("[{}]".format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index("[{}]".format(tgt)))

    return src_dataset, tgt_dataset


@register_task("translation_with_backtranslation")
class TranslationWithBacktranslationTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        super(TranslationWithBacktranslationTask, TranslationWithBacktranslationTask).add_args(parser)
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--skip-backtranslation-data', default=False, action='store_true', help='Should we skip reading the backtranslation data? \
This is useful to set if you have no backtranslation data but would like the BPE noise on the main training data.')  # noqa
        parser.add_argument('--max-word-shuffle-distance', default=3.0, type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.1, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')
        parser.add_argument('--prepend-bos', action='store_true', help='prepend bos token to each sentence')
        parser.add_argument('--tagged-backtranslation', action='store_true', help='do not append tag to backtranslated sentences during training')  # noqa
        parser.add_argument('--bpe-dropout', type=float, default=0.0, metavar='N', help='Set GPT2-BPE dropout amount')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super(TranslationWithBacktranslationTask, self).__init__(args, src_dict, tgt_dict)

        self.bt_idx = self.src_dict.add_symbol("<bt>")
        self.tgt_dict.add_symbol("<bt>")
        self.src_dict.pad_to_multiple_(8)
        self.tgt_dict.pad_to_multiple_(8)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        is_train_subset = split == getattr(self.args, "train_subset", None)
        if not is_train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        """
        this is mask_word_initial
        WordNoising uses mask_word_end or mask_bpe_cont
        probably easiest to write FlippedDataset that reverses sequences
        and use the standard pipeline

        load_langpair_dataset:
            find files by pattern
            load_indexed source
                maybe truncate
                load target
            check shard counts
            sample ratios
            bos, source_id
            load_alignments
            LangpairDataset constructor

        """

        src_dataset, tgt_dataset = load_unpaired_langpair(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            truncate_source=self.args.truncate_source,
            prepend_bos=self.args.prepend_bos,
        )

        if self.args.bpe_dropout > 0:
            src_dataset = DynamicGPT2BPEDropoutResampling(
                self.args,
                src_dataset,
                self.source_dictionary,
                dropout=self.args.bpe_dropout,
            )

        # load backtranslation
        if is_train_subset and not self.args.skip_backtranslation_data:
            """
            noised vs unnoised valdation set? they might converge at different times
            """
            bt_src_dataset, bt_tgt_dataset = load_unpaired_langpair(
                # data_path, "{}.bt".format(split), src, self.src_dict, tgt, self.tgt_dict,
                data_path,
                "{}.bt".format(split),
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.args.dataset_impl,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                truncate_source=self.args.truncate_source,
                prepend_bos=self.args.prepend_bos,
            )
            if self.args.bpe == "gpt2":
                mask_is_beginning_of_word = get_whole_word_mask(self.args, self.source_dictionary)
                mask_is_beginning_of_word = mask_is_beginning_of_word.numpy().astype(np.bool)
                # noiser = GPT2WordNoising(
                #     self.src_dict,
                #     mask_is_beginning_of_word,
                #     self.args.max_word_shuffle_distance,
                #     self.args.word_dropout_prob,
                #     self.args.word_blanking_prob,
                # )
                if self.args.bpe_dropout > 0:
                    bt_src_dataset = DynamicGPT2BPEDropoutResampling(
                        self.args,
                        bt_src_dataset,
                        self.source_dictionary,
                        dropout=self.args.bpe_dropout,
                    )
                noiser = GPT2WordNoisingV2(
                    self.src_dict,
                    mask_is_beginning_of_word,
                    self.args.max_word_shuffle_distance,
                    self.args.word_dropout_prob,
                    self.args.word_blanking_prob,
                )
                bt_src_dataset = DynamicNoisingDataset(
                    bt_src_dataset,
                    self.src_dict,
                    seed=1,
                    noiser=noiser,
                )

                # try:
                #     from icecream import ic
                #     ic.configureOutput(includeContext=True)
                # except ImportError:  # Graceful fallback if IceCream isn't installed.
                #     ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa
                # ic("gpt2 bbpe")
                # bpe = encoders.build_bpe(self.args)
                # def decode(foo):
                #     return bpe.decode(self.src_dict.string(foo))
                # def disp(foo):
                #     return " ".join([bpe.decode(i) for i in self.src_dict.string(foo).split(" ")])
                #     # foo = [bpe.decode(str(i)) for i in range(0,1000)]
                #     # doo = [bpe.decode((i)) for i in self.src_dict.symbols[4:1000]]
                # for i in range(5):
                #     ic(_bt_src_dataset[i])
                #     ic(decode(_bt_src_dataset[i]))
                #     ic(disp(_bt_src_dataset[i]))
                #     ic(disp(bt_src_dataset[i]))
                #     ic(bt_src_dataset[i])
                # import pdb; pdb.set_trace()
            else:
                assert self.args.bpe_dropout <= 0, "BPE dropout not supported for this BPE scheme"
                # standard bpe with @@ as continuation marker
                bt_src_dataset = DynamicNoisingDataset(
                    bt_src_dataset,
                    self.src_dict,
                    seed=1,
                    max_word_shuffle_distance=self.args.max_word_shuffle_distance,
                    word_dropout_prob=self.args.word_dropout_prob,
                    word_blanking_prob=self.args.word_blanking_prob,
                )
            # if self.append_backtranslation_tag:
            if self.args.tagged_backtranslation:
                bt_src_dataset = AppendTokenDataset(
                    AppendTokenDataset(StripTokenDataset(bt_src_dataset, self.src_dict.eos()), self.bt_idx),
                    self.src_dict.eos(),
                )

            sample_ratios = [self.args.upsample_primary, 1]
            src_dataset = ConcatDataset([src_dataset, bt_src_dataset], sample_ratios)
            tgt_dataset = ConcatDataset([tgt_dataset, bt_tgt_dataset], sample_ratios)

        self.datasets[split] = LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            self.src_dict,
            tgt_dataset,
            tgt_dataset.sizes,
            self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            align_dataset=None,
            eos=self.tgt_dict.eos(),
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split not in ("test", "valid")),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, tgt_dict=self.target_dictionary, constraints=constraints
        )

    @classmethod
    def load_dictionary(cls, filename):
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<bt>")
        dictionary.pad_to_multiple_(8)
        return dictionary

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args)
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    # @property
    # def append_backtranslation_tag(self):
    #     return self._append_backtranslation_tag
