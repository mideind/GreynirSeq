#
# Based on fairseq/tasks/translation.py that has the following license
#
#    Copyright (c) Facebook, Inc. and its affiliates.
#
#    This source code is licensed under the MIT license found in the
#    LICENSE file in the root directory of this source tree.

from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np
import torch

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import load_langpair_dataset, TranslationTask
from fairseq.data.noising import (
    NoisingDataset, UnsupervisedMTNoising, WordShuffle, WordDropout, WordNoising
)
from fairseq.data.encoders.utils import get_whole_word_mask

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


class DynamicNoisingDataset(NoisingDataset):
    def __init__(
        self,
        src_dataset,
        src_dict,
        seed,
        noiser=None,
        noising_class=UnsupervisedMTNoising,
        epoch=1,
        **kwargs
    ):
        super().__init__(
            src_dataset, src_dict, seed, noiser=noiser, noising_class=noising_class, **kwargs
        )
        self.epoch = 1

    def __getitem__(self, index):
        """
           Same as super method, except includes epoch for RNG
        """
        src_tokens = self.src_dataset[index]
        has_bos = src_tokens[0] == self.src_dict.bos()
        has_eos = src_tokens[-1] == self.src_dict.eos()
        src_lengths = torch.LongTensor([len(src_tokens)])
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
        logger.debug('DynamicNoisingDataset.set_epoch: {}'.format(epoch))
        self.epoch = epoch

    @property
    def sizes(self):
        return self.src_dataset.sizes

class GPT2WordNoising(UnsupervisedMTNoising):
    """Same as WordNoising class, except handles GPT2 byte-level bpe
       instead of standard bpe. This assumes lookup-table is for dictionary bpes.
        I.e. not raw gpt2 bpe itself """
    def __init__(
            self,
            dictionary,
            mask_is_beginning_of_word,
            max_word_shuffle_distance,
            word_dropout_prob,
            word_blanking_prob,
            has_bos=False,
    ):
        super(GPT2WordNoising, self).__init__(
            dictionary,
            max_word_shuffle_distance,
            word_dropout_prob,
            word_blanking_prob,
        )
        self.dictionary = dictionary
        self.has_bos = has_bos
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
        if self.has_bos:
            y = x.clone()
            y[1:length_no_eos] = y[1:length_no_eos].flip(0)
            x = y
        else:
            x[:length_no_eos] = x[:length_no_eos].flip(0)
        noisy_src_tokens = super(GPT2WordNoising, self).noising(x, lengths)

        if self.has_bos:
            noisy_src_tokens[1:length_no_eos] = noisy_src_tokens[1:length_no_eos].flip(0)
        else:
            noisy_src_tokens[:length_no_eos] = noisy_src_tokens[:length_no_eos].flip(0)
        return noisy_src_tokens

def load_unpaired_langpair(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl,
        max_source_positions, max_target_positions,
        prepend_bos=False, truncate_source=False, append_source_id=False,
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

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

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

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

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    return src_dataset, tgt_dataset


@register_task('translation_with_backtranslation')
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
        parser.add_argument('--max-word-shuffle-distance', default=3.0, type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.2, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')
        parser.add_argument('--prepend-bos', action='store_true', help='prepend bos token to each sentence')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super(TranslationWithBacktranslationTask, self).__init__(args, src_dict, tgt_dict)

        if self.append_backtranslation_tag:
            self.bt_idx = self.src_dict.add_symbol('<bt>')
            self.tgt_dict.add_symbol('<bt>')

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
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            truncate_source=self.args.truncate_source,
            prepend_bos=self.args.prepend_bos,
        )

        # load backtranslation
        if is_train_subset:
            """
            noised vs unnoised valdation set? they might converge at different times
            """
            bt_src_dataset, bt_tgt_dataset = load_unpaired_langpair(
                data_path, "{}.bt".format(split), src, self.src_dict, tgt, self.tgt_dict,
                combine=combine, dataset_impl=self.args.dataset_impl,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                truncate_source=self.args.truncate_source,
                prepend_bos=self.args.prepend_bos,
            )
            if self.args.bpe == "gpt2":
                mask_is_beginning_of_word = get_whole_word_mask(self.args, self.source_dictionary)
                mask_is_beginning_of_word = mask_is_beginning_of_word.numpy().astype(np.bool)
                noiser = GPT2WordNoising(
                    self.src_dict,
                    mask_is_beginning_of_word,
                    self.args.max_word_shuffle_distance,
                    self.args.word_dropout_prob,
                    self.args.word_blanking_prob,
                    has_bos=self.args.prepend_bos,
                )
                bt_src_dataset = DynamicNoisingDataset(
                    bt_src_dataset,
                    self.src_dict,
                    seed=1,
                    noiser=noiser,
                )
            else:
                # standard bpe with @@ as continuation marker
                bt_src_dataset = DynamicNoisingDataset(
                    bt_src_dataset,
                    self.src_dict,
                    seed=1,
                    max_word_shuffle_distance=self.args.max_word_shuffle_distance,
                    word_dropout_prob=self.args.word_dropout_prob,
                    word_blanking_prob=self.args.word_blanking_prob,
                )
            if self.append_backtranslation_tag:
                bt_src_dataset = (
                    AppendTokenDataset(
                        AppendTokenDataset(
                            StripTokenDataset(bt_src_dataset, self.src_dict.eos())
                            , self.bt_idx
                        ),
                        self.src_dict.eos()
                    )
                )

            sample_ratios = [self.args.upsample_primary, 1]
            src_dataset = ConcatDataset([src_dataset, bt_src_dataset], sample_ratios)
            tgt_dataset = ConcatDataset([tgt_dataset, bt_tgt_dataset], sample_ratios)

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            align_dataset=None, eos=self.tgt_dict.eos(),
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != 'test'),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary,
                                   tgt_dict=self.target_dictionary,
                                   constraints=constraints)

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    @property
    def append_backtranslation_tag(self):
        return True
