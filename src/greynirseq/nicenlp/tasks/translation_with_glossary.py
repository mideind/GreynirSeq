"""Translation with Glossary Task. Enables a model to leverage glossaries during translation."""
#
# Based on fairseq/tasks/translation.py that has the following license
#
#    Copyright (c) Facebook, Inc. and its affiliates.
#
#    This source code is licensed under the MIT license found in the
#    LICENSE file in the root directory of this source tree.

import argparse
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Container, Dict, List, Optional, Tuple

import jaro
import torch
from fairseq import utils
from fairseq.data import Dictionary, encoders
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.logging import metrics
from fairseq.tasks import register_task
from scipy.stats import poisson
from torch.functional import Tensor

from greynirseq.nicenlp.data.encoding import decode_line, encode_line, get_word_beginnings
from greynirseq.nicenlp.tasks.translation_with_backtranslation import TranslationWithBacktranslationTask
from greynirseq.nicenlp.utils.dictionary import ensure_symbols_are_present
from greynirseq.utils.lemmatizer import Lemmatizer, get_lemmatizer_for_lang

logger = logging.getLogger(__name__)

non_letters_numbers = re.compile(r"\W", flags=re.UNICODE | re.IGNORECASE)


@dataclass
class GlossaryTaskConfig:
    """ Configuration for the glossary task."""

    enabled: bool
    ok_to_increase_dict_size: bool
    constraint_positional_start_idx: int
    seq_sample_ratio: float
    mean_whole_word: float
    glossary_file: str
    glossary_subset_prefix: str
    fuzzy_match_threshold: int
    negative_example_ratio: float
    eval_glossary: bool

    @staticmethod
    def from_args(args: argparse.Namespace):
        """Create a config from the args object. Call this in task __init__"""
        # It is not ok to increase the dictionary size if we are loading a pretrained model.
        # Or when we are loading for inference
        # TODO: verify that this is the case.
        ok_to_increase_dict_size = "restore_file" not in args and "path" not in args
        return GlossaryTaskConfig(
            enabled=args.glossary_enabled,
            ok_to_increase_dict_size=ok_to_increase_dict_size,
            constraint_positional_start_idx=args.glossary_constraint_positional_start_idx,
            seq_sample_ratio=args.glossary_seq_sample_ratio,
            mean_whole_word=args.glossary_mean_whole_word,
            glossary_file=args.glossary_lookup_file,
            glossary_subset_prefix=args.glossary_glosspref,
            fuzzy_match_threshold=args.glossary_fuzzy_match_threshold,
            negative_example_ratio=args.glossary_negative_example_ratio,
            eval_glossary=args.glossary_eval,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add arguments to the parser for this task. Call this in the task's add_args()."""
        parser.add_argument(
            "--glossary-enabled",
            default=False,
            action="store_true",
            help="Should the glossary task be enabled? \
This will add soft constraints automatically during training and \
their effectiveness can be evaluated given a glossary.",
        )
        parser.add_argument(
            "--glossary-eval",
            default=False,
            action="store_true",
            help="Should we evaluate the glossary functionality on ALL the validation sets? \
This measures the how often a term is used in the source and how often the corresponding term translation \
is in the target?",
        )
        parser.add_argument(
            "--glossary-seq-sample-ratio",
            default=0.2,
            type=float,
            help="During training, soft constraints are automatically added to this fraction of sequences.",
        )
        parser.add_argument(
            "--glossary-mean-whole-word",
            default=3.0,
            type=float,
            help="During training, soft constraints are sampled from the target. \
The sampling selects whole-words (potentially multiple subwords). \
The sampling distribution is the poisson distribution and this parameter is the mean of that distribution.",
        )
        parser.add_argument(
            "--glossary-constraint-positional-start-idx",
            default=0,
            type=int,
            help="When constraints are added as a part of the input, their positional indices are changed by the task. \
This parameter sets the starting index of the positional indices of the constraints. \
Starting from 0 will 'restart' the positional indicies for the constraints.",
        )
        parser.add_argument(
            "--glossary-lookup-file",
            default="",
            type=str,
            help="The file which defines the glossary/lookup which should be used when evaluating the glossary functionality. \
The file should be in DATA_DIR, i.e. a relative path. \
This file is expected to be a tsv file with SRC *tab* TGT, where TGT is the translation of SRC. \
Currently, we only support single word SRCs. \
The glossary is used when we translate the validation/test subset defined in --glossary-glosspref. \
We perform a fuzzy search for SRC in the input. \
If the fuzzy match is above --glossary-fuzzy-match-threshold the corresponding TGT is added as a constraint.",
        )
        parser.add_argument(
            "--glossary-glosspref",
            default="",
            type=str,
            help="The subset name of the validation or testing data which should use the --glossary-lookup-file. \
See --glossary-lookup-file help for more details",
        )
        parser.add_argument(
            "--glossary-fuzzy-match-threshold",
            default=1.0,
            type=float,
            help="The fuzzy match threshold when searching for SRC in the input. \
If the fuzzy match is greater than the threshold, we append the corresponding glossary TGT as constraints. \
The threshold is in [0, 1.0], 1.0 means exact match. \
See --glossary-lookup-file help for more details",
        )
        parser.add_argument(
            "--glossary-negative-example-ratio",
            default=0.1,
            type=float,
            help="The fraction of negative constraint examples to use against positive examples.",
        )


def make_positions_with_constraints(
    tensor: Tensor,
    padding_idx: int,
    onnx_trace: bool = False,
    positional_idx_restart_offset=0,
    positional_marker_symbol_idx: Optional[int] = None,
):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.

    This code is based on fairseq's make_positions.
    The change allows the positional indices to be restarted when encountering the 'positional_marker_symbol_idx'.
    The positional indices after the restart are additionally offset by the 'positional_idx_restart_offset'.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    padding_mask = tensor.ne(padding_idx).int()
    positional_marker_mask = torch.ones_like(padding_mask)
    positional_marker_idx = torch.zeros_like(padding_mask).bool().nonzero()
    if positional_marker_symbol_idx is not None:
        positional_marker_idx = tensor.eq(positional_marker_symbol_idx).nonzero()
    # TODO: find a way to do this efficently using tensors
    for marker_idx in positional_marker_idx:
        positional_marker_mask[marker_idx[0], marker_idx[1] :] = 0
    before_positional_marker_mask = positional_marker_mask
    after_positional_marker_mask = (~positional_marker_mask.bool()).int()  # int -> bool -> negation -> int again

    # First the positions before the positional_marker_mask
    before_positional_marker_mask_and_padding = before_positional_marker_mask * padding_mask
    positions = (
        torch.cumsum(before_positional_marker_mask_and_padding, dim=1).type_as(padding_mask)
        * before_positional_marker_mask_and_padding
    )
    # Then the positions after the positional_marker_mask
    after_positional_marker_mask_and_padding = after_positional_marker_mask * padding_mask
    positions += (
        torch.cumsum(after_positional_marker_mask_and_padding, dim=1).type_as(padding_mask)
        + positional_idx_restart_offset
    ) * after_positional_marker_mask_and_padding
    positions += padding_idx
    return positions.long()


def apply_monkey_patch_for_make_positions(positional_marker_symbol_idx: int, positional_idx_restart_offset: int):
    """Monkey-patch the make_positions function in fairseq to be aware of the glossary constraints.
    The monkey patch only extends the original function."""
    utils.make_positions = partial(
        make_positions_with_constraints,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )


def read_glossary(path: Path) -> Dict[str, str]:
    """Read the glossary from the given file. Return an empty dict if it does not exist."""
    if not path.exists():
        return {}
    with path.open("r") as f:
        return {line.split("\t")[0]: line.split("\t")[1].strip() for line in f}


@register_task("translation_with_glossary")
class TranslationWithGlossaryTask(TranslationWithBacktranslationTask):
    """
    Translate from one (source) language to another (target) language with soft constraints.

    """

    def __init__(
        self,
        args: argparse.Namespace,
        src_dict: Dictionary,
        tgt_dict: Dictionary,
    ):
        super().__init__(args, src_dict, tgt_dict)  # type: ignore
        self.config = GlossaryTaskConfig.from_args(args)
        if self.config.enabled:
            logger.info("Glossary is ENABLED")
            logger.info(f"Glossary config: {self.config}")
        else:
            logger.info("Glossary is DISABLED")
        # Ensure that <sep> and <c> are defined in the dictionaries.
        self.added_extra_symbols = ["<c>", "<sep>"]
        ensure_symbols_are_present(
            self.source_dictionary,
            self.added_extra_symbols,
            self.config.ok_to_increase_dict_size,
        )
        ensure_symbols_are_present(
            self.target_dictionary,
            self.added_extra_symbols,
            self.config.ok_to_increase_dict_size,
        )

        assert (
            self.target_dictionary == self.source_dictionary
        ), "The target dictionary must be the same as the source dictionary, \
    because we use is_word_initial based on a single dictionary and use it for both src and tgt."
        is_word_initial = get_word_beginnings(args, self.source_dictionary)
        if is_word_initial is None:
            raise ValueError("The is_word_initial function is None.")
        self.is_word_initial = is_word_initial

        # Get the data path
        data_path = args.data.split(":")[0]
        self.glossary = read_glossary(Path(data_path) / self.config.glossary_file)

        # Apply the monkey patch for make_positions so that we glossary items are placed at the right position.
        apply_monkey_patch_for_make_positions(
            positional_marker_symbol_idx=self.source_dictionary.index("<sep>"),
            positional_idx_restart_offset=self.config.constraint_positional_start_idx,
        )
        self.bpe = encoders.build_bpe(args)

        self.src_lemmatizer = get_lemmatizer_for_lang(self.args.source_lang)
        self.tgt_lemmatizer = get_lemmatizer_for_lang(self.args.target_lang)

    @staticmethod
    def add_args(parser) -> None:
        """Add task-specific arguments to the parser."""
        super(TranslationWithGlossaryTask, TranslationWithGlossaryTask).add_args(parser)  # type: ignore
        GlossaryTaskConfig.add_args(parser)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs) -> None:
        """Load a given dataset split."""
        # Load the super stuff
        super().load_dataset(split=split, epoch=epoch, combine=combine, **kwargs)
        logger.debug(f"Split: {split}")
        if not self.config.enabled:
            return
        is_glossary_pref = split == self.config.glossary_subset_prefix
        is_train_subset = split == getattr(self.args, "train_subset", None)

        # We make sure that the loaded ds is LanguagePairDataset - since we make assumptions about the ds structure.
        ds = self.datasets[split]
        src_dataset = ds.src
        assert isinstance(ds, LanguagePairDataset)
        if is_train_subset:
            logger.debug("Training subset. Sampling whole words from target")
            # We sample whole words from the target side to use as constraints.
            constraints = SampleWholeWordDataset(
                dataset=ds.tgt,
                whole_word_masker=self.is_word_initial,
                seq_sample_ratio=self.config.seq_sample_ratio,
                mean_whole_word=self.config.mean_whole_word,
                pad_idx=self.target_dictionary.index("<pad>"),
                negative_example_ratio=self.config.negative_example_ratio,
                bpe=self.bpe,
            )
            logger.debug("Training subset. Appending constraints to SRC")
            # We append the constraints to the source.
            src_dataset = AppendConstraintsDataset(
                dataset=ds.src,
                constraints=constraints,
                sep_symbol_idx=self.target_dictionary.index("<sep>"),
                constraint_symbol_idx=self.target_dictionary.index("<c>"),
                max_seq_len=self.args.max_source_positions,
                bpe=self.bpe,
            )
        elif is_glossary_pref:
            logger.debug("Glossary subset. Fuzzy matching glossary SRC to input.")
            constraints = FuzzyGlossaryConstraintsDataset(
                dataset=ds.src,
                dictionary=self.source_dictionary,
                glossary=self.glossary,
                fuzzy_match_threshold=self.config.fuzzy_match_threshold,
                lemmatizer=self.src_lemmatizer,
                bpe=self.bpe,
            )
            src_dataset = AppendConstraintsDataset(
                dataset=ds.src,
                constraints=constraints,
                sep_symbol_idx=self.target_dictionary.index("<sep>"),
                constraint_symbol_idx=self.target_dictionary.index("<c>"),
                max_seq_len=self.args.max_source_positions,
                bpe=self.bpe,
            )
            logger.debug("Glossary subset. Appending constraints to SRC")

        self.datasets[split] = LanguagePairDataset(
            src_dataset,
            ds.src.sizes,
            ds.src_dict,
            ds.tgt,
            ds.tgt.sizes,
            ds.tgt_dict,
            left_pad_source=ds.left_pad_source,
            left_pad_target=ds.left_pad_target,
            align_dataset=ds.align_dataset,
            eos=ds.eos,
            num_buckets=self.args.num_batch_buckets,
            shuffle=ds.shuffle,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        """Build a dataset used for inference."""
        ds = super().build_dataset_for_inference(src_tokens, src_lengths, constraints=None)  # type: ignore
        if not self.config.enabled:
            return ds

        logger.debug("Using glossary for inference.")
        constraints = FuzzyGlossaryConstraintsDataset(
            dataset=ds.src,
            dictionary=self.source_dictionary,
            glossary=self.glossary,
            fuzzy_match_threshold=self.config.fuzzy_match_threshold,
            lemmatizer=self.src_lemmatizer,
            bpe=self.bpe,
        )
        src_dataset = AppendConstraintsDataset(
            dataset=ds.src,
            constraints=constraints,
            sep_symbol_idx=self.target_dictionary.index("<sep>"),
            constraint_symbol_idx=self.target_dictionary.index("<c>"),
            max_seq_len=self.args.max_source_positions,
            bpe=self.bpe,
        )
        return LanguagePairDataset(
            src_dataset,
            src_lengths,
            ds.src_dict,
            left_pad_source=ds.left_pad_source,
            left_pad_target=ds.left_pad_target,
            align_dataset=ds.align_dataset,
            eos=ds.eos,
            num_buckets=self.args.num_batch_buckets,
            shuffle=ds.shuffle,
        )


    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        # We have no way of knowing whether the sample is from glosspref or not...
        if self.config.eval_glossary:
            src_glossary_counts, tgt_glossary_counts = self.get_initial_glossary_counts()
            self._inference_with_glossary_counting(
                self.sequence_generator,  # type: ignore
                sample,
                model,
                src_glossary_counts=src_glossary_counts,
                tgt_glossary_counts=tgt_glossary_counts,
            )
            logging_output["_glossary_src_counts"] = src_glossary_counts
            logging_output["_glossary_tgt_counts"] = tgt_glossary_counts
        return loss, sample_size, logging_output

    def _inference_with_glossary_counting(
        self, generator, sample, model, tgt_glossary_counts: Dict[str, int], src_glossary_counts: Dict[str, int]
    ) -> None:
        """Take an inference step and update the glossary counts (in-place) with the number of terms in sample."""
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)

        def get_term_count_in_sents(
            sents: Tensor, glossary: Container[str], idx_dict: Dictionary, lemmatizer: Lemmatizer
        ) -> Dict[str, int]:
            decoded_sent = decode_line(
                sents,
                bpe=self.bpe,
                dictionary=idx_dict,
                extra_symbols_to_ignore=[
                    idx_dict.index(added_extra_symbol) for added_extra_symbol in self.added_extra_symbols
                ],
            )
            # We also lower-case the lemmas, since we count against lower-cased terms.
            lemmas = [lemma.lower() for lemma in lemmatizer.lemmatize(decoded_sent)]
            counts: Dict[str, int] = dict()
            for lemma in lemmas:
                if lemma in glossary:
                    counts[lemma] = counts.get(lemma, 0) + 1
            return counts

        for i in range(len(gen_out)):
            best_translation = gen_out[i][0]["tokens"]
            source = utils.strip_pad(sample["net_input"]["src_tokens"][i], self.source_dictionary.pad())
            src_term_counts = get_term_count_in_sents(
                source, self._glossary_lower.keys(), self.target_dictionary, self.src_lemmatizer
            )
            tgt_term_counts = get_term_count_in_sents(
                best_translation, self._glossary_lower.values(), self.source_dictionary, self.tgt_lemmatizer
            )
            for term, count in src_term_counts.items():
                src_glossary_counts[term] += count
                tgt_term_count_true_positive = tgt_term_counts.get(self._glossary_lower[term], 0)
                # We do not overcount the number of times a term is used in the target.
                tgt_glossary_counts[self._glossary_lower[term]] += min(tgt_term_count_true_positive, count)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.config.eval_glossary:
            total_src_counts = Counter(self.get_initial_glossary_counts()[0])
            total_tgt_counts = Counter(self.get_initial_glossary_counts()[1])
            for logging_output in logging_outputs:
                try:
                    total_src_counts.update(Counter(logging_output["_glossary_src_counts"]))
                    total_tgt_counts.update(Counter(logging_output["_glossary_tgt_counts"]))
                # reduce_metrics is also called during training and then we do not have access to these keys.
                except KeyError:
                    return

            def compute_glossary_accuracy(src_counts: Counter, tgt_counts: Counter) -> float:
                total_count = sum(src_counts.values())
                total_correct = sum(tgt_counts.values())
                try:
                    return round(total_correct / total_count, 4)
                except ZeroDivisionError:
                    return 0.0

            metrics.log_scalar("glossary_accuracy", compute_glossary_accuracy(total_src_counts, total_tgt_counts))


def apply_constraints(constraints: List[Tensor], t: Tensor, constraint_idx: int, sep_idx: int) -> Tensor:
    """
    Apply constraints to a source sequence.
    """
    # t = t_1, t_2, ..., t_n (t_i are subwords)
    # t = t + <sep>
    # We do not add the <sep> when there are no constraints
    if len(constraints) == 0:
        return t
    t = torch.cat((t, torch.Tensor([sep_idx]).long()), dim=0)
    for constraint in constraints:
        # t = t + c_i + <c>
        t = torch.cat((t, constraint, torch.Tensor([constraint_idx]).long()), dim=0)
    return t


def whole_word_sampling(
    example: Tensor,
    whole_word_masker: Dict[int, int],
    seq_sample_ratio: float,
    word_count_to_sample: float,
    contains_eos: bool,
) -> List[Tensor]:
    """
    Create sampling constraints.
    """
    sampled_whole_words: List[Tensor] = []
    if contains_eos:
        # We need to remove the eos symbol from the target sequence
        example = example[:-1]
    tgt_whole_word_mask = [whole_word_masker[elm] for elm in tensor_to_list(example)]
    # The number whole words in the target
    tgt_whole_word_count = sum(tgt_whole_word_mask)
    # The number of whole words in the target that we want to use as contraints
    word_count_to_sample = round(min(max(0, word_count_to_sample), tgt_whole_word_count))

    fraction_sequences_to_constrain = seq_sample_ratio
    # Should we use this example as a constraint?
    use_example_as_constraint = random.random() < fraction_sequences_to_constrain
    if not use_example_as_constraint and word_count_to_sample == 0:
        return sampled_whole_words
    # We sample the indices of the whole words we want to use.
    sampled_idxs = sorted(list(random.sample(range(tgt_whole_word_count), word_count_to_sample)))
    # The number of subwords in the whole words
    tgt_whole_word_lengths = whole_word_lengths(tgt_whole_word_mask)
    # The indices of the whole words in the original target
    tgt_whole_word_start_idxs = tensor_to_list(torch.Tensor(tgt_whole_word_mask).nonzero().squeeze())
    for sampled_idx in sampled_idxs:
        tgt_whole_word_start_idx = tgt_whole_word_start_idxs[sampled_idx]
        tgt_whole_word_length = tgt_whole_word_lengths[sampled_idx]
        sampled_whole_words.append(example[tgt_whole_word_start_idx : tgt_whole_word_start_idx + tgt_whole_word_length])
    return sampled_whole_words


def whole_word_lengths(whole_word_mask: List[int]) -> List[int]:
    """Return the masks length, i.e. a padding_mask=[1,0,0,1,1,0] should return [3,1,2]"""
    lengths: List[int] = []
    if len(whole_word_mask) == 0:
        return lengths
    mask_idxs_ten = (
        torch.Tensor(whole_word_mask).nonzero().squeeze()
    )  # For some reason, nonzero returns a tensor of size (1,2)
    mask_idxs = tensor_to_list(mask_idxs_ten)
    prev_idx: Optional[int] = None
    for idx in mask_idxs:
        if prev_idx is None:
            prev_idx = idx
            continue
        lengths.append(idx - prev_idx)
        prev_idx = idx
    if prev_idx is not None:
        lengths.append(len(whole_word_mask) - prev_idx)
    return lengths



class FuzzyGlossaryConstraintsDataset(BaseWrapperDataset):
    """A Dataset which lemmatizes and then fuzzy matches  glossary entries to 'dataset'.


    TODO: More explaination"""

    def __init__(
        self,
        dataset,
        dictionary: Dictionary,
        glossary: Dict[str, str],
        fuzzy_match_threshold: int,
        lemmatizer: Lemmatizer,
        bpe,
    ):
        super().__init__(dataset)
        self.glossary = glossary
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.dictionary = dictionary
        self.lemmatizer = lemmatizer
        self.bpe = bpe

    def __getitem__(self, idx):
        # Dim = (seq_len)
        example: Tensor = self.dataset[idx]

        sentence = decode_line(
            example,
            self.bpe,
            self.dictionary,
            extra_symbols_to_ignore=[self.dictionary.index("<sep>"), self.dictionary.index("<c>")],
        )
        # since the example is only a 1-dim tensor, the sentences should only contain one sentence
        sentence = " ".join(self.lemmatizer.lemmatize(sent=sentence))
        fuzzy_matches = fuzzy_match_glossary(sentence, glossary=self.glossary)
        glossary_constraints = [x[0] for x in filter(lambda x: x[1] >= self.fuzzy_match_threshold, fuzzy_matches)]
        logger.debug(f"Glossary Constraints: {glossary_constraints}")
        constraints = [encode_line(constraint, self.bpe, self.dictionary) for constraint in glossary_constraints]
        logger.debug(f"Original sentence: {sentence}")
        return constraints


class AppendConstraintsDataset(BaseWrapperDataset):
    """A Dataset class which appends constraints to 'dataset'.

    A separator symbol is inserted between the original example and the constraints.
    Between each constraint, a constraint symbol is inserted.
    If the example goes over the max_seq_len, the example is truncated.
    Assumes that the 'constraints' dataset returns a List[Tensor] when indexed."""

    def __init__(
        self,
        dataset: BaseWrapperDataset,
        constraints: BaseWrapperDataset,
        sep_symbol_idx: int,
        constraint_symbol_idx: int,
        max_seq_len: int,
        bpe,
    ):
        super().__init__(dataset)
        self.constraints = constraints
        self.sep_symbol_idx = sep_symbol_idx
        self.constraint_symbol_idx = constraint_symbol_idx
        self.max_seq_len = max_seq_len
        # Just for debugging now
        self.bpe = bpe

    def __getitem__(self, idx):
        # Dim = (seq_len)
        example: Tensor = self.dataset[idx]
        constraints: List[Tensor] = self.constraints[idx]
        random.shuffle(constraints)
        constraint_src = apply_constraints(constraints, example, self.constraint_symbol_idx, self.sep_symbol_idx)
        constraint_src = constraint_src[: self.max_seq_len]
        # logger.info(
        #     f"constraint_src: {self.bpe.decode(' '.join(str(x) for x in constraint_src.tolist() if x <= 32000))}"
        # )
        return constraint_src


class SampleWholeWordDataset(BaseWrapperDataset):
    """A Dataset class which samples whole words from 'dataset'.

    For each example, we sample a random number (0, 1] and if it is less than the 'seq_sample_ratio'
    then we will sample whole words from that example.
    The whole words samples are drawn based on a possion distribution.
    The mean is given by the 'mean_whole_word' parameter.
    Additionally, a fraction of the whole words are negative examples.

    This Dataset returns List[Tensor] when indexed."""

    def __init__(
        self,
        dataset: BaseWrapperDataset,
        whole_word_masker: Dict[int, int],
        seq_sample_ratio: float,
        mean_whole_word: float,
        pad_idx: int,
        bpe,
        include_negative_examples=True,
        negative_example_ratio=0.3,
    ):
        super().__init__(dataset)
        self.whole_word_masker = whole_word_masker
        self.seq_sample_ratio = seq_sample_ratio
        self.mean_whole_word = mean_whole_word
        self.pad_idx = pad_idx
        self.include_negative_examples = include_negative_examples
        self.negative_example_ratio = negative_example_ratio
        # Just for debugging now
        self.bpe = bpe

    def __getitem__(self, idx):
        # Dim = (seq_len)
        constraints: List[Tensor] = []
        sampled_whole_word_count = poisson.rvs(self.mean_whole_word)
        positive_whole_word_count = sampled_whole_word_count
        if self.include_negative_examples:
            negative_whole_word_mean = sampled_whole_word_count * self.negative_example_ratio
            negative_whole_word_count = min(poisson.rvs(negative_whole_word_mean), positive_whole_word_count)
            positive_whole_word_count -= negative_whole_word_count
            random_idx = random.choice(range(len(self.dataset)))
            negative_example = self.dataset[random_idx]
            constraints.extend(
                whole_word_sampling(
                    negative_example,
                    self.whole_word_masker,
                    self.seq_sample_ratio,
                    word_count_to_sample=positive_whole_word_count,
                    contains_eos=True,
                )
            )
        positive_example: Tensor = self.dataset[idx]
        constraints.extend(
            whole_word_sampling(
                positive_example,
                self.whole_word_masker,
                self.seq_sample_ratio,
                word_count_to_sample=positive_whole_word_count,
                contains_eos=True,
            )
        )
        return constraints

Translation = str
Source = str
Original = str
FuzzyResults = List[Tuple[Translation, float, Source, Original]]


def match_glossary(
    sentence: str, glossary: Dict[str, str], lemmatizer: Lemmatizer = None, token_limit=5
) -> FuzzyResults:
    """
    Match a sentence to a glossary.
    """
    if lemmatizer is not None:
        sentence = " ".join(lemmatizer.lemmatize(sentence))
    return fuzzy_match_glossary(sentence, glossary, token_limit)


def fuzzy_match_glossary(sentence: str, glossary: Dict[str, str], token_limit=5) -> FuzzyResults:
    """
    Return a list of glossary translations that are fuzzy matches of the sentence.

    Threshold is an integer between 0 and 100.
    """
    glossary_translations: FuzzyResults = []

    for token in sentence.split():
        token_matches: FuzzyResults = []
        for key, value in glossary.items():
            score = jaro.jaro_winkler_metric(token, key)
            token_matches.append((value, score, key, token))
        token_matches.sort(key=lambda x: x[1], reverse=True)
        glossary_translations.extend(token_matches[:token_limit])
    return glossary_translations


def tensor_to_list(tensor: Tensor) -> List:
    """
    Convert a tensor to a list.

    tensor.tolist() returns a list of numbers OR a single number if the tensor only contains one element.
    """
    t = tensor.tolist()
    if isinstance(t, list):
        return t
    return [t]
