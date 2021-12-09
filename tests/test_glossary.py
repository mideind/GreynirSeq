from pathlib import Path

import pytest
import torch
from fairseq.utils import make_positions

from greynirseq.nicenlp.tasks.translation_with_glossary import (
    ENLemmatizer,
    ISLemmatizer,
    make_positions_with_constraints,
    match_glossary,
    read_glossary,
    whole_word_lengths,
    whole_word_sampling,
)


def test_original_make_position():
    pad_idx = 0
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 0],
            [5, 6, 99, 7, 8, 0, 0],
        ]
    )
    expected_positions = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0],
            [1, 2, 3, 4, 5, 0, 0],
        ]
    )
    calculated_positions = make_positions(test, padding_idx=pad_idx)
    assert torch.all(expected_positions.eq(calculated_positions))


def test_original_make_position_with_padding():
    pad_idx = 100
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 105, 106, 100],
            [101, 102, 103, 104, 105, 100, 100],
        ]
    )
    calculated_positions = make_positions(test, padding_idx=pad_idx)
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_no_constraint_params():
    pad_idx = 0
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 0],
            [5, 6, 99, 7, 8, 0, 0],
        ]
    )
    expected_positions = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0],
            [1, 2, 3, 4, 5, 0, 0],
        ]
    )
    calculated_positions = make_positions_with_constraints(test, padding_idx=pad_idx)
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_no_constraint_params_with_padding():
    pad_idx = 100
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 105, 106, 100],
            [101, 102, 103, 104, 105, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(test, padding_idx=pad_idx)
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_apply_offset():
    pad_idx = 0
    positional_marker_symbol_idx = 99
    positional_idx_restart_offset = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 0],
            [5, 6, 99, 7, 8, 0, 0],
        ]
    )
    expected_positions = torch.tensor(
        [
            [1, 2, 3, 4, 11, 12, 0],
            [1, 2, 11, 12, 13, 0, 0],
        ]
    )
    calculated_positions = make_positions_with_constraints(
        test,
        padding_idx=pad_idx,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_apply_shift_and_padding():
    pad_idx = 100
    positional_marker_symbol_idx = 99
    positional_idx_restart_offset = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 111, 112, 100],
            [101, 102, 111, 112, 113, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(
        test,
        padding_idx=pad_idx,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_apply_shift_and_padding_additional_sent_with_no_constraints():
    pad_idx = 100
    positional_marker_symbol_idx = 99
    positional_idx_restart_offset = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
            [5, 6, 9, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 111, 112, 100],
            [101, 102, 111, 112, 113, 100, 100],
            [101, 102, 103, 104, 105, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(
        test,
        padding_idx=pad_idx,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_apply_shift_and_padding_all_sent_with_no_constraints():
    pad_idx = 100
    positional_marker_symbol_idx = 99
    positional_idx_restart_offset = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 5, 4, 100],
            [5, 6, 6, 7, 8, 100, 100],
            [5, 6, 9, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 105, 106, 100],
            [101, 102, 103, 104, 105, 100, 100],
            [101, 102, 103, 104, 105, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(
        test,
        padding_idx=pad_idx,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )
    assert torch.all(expected_positions.eq(calculated_positions))


def test_masks_lengths():
    # The test sequence does not start with a whole word, so it is not included in the lengths.
    test = [0, 1, 0, 0, 1, 1, 1, 0, 0]
    expected_lengths = [3, 1, 1, 3]
    calculated_lengths = whole_word_lengths(test)
    assert expected_lengths == calculated_lengths


def test_whole_word_target_sampling():
    test = torch.arange(0, 10).long()
    whole_word_masker = {idx: 1 for idx in range(10)}
    # Set some words as partial tokens
    whole_word_masker[0] = 0
    whole_word_masker[5] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=2, contains_eos=False
    )
    assert len(result) == 2, "There should be two constraints"


def test_whole_word_target_sampling_with_eos():
    test = torch.arange(0, 10).long()  # eos is the last element and is considered a whole word
    whole_word_masker = {idx: 1 for idx in range(10)}
    # Set some words as partial tokens
    whole_word_masker[5] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=2, contains_eos=True
    )
    assert len(result) == 2, "There should be two constraints"


def test_whole_word_target_sampling_all_partial():
    test = torch.arange(0, 10).long()
    whole_word_masker = {idx: 1 for idx in range(10)}
    # make sure that every other word is a partial token
    whole_word_masker[1] = 0
    whole_word_masker[3] = 0
    whole_word_masker[5] = 0
    whole_word_masker[7] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=2, contains_eos=False
    )
    assert len(result) == 2, "There should be two constraints"
    assert all(list(len(constraint) == 2 for constraint in result)), "All constraints should have two elements"


def test_whole_word_target_sampling_count_larger_than_length():
    test = torch.arange(0, 10).long()
    whole_word_masker = {idx: 1 for idx in range(10)}
    # Set some words as partial tokens
    whole_word_masker[0] = 0
    whole_word_masker[5] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=10, contains_eos=False
    )
    assert len(result) == 7, "There should be seven constraints"


def test_whole_word_target_sampling_count_negative():
    test = torch.arange(0, 10).long()
    whole_word_masker = {idx: 1 for idx in range(10)}
    # Set some words as partial tokens
    whole_word_masker[0] = 0
    whole_word_masker[5] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=-10, contains_eos=False
    )
    assert len(result) == 0, "There should be no constraints"


def test_is_lemmatization():
    test = "[1] Skammstöfunin „OB“ vísar til heitra, bjartra og skammlífra stjarna af litrófsgerð O og B sem enn skína skært í gisnum stjörnuþyrpingum sem ferðast um Vetrarbrautina."
    lemmatizer = ISLemmatizer()
    assert (
        " ".join(lemmatizer.lemmatize(test))
        == "[ 1 ] skammstöfun „ OB “ vís til heitur , bjartur og skammlífur stjarna af litrófsgera O og B sem enn skína skær í gisinn stjörnuþyrping sem ferðast um Vetrarbrautin ."
    )


def test_en_lemmatization():
    test = "If this is indeed the case, the currently-held picture of how galaxies formed in the early Universe may also require a complete overhaul."
    lemmatizer = ENLemmatizer()
    assert (
        " ".join(lemmatizer.lemmatize(test))
        == "if this be indeed the case , the currently - hold picture of how galaxy form in the early Universe may also require a complete overhaul ."
    )


def test_match_glossary_exact_match():
    test_glossary = {"hello": "hæ", "world": "heimur"}
    test_sentence = "hello world"
    expected_result = ["hæ", "heimur"]
    results = match_glossary(sentence=test_sentence, glossary=test_glossary)
    threshold = 1.0
    print(results)
    assert [x[0] for x in filter(lambda x: x[1] >= threshold, results)] == expected_result


def test_match_glossary_not_exact():
    test_glossary = {"hello": "hæ", "world": "heimur"}
    test_sentence = "hell worl"
    expected_result = []
    results = match_glossary(sentence=test_sentence, glossary=test_glossary)
    threshold = 1.0
    print(results)
    assert [x[0] for x in filter(lambda x: x[1] >= threshold, results)] == expected_result


def test_match_glossary_threshold():
    test_glossary = {"hello": "hæ", "world": "heimur"}
    test_sentence = "hell worl"
    expected_result = ["hæ", "heimur"]
    results = match_glossary(sentence=test_sentence, glossary=test_glossary)
    threshold = 0.8
    print(results)
    assert [x[0] for x in filter(lambda x: x[1] >= threshold, results)] == expected_result


@pytest.fixture(scope="session")
def glossary_data():
    p = Path("tests/test_glossary.tsv")
    return read_glossary(p)


# flake8: noqa
# pytest.mark.skip(reason="Test for finding good hyperparameters. Not to be run automatically.")
@pytest.mark.parametrize(
    "src,tgt,expected",
    [
        (
            "Upp úr 1830 áttaði Thomas Brisbane sig á að π1 Gruis var mun nálægara tvístirni en hin stjarnan.",
            "Thomas Brisbane realised in the 1830s that π1 Gruis was itself also a much closer binary star system.",
            ["binary"],
        ),
        (
            "Stóra Magellansskýið er risavaxið en mjög lítið í samanburði við Vetrarbrautina okkar, eða aðeins 14.000 ljósár í þvermál - næstum tífallt minni en Vetrarbrautin okkar.",
            "The Large Magellanic Cloud is enormous, but when compared to our own galaxy it is very modest in extent, spanning just 14 000 light-years - about ten times smaller than the Milky Way.",
            [
                "light-year",
            ],  # 'Vetrarbrautina okkar' er 'Milky-way', en það er ekki í glossary, bara 'vetrarbraut' sem 'galaxy'
        ),
        (
            "Stjörnufræðingarnir hafa nú notað gögnin til að setja saman stærstu skrá sem til er yfir stjörnur í miðju vetrarbrautarinnar [2].",
            "The team has now used these data to compile the largest catalogue of the central concentration of stars in the Milky Way ever created [2].",
            ["galaxy"],  # enskan er ekki rétt miðað við að vetrarbraut er ekki með hástaf
        ),
        (
            "Öflug geislun frá glóandi heitum afkvæmum þess hefur haft feikilega mikil áhrif á skýið.",
            "It has been dramatically affected by the powerful radiation coming from its smoldering offspring.",
            ["radiation"],
        ),
        (
            "Ekki er hægt að útiloka slíkan staðbundinn lofthjúp, sem er fræðilega mögulegur, með þessari rannsókn.",
            "Such a local atmosphere, which is possible in theory, is not excluded by the observations.",
            ["atmosphere"],
        ),
        (
            "Mikill massi þyrpingarinnar brýtur ljós fjarlægu vetrarbrautarinnar og verkar því eins og þyngdarlinsa [2].",
            "The vast mass of this cluster bends the light of the more distant galaxy, acting as a gravitational lens [2].",
            ["cluster", "galaxy"],
        ),
    ],
)
def test_glossary_matching(src, tgt, expected, glossary_data):
    """Testing the fuzzy matching used to pair glossary entries.

    Not all the test cases will (probably ever) pass but are left here to try different hyperparameters/heuristics.
    The threshold chosen should preferably allow more matches than too few, since the training is done with negative examples as well.
    This test case should not be run automatically."""
    lemmatizer = ISLemmatizer()
    results = match_glossary(sentence=src, glossary=glossary_data, lemmatizer=lemmatizer)
    results = sorted(results, key=lambda x: x[1], reverse=True)
    threshold = 0.96
    print(results[:10])
    assert [x[0] for x in filter(lambda x: x[1] >= threshold, results)] == expected


# ljósmæling	photometry
# ljósmælir	photometer
