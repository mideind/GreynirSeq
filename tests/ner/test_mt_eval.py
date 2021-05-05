from greynirseq.ner.mt_eval import read_embedded_markers
from greynirseq.ner.ner_extracter import NERMarker


def test_read_embedded_markers(embedded_ner_tagged_sentences_is):
    actual_sents_markers, bad_markers = read_embedded_markers(
        embedded_ner_tagged_sentences_is, contains_model_marker=False
    )
    assert not bad_markers, "There are no bad markers."
    expected_sents_makers = [
        [NERMarker(0, 1, "P", "Guðrún"), NERMarker(5, 7, "P", "Einar Jónssonar")],
        [
            NERMarker(0, 1, "P", "Anna"),
            NERMarker(4, 5, "P", "Alexei"),
            NERMarker(5, 6, "P", "Pétri"),
            NERMarker(7, 8, "P", "Páli"),
        ],
    ]
    all(
        [
            expected == actual
            for sent_expected, sent_actual in zip(expected_sents_makers, actual_sents_markers)
            for expected, actual in zip(sent_expected, sent_actual)
        ]
    )
