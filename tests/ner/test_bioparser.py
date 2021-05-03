from greynirseq.nicenlp.utils.ner_parser import BIOParser

# First value is the incorrect labels, the second the correct
TEST_LABELS = [
    ("B-X B-Y I-Y O", "B-X B-Y I-Y O"),
    ("O I-a I-a B-u O", "O B-a I-a B-u O"),
    ("O I-a O I-a B-u O", "O B-a O B-a B-u O"),
    ("B-p I-r O I-a B-p", "B-p I-p O B-a B-p"),
    (
        "O O B-person I-money O O I-time I-time O O B-location I-location",
        "O O B-person I-person O O B-time I-time O O B-location I-location",
    ),
]


def test_bioparser():
    for incorrect, correct in TEST_LABELS:
        assert BIOParser.parse(incorrect.split()) == correct.split()
