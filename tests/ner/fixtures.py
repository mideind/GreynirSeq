import re
from typing import Tuple

import pytest


@pytest.fixture
def ner_sentence_pair():
    return (
        "Um <e:0:nvxo:>Guðrúnu Helgadóttur</e0> hefur <e:1:nkxn:>Einar</e1> ort .",
        "<e:1:nkxn:>Einar</e1> has written about <e:0:nvxo:>Guðrún Helgadóttir</e0> .",
    )


@pytest.fixture
def ner_final():
    return (
        (
            "<e:0:nkxe:>Einar Jónsson</e0> was visited by <e:1:nvxn:>Guðrún</e1> .",
            "<e:1:nvxn:>Guðrún</e1> fór í heimsókn til <e:0:nkxe:>Einars Jónssonar</e0> .",
        ),
        (
            "<e:0:nvxn:>Anna</e0> got a gift from <e:1:nkxþ:>Pétur</e1> , <e:2:nkxþ:>Páll</e2> and <e:3:nkxþ:>Alexei</e3> .",  # noqa
            "<e:0:nvxn:>Anna</e0> fékk gjöf frá <e:3:nkxþ:>Alexei</e3> , <e:1:nkxþ:>Pétri</e1> og <e:2:nkxþ:>Páli</e2> .",  # noqa
        ),
    )


@pytest.fixture
def ner_final_simple(ner_final: Tuple[Tuple[str, str], ...]):
    simplified = []
    pos_tag_pattern = ":n.*?:>"
    replacement_pattern = ":x:>"
    for en_sent, is_sent in ner_final:
        simplified.append(
            (
                re.sub(pos_tag_pattern, replacement_pattern, en_sent),
                re.sub(pos_tag_pattern, replacement_pattern, is_sent),
            )
        )

    return tuple(simplified)


@pytest.fixture
def ner_tagged_sentences_en():
    return (
        "Einar Jónsson was visited by Guðrún .	I-PER I-PER O O O I-PER O	hf",
        "Anna got a gift from Pétur , Páll and Alexei .	I-PER O O O O I-PER O I-PER O I-PER O	hf",
    )


@pytest.fixture
def ner_tagged_sentences_is():
    return (
        "Guðrún fór í heimsókn til Einars Jónssonar .	B-Person O O O O B-Person I-Person O",
        "Anna fékk gjöf frá Alexei , Pétri og Páli .	B-Person O O O B-Person O B-Person O B-Person O",
    )


@pytest.fixture
def is_pos_tags():
    return (
        ["nven-s", "sfg3eþ", "ao", "nveo", "ae", "nkee-s", "nkee-s", "p"],
        ["nven-s", "sfg3eþ", "nveo", "aþ", "nkeþ-s", "p", "nkeþ-s", "c", "nkeþ-s", "p"],
    )


@pytest.fixture
def is_ner():
    return "tests/ner/data/is.ner"


@pytest.fixture
def en_ner():
    return "tests/ner/data/en.ner"
