import pytest


@pytest.fixture
def ner_sentence_pair():
    return (
        "Um <e:0:nvxo:>Guðrúnu Helgadóttur</e0> hefur <e:1:nkxn:>Einar</e1> ort .",
        "<e:1:nkxn:>Einar</e1> has written about <e:0:nvxo:>Guðrún Helgadóttir</e0> .",
    )


@pytest.fixture
def ner_sentence_pair_dummy_pos_tags():
    return (
        (
            "<e:0:nkxe>Einar Jónsson</e0> was visited by <e:1:nvxn>Guðrún</e1> .",
            "<e:1:nvxn>Guðrún</e1> fór í heimsókn til <e:0:nkxe>Einars Jónssonar</e0> .",
        ),
        (
            "<e:0:nvxn>Anna</e0> got a gift from Pétur , Páll and Alexei .",
            "<e:0:nvxn>Anna</e0> fékk gjöf frá <e:1:nvxf>Alexei , <e:0:nvxþ>Pétri og <e:0:nvxþ>Páli .",
        ),
    )


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
def is_ner():
    return "tests/ner/data/is.ner"


@pytest.fixture
def en_ner():
    return "tests/ner/data/en.ner"