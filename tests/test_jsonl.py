from greynirseq.utils.jsonl_utils import LineSemantics, MonolingualJSONDocument, read_documents


def assert_document_counts(documents, expected_doc_count, expected_para_count, expected_sent_count, line_semantics):
    assert (
        len(documents) == expected_doc_count
    ), f"Expected {expected_doc_count} documents, got {len(documents)} semantics={line_semantics}"
    for document in documents:
        assert (
            len(document.document) == expected_para_count
        ), f"Expected {expected_para_count} paragraphs, got {len(document.document)} semantics={line_semantics}"
        for paragraph in document.document:
            assert (
                len(paragraph) == expected_sent_count
            ), f"Expected {expected_sent_count} sentences, got {len(paragraph)} semantics={line_semantics}"


def test_read_documents_single_line():
    test_document = ["test sentence.\n"]
    lang = "en"
    for line_semantics in LineSemantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert len(documents) == 1, f"Expected 1 document, got {len(documents)}"
        assert len(documents[0].document) == 1, f"Expected 1 paragraph, got {len(documents[0].document)}"
        assert len(documents[0].document[0]) == 1, f"Expected 1 sentence, got {len(documents[0].document[0])}"


def test_read_documents_empty_line():
    test_document = ["\n"]
    lang = "en"
    for line_semantics in LineSemantics:
        assert (
            list(read_documents(test_document, line_sematics=line_semantics, lang=lang)) == []
        ), "Expected no documents"


def test_read_documents_empty_lines():
    test_document = ["\n", "\n", "\n"]
    lang = "en"
    for line_semantics in LineSemantics:
        assert (
            list(read_documents(test_document, line_sematics=line_semantics, lang=lang)) == []
        ), "Expected no documents"


def test_read_documents_two_lines():
    test_document = ["test sentence.\n", "Test 3.\n"]
    lang = "en"
    # Two sentences
    one_document_one_paragraph_semantics = [
        LineSemantics.sent_ignore,
        LineSemantics.sent_para,
        LineSemantics.sent_doc,
        LineSemantics.sent_para_doc,
    ]
    one_document_two_paragraph_semantics = [
        LineSemantics.para_doc,
        LineSemantics.para_ignore,
        LineSemantics.para_ignore_doc,
    ]
    two_documents_one_paragraph_semantics = [LineSemantics.doc_ignore]
    for line_semantics in one_document_one_paragraph_semantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 1, 2, line_semantics)
    for line_semantics in one_document_two_paragraph_semantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 2, 1, line_semantics)
    # Two documents
    for line_semantics in two_documents_one_paragraph_semantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 2, 1, 1, line_semantics)
    assert len(LineSemantics) == len(
        one_document_two_paragraph_semantics
        + one_document_one_paragraph_semantics
        + two_documents_one_paragraph_semantics
    )


def test_read_documents_two_lines_with_empty_line():
    test_document = ["test sentence.\n", "\n", "Test 3.\n"]
    lang = "en"
    one_document_two_sentences_semantics = [
        LineSemantics.sent_ignore,
    ]
    one_document_two_paragraph_semantics = [
        LineSemantics.sent_para,
        LineSemantics.sent_para_doc,
        LineSemantics.para_ignore_doc,
        LineSemantics.para_ignore,
    ]
    two_documents_one_paragraph_semantics = [
        LineSemantics.sent_doc,
        LineSemantics.para_doc,
        LineSemantics.doc_ignore,
    ]
    for line_semantics in one_document_two_sentences_semantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 1, 2, line_semantics)
    for line_semantics in one_document_two_paragraph_semantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 2, 1, line_semantics)
    # Two documents
    for line_semantics in two_documents_one_paragraph_semantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 2, 1, 1, line_semantics)
    assert len(LineSemantics) == len(
        one_document_two_paragraph_semantics
        + one_document_two_sentences_semantics
        + two_documents_one_paragraph_semantics
    )


def test_read_documents_two_lines_with_empty_lines():
    test_document = ["test sentence.\n", "\n", "\n", "Test 3.\n"]
    lang = "en"
    one_document_two_sentences_semantics = [
        LineSemantics.sent_ignore,
    ]
    one_document_two_paragraph_semantics = [
        LineSemantics.sent_para,
        LineSemantics.para_ignore,
    ]
    two_documents_one_paragraph_semantics = [
        LineSemantics.sent_doc,
        LineSemantics.sent_para_doc,
        LineSemantics.para_doc,
        LineSemantics.para_ignore_doc,
        LineSemantics.doc_ignore,
    ]
    for line_semantics in one_document_two_sentences_semantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 1, 2, line_semantics)
    for line_semantics in one_document_two_paragraph_semantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 2, 1, line_semantics)
    # Two documents
    for line_semantics in two_documents_one_paragraph_semantics:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 2, 1, 1, line_semantics)
    assert len(LineSemantics) == len(
        one_document_two_paragraph_semantics
        + one_document_two_sentences_semantics
        + two_documents_one_paragraph_semantics
    )


def test_read_documents_line_segmentation():
    test_document = ["test sentence. Second sentence.\n"]
    lang = "en"
    one_sentence = [
        LineSemantics.sent_ignore,
        LineSemantics.sent_para,
        LineSemantics.sent_doc,
        LineSemantics.sent_para_doc,
    ]
    two_sentences = [
        LineSemantics.para_ignore,
        LineSemantics.para_ignore_doc,
        LineSemantics.para_doc,
        LineSemantics.doc_ignore,
    ]
    for line_semantics in one_sentence:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 1, 1, line_semantics)
    for line_semantics in two_sentences:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 1, 2, line_semantics)
    assert len(LineSemantics) == len(two_sentences + one_sentence)


def test_read_documents_bug_fix_1():
    test_document = ["\n", "\n", "test sentence.\n", "\n", "Second sentence.\n", "\n", "Third sentence.\n"]
    lang = "en"
    one_doc_one_para_three_sent = [
        LineSemantics.sent_ignore,
    ]
    one_doc_three_para_one_sent = [
        LineSemantics.sent_para_doc,
        LineSemantics.sent_para,
        LineSemantics.para_ignore,
        LineSemantics.para_ignore_doc,
    ]
    three_doc_one_para_one_sent = [
        LineSemantics.sent_doc,
        LineSemantics.para_doc,
        LineSemantics.doc_ignore,
    ]
    for line_semantics in one_doc_one_para_three_sent:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 1, 3, line_semantics)
    for line_semantics in one_doc_three_para_one_sent:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 1, 3, 1, line_semantics)
    for line_semantics in three_doc_one_para_one_sent:
        documents = list(read_documents(test_document, line_sematics=line_semantics, lang=lang))
        assert_document_counts(documents, 3, 1, 1, line_semantics)
    assert len(LineSemantics) == len(
        three_doc_one_para_one_sent + one_doc_one_para_three_sent + one_doc_three_para_one_sent
    )


def test_json_to_text():
    test = [["test"], ["test2", "test3"]]
    doc = MonolingualJSONDocument.create_document(test, lang="en")
    # Default is to add a newline after each sentence, each paragraph and each document
    assert doc.to_text() == "test\n\ntest2\ntest3\n\n\n"


def test_text_to_json_to_text():
    test = "test\n\ntest2\ntest3\n\n\n"
    docs = list(read_documents([test], line_sematics=LineSemantics.sent_para_doc, lang="en"))
    assert len(docs) == 1
    assert docs[0].to_text() == test
