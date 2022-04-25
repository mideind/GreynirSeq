import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Tuple

from greynirseq.utils.tokenize_splitter import SentenceSegmenter


@dataclass
class MonolingualJSONDocument:
    """A monolingual JSON document."""

    document: List[List[str]]
    lang: str
    uuid: str
    domains: Optional[List[str]]
    title: Optional[str]

    @staticmethod
    def create_document(
        document: List[List[str]], lang: str, domains: Optional[List[str]] = None, title: Optional[str] = None
    ):
        """Create a new monolingual JSON document.
        Copy all the sentences.
        Set the uuid automatically.
        Does not add empty sentences or paragraphs."""
        mono_doc = MonolingualJSONDocument([[]], lang, str(uuid.uuid4()), domains, title)
        for paragraph in document:
            mono_doc.add_paragraph(paragraph)
        return mono_doc

    def is_empty(self):
        """Check if the document is empty."""
        return len(self.document[0]) == 0

    def add_sentence(self, sentence: str) -> bool:
        """Add a sentence to the document to the last paragraph. Does not add empty sentences"""
        if len(sentence) > 0:
            self.document[-1].append(sentence)
            return True
        return False

    def add_paragraph(self, paragraph: List[str]) -> bool:
        """Add a paragraph to the end of the document. Does not add empty paragraphs"""
        if len(paragraph) > 0:
            # Figure out the paragraph index
            if self.is_empty():
                added = False
                for sentence in paragraph:
                    added = self.add_sentence(sentence) or added
                return added
            # Otherwise we want to add a new paragraph to the document
            else:
                self.document.append([])
                added = False
                for sentence in paragraph:
                    added = self.add_sentence(sentence) or added
                # If all the sentences we tried to add were empty, we want to delete the last paragraph
                if not added:
                    self.document.pop()
                else:
                    return True
        return False


@dataclass
class EnumeratedMonolingualJSONDocument(MonolingualJSONDocument):
    """A monolingual JSON document with a document index."""

    example_idx: int

    @staticmethod
    def from_monolingual_json_document(
        document: MonolingualJSONDocument, example_idx: int
    ) -> "EnumeratedMonolingualJSONDocument":
        """Create an enumerated monolingual JSON document from a monolingual JSON document."""
        # No deepcopy here to reduce memory usage.
        return EnumeratedMonolingualJSONDocument(
            document=document.document,
            lang=document.lang,
            uuid=document.uuid,
            domains=document.domains,
            title=document.title,
            example_idx=example_idx,
        )

    def flatten(self) -> List["FlattenedMonolingualJSONDocument"]:
        """Flatten the document into a list of sentences."""
        return [
            FlattenedMonolingualJSONDocument(
                sentence=sentence,
                lang=self.lang,
                uuid=self.uuid,
                domains=self.domains,
                title=self.title,
                example_idx=(self.example_idx, paragraph_idx, sentence_idx),
            )
            for paragraph_idx, paragraph in enumerate(self.document)
            for sentence_idx, sentence in enumerate(paragraph)
        ]


@dataclass
class FlattenedMonolingualJSONDocument:
    """A monolingual JSON document flattened to the sentence level."""

    sentence: str
    lang: str
    uuid: str
    domains: Optional[List[str]]
    title: Optional[str]
    example_idx: Tuple[int, int, int]

    @staticmethod
    def unflatten(
        flattened_documents: Iterable["FlattenedMonolingualJSONDocument"],
    ) -> Iterable[MonolingualJSONDocument]:
        """Map an iterable of flattened documents to unflattened documents."""
        prev_para_num = None
        prev_doc_num = None
        current_document = None
        for example in iter(flattened_documents):
            current_doc_num, current_para_num, _current_sent_num = example.example_idx
            # Only True in first iteration
            if current_document is None:
                current_document = MonolingualJSONDocument(
                    document=[[example.sentence]],
                    lang=example.lang,
                    uuid=example.uuid,
                    domains=example.domains,
                    title=example.title,
                )
            elif current_doc_num != prev_doc_num:
                yield current_document
                current_document = MonolingualJSONDocument(
                    document=[[example.sentence]],
                    lang=example.lang,
                    uuid=example.uuid,
                    domains=example.domains,
                    title=example.title,
                )
            elif current_para_num != prev_para_num:
                current_document.add_paragraph([example.sentence])
            else:
                current_document.add_sentence(example.sentence)
            prev_doc_num = current_doc_num
            prev_para_num = current_para_num
        if current_document is not None:
            yield current_document


class LineSemantics(str, Enum):
    """Line semantics define how to interpret lines.

    sent+ignore: Each line is a sentence, ignore empty lines.
    sent+para: Each line is a sentence, empty lines indicate a new paragraph.
    sent+doc: Each line is a sentence, empty lines indicate a new document.
    sent+para+doc: Each line is a sentence, empty lines indicate a new paragraph, the next, a new document.
    para+ignore: Each line is a paragraph (will be segmented), ignore empty lines.
    para+ignore+doc: Each line is a paragraph (will be segmented), ignore first empty line, the next, a new document.
    para+doc: Each line is a paragraph (will be segmented), empty lines indicate a new document.
    doc+ignore: Each line is a document (will be segmented), ignore empty lines.
    """

    sent_ignore = "sent+ignore"
    sent_para = "sent+para"
    sent_doc = "sent+doc"
    sent_para_doc = "sent+para+doc"
    para_ignore = "para+ignore"
    para_ignore_doc = "para+ignore+doc"
    para_doc = "para+doc"
    doc_ignore = "doc+ignore"

    def read_line(self, line: str, sentence_segmenter: SentenceSegmenter) -> List[str]:
        """Read a line and return the sentences based on the semantics."""
        if (
            self == LineSemantics.sent_ignore
            or self == LineSemantics.sent_para
            or self == LineSemantics.sent_doc
            or self == LineSemantics.sent_para_doc
        ):
            return [line]
        # For doc/para lines we segment
        else:
            return sentence_segmenter.segment_text(line)


def read_documents(
    input_file: Iterable[str],
    line_sematics: LineSemantics,
    lang: str,
    domains: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> Iterable[MonolingualJSONDocument]:
    """Translates a sequence of lines into multiple MonolinugalJSONDocuments."""
    sentence_segmenter = SentenceSegmenter(lang=lang)
    read_empty_line = False
    read_many_empty_lines = False
    # We make sure that our current document is never None
    line_iterator = iter(input_file)
    line = next(line_iterator).strip()
    buffer = line_sematics.read_line(line, sentence_segmenter)
    current_document = MonolingualJSONDocument.create_document([buffer], lang=lang, domains=domains, title=title)
    read_empty_line = line == ""
    for line in line_iterator:
        line = line.strip()
        # Empty line state
        if line == "":
            if read_empty_line:
                read_many_empty_lines = True
            read_empty_line = True
            continue
        buffer = line_sematics.read_line(line, sentence_segmenter)
        if line_sematics == LineSemantics.sent_ignore:
            for sent in buffer:
                current_document.add_sentence(sent)
        elif line_sematics == LineSemantics.para_ignore:
            # add_paragraph takes care of not adding empty lines
            current_document.add_paragraph(buffer)
        elif line_sematics == LineSemantics.doc_ignore:
            if not current_document.is_empty():
                yield current_document
            current_document = MonolingualJSONDocument.create_document(
                [buffer], lang=lang, domains=domains, title=title
            )
        elif line_sematics == LineSemantics.sent_para:
            if read_empty_line:
                current_document.add_paragraph(buffer)
            else:
                for sent in buffer:
                    current_document.add_sentence(sent)
        # Here we need to maintain two states, read_paragraph_boundry and read_document_boundry
        elif line_sematics == LineSemantics.sent_para_doc:
            if read_many_empty_lines:
                if not current_document.is_empty():
                    yield current_document
                current_document = MonolingualJSONDocument.create_document(
                    [buffer], lang=lang, domains=domains, title=title
                )
            elif read_empty_line:
                current_document.add_paragraph(buffer)
            else:
                for sent in buffer:
                    current_document.add_sentence(sent)
        elif line_sematics == LineSemantics.sent_doc:
            if read_empty_line:
                if not current_document.is_empty():
                    yield current_document
                current_document = MonolingualJSONDocument.create_document(
                    [buffer], lang=lang, domains=domains, title=title
                )
            else:
                for sent in buffer:
                    current_document.add_sentence(sent)
        elif line_sematics == LineSemantics.para_ignore_doc:
            if read_many_empty_lines:
                if not current_document.is_empty():
                    yield current_document
                current_document = MonolingualJSONDocument.create_document(
                    [buffer], lang=lang, domains=domains, title=title
                )
            else:
                current_document.add_paragraph(buffer)
        elif line_sematics == LineSemantics.para_doc:
            if read_empty_line:
                if not current_document.is_empty():
                    yield current_document
                current_document = MonolingualJSONDocument.create_document(
                    [buffer], lang=lang, domains=domains, title=title
                )
            else:
                current_document.add_paragraph(buffer)
        else:
            raise ValueError(f"Unknown line semantics: {line_sematics}")
        # We didn't read an empty line since we got here. We can reset the state
        read_empty_line = False
        read_many_empty_lines = False
    if not current_document.is_empty():
        yield current_document
