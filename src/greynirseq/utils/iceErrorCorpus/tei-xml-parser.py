#!/usr/bin/env python
"""
    Copyright (C) 2020 Miðeind ehf.

    This software is licensed under the MIT License:

        Permission is hereby granted, free of charge, to any person
        obtaining a copy of this software and associated documentation
        files (the "Software"), to deal in the Software without restriction,
        including without limitation the rights to use, copy, modify, merge,
        publish, distribute, sublicense, and/or sell copies of the Software,
        and to permit persons to whom the Software is furnished to do so,
        subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import (
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Iterable,
    cast,
    NamedTuple,
    Any,
    Set,
)

import os
from collections import defaultdict
from datetime import datetime
import glob
import random
import heapq
import argparse
import xml.etree.ElementTree as ET

import reynir_correct as gc
from reynir import _Sentence
from tokenizer import detokenize, Tok, TOK

from error_definitions import OUT_OF_SCOPE, ERROR_NUMBERS, to_simcategory, to_supercategory

# The type of a single error descriptor, extracted from a TEI XML file
ErrorDict = Dict[str, Union[str, int, bool]]


def element_text(element: ET.Element) -> str:
    """ Return the text of the given element,
        including all its subelements, if any """
    return "".join(element.itertext())


def correct_spaces(tokens: List[Tuple[str, str]]) -> str:
    """ Returns a string with a reasonably correct concatenation
        of the tokens, where each token is a (tag, text) tuple. """
    return detokenize(
        Tok(TOK.PUNCTUATION if tag == "c" else TOK.WORD, txt, None)
        for tag, txt in tokens
    )


def parse_file(path: str) -> List[Dict[str, Any]]:
    """ Process a single error corpus file in TEI XML format. """

    # Set up XML namespace stuff
    NS = "http://www.tei-c.org/ns/1.0"
    # Length of namespace prefix to cut from tag names, including { }
    nl = len(NS) + 2
    # Namespace dictionary to be passed to ET functions
    ns = dict(ns=NS)

    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        print(f"000: *** Unable to parse XML file {path} ***")
        raise e
    # Obtain the root of the XML tree

    res = []
    root = tree.getroot()
    # Iterate through the sentences in the file
    for sent in root.findall("ns:text/ns:body/ns:p/ns:s", ns):
        # Sentence identifier (index)
        index = sent.attrib.get("n", "")
        tokens: List[Tuple[str, str]] = []
        errors: List[ErrorDict] = []
        # A dictionary of errors by their index (idx field)
        error_indexes: Dict[str, ErrorDict] = {}
        dependencies: List[Tuple[str, ErrorDict]] = []
        # Error corpora annotations for sentences marked as unparsable
        # Enumerate through the tokens in the sentence
        for el in sent:
            tag = el.tag[nl:]
            if tag == "revision":
                # An error annotation starts here, eventually
                # spanning multiple tokens
                original = ""
                corrected = ""
                # Note the index of the starting token within the span
                start = len(tokens)
                # Revision id
                rev_id = el.attrib["id"]
                # Look at the original text
                el_orig = el.find("ns:original", ns)
                if el_orig is not None:
                    # We have 0 or more original tokens embedded within the revision tag
                    orig_tokens = [
                        (subel.tag[nl:], element_text(subel)) for subel in el_orig
                    ]
                    tokens.extend(orig_tokens)
                    original = " ".join(t[1] for t in orig_tokens).strip()
                # Calculate the index of the ending token within the span
                end = max(start, len(tokens) - 1)
                # Look at the corrected text
                el_corr = el.find("ns:corrected", ns)
                if el_corr is not None:
                    corr_tokens = [element_text(subel) for subel in el_corr]
                    corrected = " ".join(corr_tokens).strip()
                # Accumulate the annotations (errors)
                for el_err in el.findall("ns:errors/ns:error", ns):
                    attr = el_err.attrib
                    # Collect relevant information into a dict
                    xtype: str = attr["xtype"].lower()
                    error: ErrorDict = dict(
                        start=start,
                        end=end,
                        rev_id=rev_id,
                        xtype=xtype,
                        in_scope=xtype not in OUT_OF_SCOPE,
                        eid=attr.get("eid", ""),
                        original=original,
                        corrected=corrected,
                    )
                    errors.append(error)
                    # Temporarily index errors by the idx field
                    idx = attr.get("idx")
                    if idx:
                        error_indexes[idx] = error
                    # Accumulate dependencies that need to be "fixed up",
                    # i.e. errors that depend on and refer to other errors
                    # within the sentence
                    if xtype == "dep":
                        dep_id = attr.get("depId")
                        if dep_id:
                            # Note the fact that this error depends on the
                            # error with idx=dep_id
                            dependencies.append((dep_id, error))
                        else:
                            print(f"In file {path}:")
                            print(
                                f"\n{index}: *** 'depId' attribute missing for dependency ***"
                            )
            else:
                tokens.append((tag, element_text(el)))
        # Fix up the dependencies, if any
        for dep_id, error in dependencies:
            if dep_id not in error_indexes:
                print(f"In file {path}:")
                print(f"\n{index}: *** No error has idx='{dep_id}' ***")
            else:
                # Copy the in_scope attribute from the original error
                error["in_scope"] = error_indexes[dep_id]["in_scope"]

        # Reconstruct the original sentence
        # TODO switch for sentence from original text file
        text = correct_spaces(tokens)
        if not text:
            # Nothing to do: drop this and go to the next sentence
            continue

        res.append({"text": text, "errors": errors})

    return res


def parse_all(dataset_name: str, path: str) -> None:
    """ Parse all examples from the given path and save to `dataset_name`. """
    textfile = open(dataset_name + ".input0", "w")
    labelfile = open(dataset_name + ".label", "w")
    multilabelfile = open(dataset_name + ".multilabel", "w")
    simcategorylabelfile = open(dataset_name + ".simcategory.multilabel", "w")
    supercategorylabelfile = open(dataset_name + ".supercategory.multilabel", "w")

    for f in glob.iglob(path, recursive=True):
        res = parse_file(f)
        for r in res:
            label = 0
            error_labels = []
            # Check if there are any errors that are in scope
            if len(r["errors"]) > 0:
                for e in r["errors"]:
                    if e["in_scope"] and not e['xtype'] == 'dep':
                        error_labels.append(e['xtype'])
                        label = 1

            try:
                simcats = to_simcategory(error_labels)
                supercats = to_supercategory(error_labels)
            except:
                print(f"category not known: {error_labels}")
                continue


            textfile.write(r["text"])
            textfile.write("\n")

            labelfile.write(str(label))
            labelfile.write("\n")

            multilabelfile.write(" ".join(error_labels))
            multilabelfile.write("\n")

            simcategorylabelfile.write(" ".join(simcats))
            simcategorylabelfile.write("\n")

            supercategorylabelfile.write(" ".join(supercats))
            supercategorylabelfile.write("\n")

    textfile.close()
    labelfile.close()
    multilabelfile.close()
    simcategorylabelfile.close()
    supercategorylabelfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turn TEI XML sentence files into something a fairseq model can use.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset. Output files will be named according to this.",
    )

    parser.add_argument(
        "path", type=str, help="Glob path of XML files to process.",
    )

    args = parser.parse_args()

    parse_all(args.dataset_name, args.path)
