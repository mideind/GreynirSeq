import argparse
import random
import re

import tqdm
from reynir import NounPhrase

NER_PATTERN = "<\s*e:([0-9]):([^:]*):>([^>]*?)<\s*/\s*e[0-9]+>"  # noqa


def parse_sentence(sentence):
    # Parses sentence of form
    # A dog is named <e:0:asd:>Doug Cat</e0> .
    # into a list of dictionaries
    # [..., {"ner": 0, "text": "Doug Cat", "pos": ""}, ...]

    parsed = []
    matches = re.finditer(NER_PATTERN, sentence)

    last_span = 0
    for match in matches:
        start, end = match.span()
        if last_span < start:
            parsed.append({"text": sentence[last_span:start], "ner": None, "pos": None})
        idx, pos, text = match.groups()
        parsed.append({"text": text, "ner": idx, "pos": pos})

        last_span = end

    if last_span != len(sentence):
        # Does not end with NER
        parsed.append({"text": sentence[last_span:], "ner": None, "pos": None})
    return parsed


def parse_sentence_pair(sentence_1, sentence_2):
    # Requires sentence_1 and sentence_2 to have
    # matching NER tags.
    # Accepts sentences of form as in parse_sentence
    # Returns "linked" sentence lists, with oidx added to
    # output form parse_sentence

    psent_1 = parse_sentence(sentence_1)
    psent_2 = parse_sentence(sentence_2)

    for idx, segment in enumerate(psent_1):
        nidx = segment["ner"]
        if nidx is None:
            continue
        for oidx, segment_2 in enumerate(psent_2):
            if segment_2["ner"] == nidx:
                segment["oidx"] = oidx
                segment_2["oidx"] = idx

    return psent_1, psent_2


def idf2kasus(idf_tag):
    # Only works for nouns
    if idf_tag[0] != "n":
        return None

    gender = idf_tag[1]
    kasus = idf_tag[3]
    return gender, kasus


def decline_np(phrase, idf_tag):
    kasus_map = {"n": "nominative", "o": "accusative", "þ": "dative", "e": "genitive"}
    np = NounPhrase(phrase)
    return getattr(np, kasus_map[idf_tag])


def patch_sentence(sentence, names, force=None):
    patched = ""
    for segment in sentence:
        neridx = segment["ner"]
        if neridx:
            try:
                _, kasus = idf2kasus(segment["pos"])
            except TypeError:
                return None
            if force:
                kasus = force
            try:
                decl = decline_np(names[int(neridx)], kasus)
            except:  # noqa
                # TODO FIX!!
                return None
            if decl is None:
                return None
            patched += decl
        else:
            patched += segment["text"]
    return patched


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--names")
    args = parser.parse_args()

    # We fill sentences with random names until all names have been
    # used in all declentions.

    fem_names = {"n": set(), "o": set(), "þ": set(), "e": set()}
    masc_names = {"n": set(), "o": set(), "þ": set(), "e": set()}

    with open(args.names) as names_file:
        for line in names_file.readlines():
            gen, name = line.strip().split("\t")
            if gen == "kvk":
                for k in fem_names:
                    fem_names[k].add(name)
                    first_name = name.split()[0]
                    fem_names[k].add(first_name)
            if gen == "kk":
                for k in masc_names:
                    masc_names[k].add(name)
                    first_name = name.split()[0]
                    masc_names[k].add(first_name)

    ofile = open(args.output, "w")
    with open(args.input) as sentence_file:
        for line in tqdm.tqdm(sentence_file.readlines()):
            en_sent, is_sent = line.strip().split("\t")
            en_sent, is_sent = parse_sentence_pair(en_sent, is_sent)

            names = []

            for segment in en_sent:
                if segment["ner"] is not None:
                    try:
                        gen, kasus = idf2kasus(segment["pos"])
                    except TypeError:
                        continue

                    if gen == "v":
                        name = random.sample(fem_names[kasus], 1)[0]
                        fem_names[kasus].remove(name)
                    else:
                        name = random.sample(masc_names[kasus], 1)[0]
                        masc_names[kasus].remove(name)
                    names.append(name)

            patch_sent_is = patch_sentence(is_sent, names)
            if patch_sent_is is None:
                continue
            patch_sent_en = patch_sentence(en_sent, names, force="n")
            ofile.writelines("{}\t{}\n".format(patch_sent_en, patch_sent_is))
    ofile.close()


if __name__ == "__main__":
    main()
