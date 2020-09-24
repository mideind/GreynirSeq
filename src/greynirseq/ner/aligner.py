import argparse
import copy
import re
import spacy
import tqdm

from collections import defaultdict
from pyjarowinkler import distance
from scipy.optimize import linear_sum_assignment
from spacy.gold import offsets_from_biluo_tags
nlp = spacy.load("en_core_web_lg")

NULL_TAG = 'O'

PUNCTUATION_SYMBOLS = (
    "'ʼ∞¥≈€∂‧Ω÷‐℉†℃‛″£™∙§«»@¯^!½³²˜−{$¼¹≠}º‗®‑#¡´&`|·≥―′¿<≤~?±"
    '…\\>”_+][°–=*"‘%„“;:-•(),…–`-—!’?;“”:.,'
)
DIGIT_PROG = re.compile(r"\d+")
PUNCT_PROG = re.compile("[" + re.escape(PUNCTUATION_SYMBOLS) + "]")
SPACE_PROG = re.compile(r"\s+")

DATASETS = "emea os2018 bible medical ees tatoeba".split()
DATA_PATHS_IS = [
    '/data/datasets/{0}/clean/train/{0}.en-is.train.is'.format(dataset)
    for dataset in DATASETS
]
DATA_PATHS_EN = [
    '/data/datasets/{0}/clean/train/{0}.en-is.train.en'.format(dataset)
    for dataset in DATASETS
]
DATA_PATHS = DATA_PATHS_IS + DATA_PATHS_EN


def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = DIGIT_PROG.sub("0", sentence)
    sentence = PUNCT_PROG.sub("", sentence)
    sentence = SPACE_PROG.sub("", sentence)
    return sentence


def split_tag(tag):
    if tag == NULL_TAG:
        return tag, None
    else:
        return tag.split('-')


def get_min_hun_distance(words1, words2):
    values = []
    hits = []
    min_dist = 0
    for i in range(len(words1)):
        w1 = words1[i]
        row = []
        for j in range(len(words2)):
            w2 = words2[j]
            row.append(1 - distance.get_jaro_distance(w1, w2, winkler=True, scaling=0.1))
        values.append(row)

    row_ids, col_ids = linear_sum_assignment(values)
    row_ids = list(row_ids)
    col_ids = list(col_ids)
    hits = []
    valsum = 0
    for i in range(len(row_ids)):
        row_id = row_ids[i]
        col_id = col_ids[i]
        hits.append((words1[row_id], words2[col_id], row_id, col_id, values[row_id][col_id]))
        valsum += values[row_id][col_id]

    min_dist = valsum / (len(words1) + len(words2))

    return min_dist, hits


BASE_STATS = {
    "sum_per_1": 0,
    "sum_per_2": 0,
    "number_lines": 0,
    "number_lines_w_per": 0,
    "number_lines_w_per_match": 0,
    "number_lines_w_mper": 0,
    "number_lines_w_per_mism": 0,
    "sum_dist": 0,
}


class ParallelNER:
    lang_1 = None
    lang_2 = None

    print_data_file = None

    stats = {
        dataset: copy.copy(BASE_STATS) for dataset in DATASETS
    }

    ner_hist_1 = defaultdict(int)
    ner_hist_2 = defaultdict(int)
    ner_pair_hist = defaultdict(int)

    default_mode = 'is'
    provenance_sets = {
        path: set() for path in DATA_PATHS
    }

    def __init__(self, lang_1, lang_2):
        self.lang_1 = lang_1
        self.lang_2 = lang_2

    def load_provenance(self):
        for path in self.provenance_sets:
            with open(path) as fp:
                for line in fp.readlines():
                    self.provenance_sets[path].add(preprocess_sentence(line))
    
    def check_provenance(self, sentence):
        hits = []
        cleaned_sentence = preprocess_sentence(sentence)
        for path in self.provenance_sets:
            if cleaned_sentence in self.provenance_sets[path]:
                hits.append(path.split('/')[-1].split('.')[0])
        return hits

    def parse_is(self, ner, *args):
        # Always B at beginning, then I for following (Icelandic)

        # TODO: check if same approach as for spacy works

        result = []
        found_tag = None
        tag_len = 0
        for i in range(len(ner)):
            tag = ner[i]
            head, tail = split_tag(tag)
            if head == 'I' and tag_len != 0:
                # Some issue here, sometimes wrong follow up... But we let it slide.
                tag_len += 1
            if head == 'B' or (tag_len == 0 and head == 'I'):
                if found_tag is not None:
                    result.append((i - tag_len, i, found_tag))
                found_tag = tail
                tag_len = 1
            if head == 'O':
                if found_tag is not None:
                    result.append((i - tag_len, i, found_tag))
                found_tag = None
                tag_len = 0
        return result

    def parse_hf(self, ner, *args):
        # Only B if there is an I in front (Huggingface)
        result = []
        found_tag = None
        tag_len = 0
        for i in range(len(ner)):
            tag = ner[i]
            head, tail = split_tag(tag)
            if head == 'I' and found_tag is not None and tail == found_tag:
                tag_len += 1
            if head == 'I' and found_tag is None:
                found_tag = tail
                tag_len = 1
            if head == 'B' or (head == 'I' and tail != found_tag):
                if found_tag is not None:
                    # Seldom fails, very seldom but is needed
                    result.append((i - tag_len, i, found_tag))
                found_tag = tail
                tag_len = 1
            if head == 'O':
                if found_tag is not None:
                    result.append((i - tag_len, i, found_tag))
                found_tag = None
                tag_len = 0
        return result

    def parse_sp(self, ner, sentence):
        # U for unique, i.e. no B or I if single etc (Spacy)
        doc = nlp(sentence)
        return offsets_from_biluo_tags(doc, ner)

    def parse_line(self, line):
        sp = line.strip().split('\t')
        
        sentence, ner_tags = sp[:2]

        mode = self.default_mode  # Assume Icelandic
        if len(sp) > 2:
            mode = sp[2]
        
        ner_parsed = getattr(self, "parse_{}".format(mode))(ner_tags.split(), sentence)

        origins = self.check_provenance(sentence)

        return sentence, ner_parsed, mode, origins

    def print_line(self, p1, p2, pair_info, file_name):
        if self.print_data_file is None:
            self.print_data_file = open(file_name, 'w')
        
        # parser1, origin1, parser2, origin2, match_cat, alignment

        has_hit = pair_info["n_tags_1"] + pair_info["n_tags_2"]

        if has_hit and pair_info["n_tags_1"] == pair_info["n_tags_2"]:
            match = 1
        elif has_hit:
            match = 2
        else:
            match = 0

        alignments = []
        max_dist = ""
        for pair in pair_info["pair_map"]:
            src, tgt, score = pair
            src = [str(a) for a in src]
            tgt = [str(a) for a in tgt]
            alignments.append(
                "{}-{}".format(":".join(src), ":".join(tgt))
            )
            
        if alignments:
            max_dist = max([p[-1] for p in pair_info["pair_map"]])

        output = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            p1[2], ",".join(p1[3]), p2[2], ",".join(p2[3]),
            match,
            max_dist,
            " ".join(alignments)
        )
        self.print_data_file.writelines(output)

    def parse_files(self, print_data):
        with open(self.lang_1) as lang_1, open(self.lang_2) as lang_2:
            for l1, l2 in tqdm.tqdm(zip(lang_1, lang_2)):
                p1 = self.parse_line(l1)
                p2 = self.parse_line(l2)
                pair_info = self.parse_pair(p1, p2)
                self.update_stats(pair_info)
                self.print_line(p1, p2, pair_info, print_data)

    def parse_files_gen(self):
        with open(self.lang_1) as lang_1, open(self.lang_2) as lang_2:
            for l1, l2 in tqdm.tqdm(zip(lang_1, lang_2)):
                p1 = self.parse_line(l1)
                p2 = self.parse_line(l2)
                pair_info = self.parse_pair(p1, p2)
                self.update_stats(pair_info)
                yield p1, p2, pair_info

    def parse_pair(self, p1, p2):
        try:
            corp1 = p1[-1][0]
            corp2 = p2[-1][0]
        except:
            if "mixup" not in self.stats:
                self.stats["mixup"] = copy.copy(BASE_STATS)
            corp1 = corp2 = "mixup"

        tags = ["per", "person"]
        tags1 = [t for t in p1[1] if t[-1].lower() in tags]
        tags2 = [t for t in p2[1] if t[-1].lower() in tags]

        if p1[-2] == 'sp':
            pers1 = [p1[0][t[0]:t[1]].lower() for t in tags1]
        else:
            pers1 = [" ".join(p1[0].split()[t[0]:t[1]]).lower() for t in tags1]
        pers2 = [" ".join(p2[0].split()[t[0]:t[1]]).lower() for t in tags2]

        for per in pers1:
            self.ner_hist_1[per] += 1
        for per in pers2:
            self.ner_hist_2[per] += 1

        pair_map = []

        min_dist = 1
        if pers1 and pers2:
            min_dist, hits = get_min_hun_distance(pers1, pers2)
            if hits:
                for hit in hits:
                    text_1, text_2, loc_1, loc_2, distance = hit
                    self.ner_pair_hist["{}\t{}\t{}\t{}".format(
                        text_1,
                        text_2,
                        text_1 == text_2,
                        distance)] += 1
                    pair_map.append([tags1[loc_1], tags2[loc_2], distance])

        return {
            "per_tags_1": tags1,
            "per_tags_2": tags2,
            "n_tags_1": len(tags1),
            "n_tags_2": len(tags2),
            "distance": min_dist,
            "origin_1": corp1,
            "origin_2": corp2,
            "pair_map": pair_map
        }

    def update_stats(self, pair_info): 
        origin = pair_info["origin_1"]
        if pair_info["origin_1"] != pair_info["origin_2"]:
            origin = "mixup"
            if origin not in self.stats:
                self.stats[origin] = copy.copy(BASE_STATS)
        self.stats[origin]["sum_per_1"] += pair_info["n_tags_1"] 
        self.stats[origin]["sum_per_2"] += pair_info["n_tags_2"]
        self.stats[origin]["number_lines"] += 1
        if pair_info["n_tags_1"] and pair_info["n_tags_2"]:
            self.stats[origin]["number_lines_w_per"] += 1
            if pair_info["n_tags_1"] == pair_info["n_tags_2"]:
                self.stats[origin]["number_lines_w_per_match"] += 1
        if pair_info["n_tags_1"] > 1 and pair_info["n_tags_2"] > 1:
            self.stats[origin]["number_lines_w_mper"] += 1
        if pair_info["n_tags_1"] != pair_info["n_tags_2"]:
            self.stats[origin]["number_lines_w_per_mism"] += 1
        self.stats[origin]["sum_dist"] = pair_info["distance"]
    
    def print_stats(self):
        tbl_string = "{:>13}   {:>10}    {:>10}    {:>10}    {:>10}    {:>10}    {:>10}"
        tbl_num_string = "{:>13}   {:>10}    {:>10}    {:>10}    {:>10}    {:>10}    {:>10.6f}"
        print(tbl_string.format("Origin", "Lines", "LWPers",  "LWMultiPers", "Mism", "LWPersMatch", "Avg.Dist"))
        for origin in self.stats:
            st = self.stats[origin]
            print(tbl_num_string.format(
                origin,
                st["number_lines"], # Actual number of lines
                st["number_lines_w_per"], # Person on both sides
                st["number_lines_w_mper"], # Multiple persons on both sideds
                st["number_lines_w_per_mism"], # Actual not same count, but includes,
                st["number_lines_w_per_match"],
                st["sum_dist"] / max(st["number_lines_w_per"], 1) # Avg dist
            ))

    def _write_ner_hist(self, hist_file, data):
        with open(hist_file, 'w') as of:
            for k, v in {k: v for k, v in sorted(data.items(), key=lambda item: -item[1])}.items():
                of.writelines("{}\t{}\n".format(k, v))

    def write_ner_hist(self, hist_file1, hist_file2, hist_file_pair):
        self._write_ner_hist(hist_file1, self.ner_hist_1)
        self._write_ner_hist(hist_file2, self.ner_hist_2)
        self._write_ner_hist(hist_file_pair, self.ner_pair_hist)
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_ent')
    parser.add_argument('--en_ent')
    parser.add_argument('--output')
    parser.add_argument('--provenance', type=bool, default=False)
    parser.add_argument('--name_histograms', type=bool, default=False)

    args = parser.parse_args()

    eval_ner = ParallelNER(
        args.en_ent,
        args.is_ent
    )

    if args.provenance:
        eval_ner.load_provenance()
        eval_ner.print_stats()

    eval_ner.parse_files(print_data=args.output)

    if args.name_histograms:   
        eval_ner.write_ner_hist('en.hist.ner', 'is.hist.ner', 'en-is.hist.ner')


if __name__ == "__main__":
    main()
