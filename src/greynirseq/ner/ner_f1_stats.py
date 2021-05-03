class EvalNER:
    EMPTY_SPAN = ["", 0, 0]

    def __init__(self, symbols, ignore=["O", "<sep>"]):
        self.labels = [lbl[2:] for lbl in symbols if lbl not in ignore]
        self.true_positive = {lbl: 0 for lbl in self.labels}
        self.false_positive = {lbl: 0 for lbl in self.labels}
        self.true_negative = {lbl: 0 for lbl in self.labels}
        self.false_negative = {lbl: 0 for lbl in self.labels}
        self._ignore = ignore

    def precision(self, labels=[]):
        if not labels:
            labels = set(self.labels)
        sum_true_positive = sum(self.true_positive[lbl] for lbl in labels)
        sum_false_positive = sum(self.false_positive[lbl] for lbl in labels)
        return sum_true_positive / max(1, sum_true_positive + sum_false_positive)

    def recall(self, labels=[]):
        if not labels:
            labels = set(self.labels)
        sum_true_positive = sum(self.true_positive[lbl] for lbl in labels)
        false_negative = sum(self.false_negative[lbl] for lbl in labels)
        return sum_true_positive / max(1, sum_true_positive + false_negative)

    def f1(self, labels=[]):
        precision = self.precision(labels)
        recall = self.recall(labels)
        return 2 * precision * recall / max(1, precision + recall)

    def idxs_to_spans(self, idxs):
        # We assume no nesting, i.e. there is no overlap of spans.

        spans = []
        cur_span = self.EMPTY_SPAN
        for i in range(len(idxs)):
            idx = idxs[i]

            if idx not in range(len(self.labels)) and cur_span[0]:
                # No NE but we need to add one that just ended
                spans.append(cur_span)
                cur_span = self.EMPTY_SPAN
            elif idx not in range(len(self.labels)):
                continue
            elif self.labels[idx][0] != "B" and self.labels[idx][2:] == cur_span[0]:
                # Continued span of same label
                cur_span[2] = i
            else:
                # New NE ("Correctly" marked B or first marked I)
                if cur_span[0]:
                    spans.append(cur_span)
                cur_span = [self.labels[idx], i, i]

        if cur_span[0] != "":
            # Sequence ended on NE
            spans.append(cur_span)

        return spans

    def print_all_stats(self):
        self.print_precision_recall_f1()
        self.print_bool_evals()

    def print_precision_recall_f1(self):
        tbl_string = "{:>13}   {:>10}    {:>10}    {:>10}"
        tbl_num_string = "{:>13}   {:>10.4f}    {:>10.4f}    {:>10.4f}"
        print(tbl_string.format("Entity", "Precision", "Recall", "F1"))
        print("")
        print(
            tbl_num_string.format(
                "All",
                self.precision(set(self.labels)),
                self.recall(set(self.labels)),
                self.f1(set(self.labels)),
            )
        )
        for lbl in set(self.labels):
            print(tbl_num_string.format(lbl, self.precision([lbl]), self.recall([lbl]), self.f1([lbl])))

    def print_bool_evals(self):
        tbl_string = "{:>13}   {:>10}    {:>10}    {:>10}    {:>10}"
        tbl_num_string = "{:>13}   {:>10.4f}    {:>10.4f}    {:>10.4f}    {:>10.4f}"
        print(tbl_string.format("Entity", "T-pos", "F-pos", "T-neg", "F-neg"))
        labels = set(self.labels)
        sum_true_positive = sum(self.true_positive[lbl] for lbl in labels)
        sum_false_positive = sum(self.false_positive[lbl] for lbl in labels)
        sum_true_negative = sum(self.true_negative[lbl] for lbl in labels)
        sum_false_negative = sum(self.false_negative[lbl] for lbl in labels)
        print(
            tbl_num_string.format(
                "All",
                sum_true_positive,
                sum_false_positive,
                sum_true_negative,
                sum_false_negative,
            )
        )
        for lbl in set(self.labels):
            print(
                tbl_num_string.format(
                    lbl,
                    self.true_positive[lbl],
                    self.false_positive[lbl],
                    self.true_negative[lbl],
                    self.false_negative[lbl],
                )
            )

    def compare(self, pred_idxs, target_idxs):

        n_pred_idxs = len(pred_idxs)

        pred_spans = self.idxs_to_spans(pred_idxs)
        target_spans = self.idxs_to_spans(target_idxs)

        for lbl in set(self.labels):
            pred_lbl_spans = set([tuple(span) for span in pred_spans if span[0] == lbl])
            target_lbl_spans = set([tuple(span) for span in target_spans if span[0] == lbl])

            true_positive = len(pred_lbl_spans.intersection(target_lbl_spans))
            false_positive = len(pred_lbl_spans.difference(target_lbl_spans))
            false_negative = len(target_lbl_spans.difference(pred_lbl_spans))

            true_span_length = sum([s[2] - s[1] for s in target_lbl_spans])
            true_negative = n_pred_idxs - true_span_length - false_negative

            self.true_positive[lbl] += true_positive
            self.true_negative[lbl] += true_negative
            self.false_positive[lbl] += false_positive
            self.false_negative[lbl] += false_negative
