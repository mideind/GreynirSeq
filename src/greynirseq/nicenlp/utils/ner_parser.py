from typing import List


class BIOParser:
    def __init__(self, labels: List[str]):
        self.labels = labels
        self.idx = 0
        self.last = None

    def _over(self) -> str:
        """Ensures legal BIO tags, i.e. I-tags have same
        label as preceding B-tag and B-tags start new
        spans.
        """
        self.idx += 1
        cur_label = self.labels[self.idx - 1]

        if self.last is None:
            return cur_label

        if cur_label == "O":
            return cur_label

        if self.last == "O":
            # In case the label starts with I
            return "B" + cur_label[1:]

        _, last_cat = self.last.split("-")
        cur_head, _ = cur_label.split("-")
        if cur_head != "B":
            return f"{cur_head}-{last_cat}"
        return cur_label

    def over(self) -> str:
        label = self._over()
        self.last = label
        return label

    @classmethod
    def parse(cls, labels: List[str]) -> List[str]:
        parser = cls(labels)
        fixed_labels = []
        while parser.idx != len(parser.labels):
            fixed_labels.append(parser.over())
        return fixed_labels
