import torch
from fairseq.models.bart import BARTHubInterface, BARTModel

"""
Override the default BARTModel to fix a bug.

The default implementation will add unknown unicode characters to the dictionary.
This causes crashes because the embedding matrix is no longer the same size as the dictionary.

The implementations here are copied from fairseq 0.10.2 with minor modifications to fix bugs.

petur: As far as I can tell:
The root issue is that SentencepieceBPE is not actually a _byte_ pair encoding, but rather a
_unicode_character_ pair encoding. This results in UNKs, which get added to the dict by default.
Example: "pöntunardagur! 🌠" (the shooting star emoji is not in mBART's vocabulary)
"""


class GreynirBARTModel(BARTModel):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return GreynirBARTHubInterface(x["args"], x["task"], x["models"][0])


class GreynirBARTHubInterface(BARTHubInterface):
    def encode(self, sentence: str, *addl_sentences, no_separator=True) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(" ")) > self.max_positions - 2:
            tokens = " ".join(tokens.split(" ")[: self.max_positions - 2])
        bpe_sentence = "<s> " + tokens + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
        return tokens.long()
