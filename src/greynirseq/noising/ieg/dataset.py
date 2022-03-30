from distutils.log import error
from torch.utils.data import Dataset
from collections import namedtuple

class ErrorDataset(Dataset):
    has_pos = False

    ErrorSentence = namedtuple("ErrorSentence", ["token", "pos"])

    def __init__(self, infile, posfile, error_handlers=[]) -> None:
        self.has_pos = posfile is not None
        
        with open(infile) as filehandler:
            self.sentences = filehandler.readlines()
        
        if self.has_pos:
            with open(posfile) as posfilehandler:
                self.postags = posfilehandler.readlines()
        
        self.error_handlers = error_handlers

    def __getitem__(self, index):
       
        for error_handler in self.error_handlers:
            if self.has_pos:
                pos = self.postags[index]
            else:
                pos = None

            errored_sentence = error_handler.apply(
                {
                    "text": self.sentences[index],
                    "pos": pos
                }
            )
            return errored_sentence