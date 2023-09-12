# Text noising for Icelandic

This submodule implements various ways to add spelling issues, grammatical noise and so on for Icelandic text.

## Usage

Introduce spelling and grammar errors into Icelandic text using rules and random noise.
Provide a file of tokenized text, one sentence per line

```python
python generate_errors.py < input_file.txt
```

You can adjust the error rate using the ``--word-spelling-error-rate`` and ``--rule-chance-error-rate`` arguments (on a scale of 0 to 1).
