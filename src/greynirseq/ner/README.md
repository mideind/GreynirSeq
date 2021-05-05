# Named Entity Processing Pipeline for NMT

In training data for NMT (neural machine translation) systems it is of benefit to have a large and varried corpus. Unfortunately this is not often the case. This submodule implements a pipeline for tagging, filtering, matching, substituing and evaluating translation of named entities in a parallel English to Icelandic corpus.

## Name Tagging

For Icelandic NER the included IceBERT-NER model is used. For english we use huggingface (for better accuracy) and spacy (due to length restrictions). (`python -m spacy download en_core_web_lg`).

The script accepts a file in English or Icelandic, the sentences will then be tokenized and tokens joined with spaces as the model expects.

In later stages we assume that the English and Icelandic sentences are translations of each other, i.e. parallel.

```example.is
Guðrún fór í heimsókn til Einars Jónssonar.
Anna fékk gjöf frá Alexei, Pétri og Páli.
```
```example.en
Einar Jónsson was visited by Guðrún.
Anna got a gift from Pétur, Páll and Alexei.
```

```bash
python nertagger.py --language is --input testdata/example.is --output testdata/example.ner.is
python nertagger.py --language en --input testdata/example.en --output testdata/example.ner.en
```

Which writes to file the original sentence, labels and a tag for the NER model used.
```
Guðrún fór í heimsókn til Einars Jónssonar .	B-Person O O O O B-Person I-Person O	is
Anna fékk gjöf frá Alexei , Pétri og Páli .	B-Person O O O B-Person O B-Person O B-Person O	is
```

and for English the last column is hf for Huggingface and sp for spacy.

```
Einar Jónsson was visited by Guðrún .	I-PER I-PER O O O I-PER O	hf
Anna got a gift from Pétur , Páll and Alexei .	I-PER O O O O I-PER O I-PER O I-PER O	hf
```

Note the different tagsets used, this is dealt with by the aligner.

## Embedding and unifying tags

In order to be able to train and NMT system on the NER tagged data we embed the NEs markers into the sentences, detokenizes and unify the tag sets.

```
lang=en
python ner_extracter.py --input testdata/example.ner.$lang --output testdata/example.ner-ext.$lang --embed_tags
# Running the command also displays the number of labels parsed.
cat testdata/example.ner-ext.$lang
hf	<P>Einar Jónsson</P> was visited by <P>Guðrún</P> .
hf	<P>Anna</P> got a gift from <P>Pétur</P> , <P>Páll</P> and <P>Alexei</P> .
```

## MT evaluation

To evaluate an MT system w.r.t. BLEU run:
```
# This should give a perfect score.
lang=en
ref=testdata/example.ner-ext.$lang
sys=testdata/example.$lang
python mt_eval.py --ref $ref --ref-contains-entities --sys $sys --tgt_lang $lang
```
This will do the following:
- Read the NER markers from the REF.
- Report the BLEU score on the cleaned REF and SYS (as is).
- Run a NER on the SYS.
- Report on NER alignment: 
  - Alignment count: How many NEs we were able to match between REF and SYS.
  - Alignment coverage: The fraction of NEs which we were able to able to align, from 0-1, **1 is best**. If REF and SYS do not contain equal counts of NEs, we use the smaller count.
  - Average alignment distance: The average distance in the alignment, from 0-1, **0 is best**.
  - Accuracy: The fraction exact matches in the alignment (string comparison).
- Run the report on each distinct tag found both REF and SYS.

This evaluation can be run with any combination of --ref/sys-contains-entities.

## Analyzing and pairing

(This can be skipped) The next step aligns the two tagged files, and optionally prints some statistics. This step is run automatically by the filtering but can be ran on its own.

```bash
python aligner.py --is_ent testdata/is.ner --en_ent testdata/en.ner --output testdata/alignment.tsv
```

The columns are `ner_tagger_1, source_1, ner_tagger_2, source_2, match_code, max_distance (1-JarWink), alignment spans`

```
is		hf		1	0.06999999999999995	0:1:Person-5:6:PER 5:7:Person-0:2:PER
is		hf		1	0.12	0:1:Person-0:1:PER 4:5:Person-9:10:PER 6:7:Person-5:6:PER 8:9:Person-7:8:PER
```

## Filtering and POS tagging
This step parses the named files, aligns entities and pos tags them.

```bash
python postagger.py --is_ent testdata/is.ner --en_ent testdata/en.ner --output testdata/en_is.pos.tsv
```

The resulting file contains tags indicating which entity ID and part of speech (POS) a given name has in the Icelandic side.

```
<e:0:nkee-s:>Einar Jónsson</e0> was visited by <e:1:nven-s:>Guðrún</e1> .	<e:1:nven-s:>Guðrún</e1> fór í heimsókn til <e:0:nkee-s:>Einars Jónssonar</e0> .
<e:0:nven-s:>Anna</e0> got a gift from <e:1:nkeþ-s:>Pétur</e1> , <e:2:nkeþ-s:>Páll</e2> and <e:3:nkeþ-s:>Alexei</e3> .	<e:0:nven-s:>Anna</e0> fékk gjöf frá <e:3:nkeþ-s:>Alexei</e3> , <e:1:nkeþ-s:>Pétri</e1> og <e:2:nkeþ-s:>Páli</e2> .
```

## Substituting

Finally, given a list of tab separated genders (kk and kvk) and sufficient names such as 

```
kk  Þröstur Helagson
kk  Jón Jónsson
kk  Bubbi Morthens
kk  Ingvar Gunnarsson
kvk Sigga
kvk Sigríður Einarsdóttir
```

we can then generate a synthetic parallel corpus with randomly inserted names (full names and first names) using

```bash
python patcher.py --input testdata/en_is.pos.tsv --output testdata/en_is.synth.tsv --names testdata/names.txt
```

which outputs

```
Jón was visited by Sigríður .	Sigríður fór í heimsókn til Jóns .
Sigga got a gift from Bubbi Morthens , Ingvar and Jón .	Sigga fékk gjöf frá Jóni , Bubba Morthens og Ingvari .
```
