# Named Entity Processing Pipeline for NMT

In training data for NMT (neural machine translation) systems it is of benefit to have a large and varried corpus. Unfortunately this is not often the case. This submodule implements a pipeline for tagging, filtering, matching and substituing named entities in a parallel English to Icelandic corpus. Currently it only support Persons but it should not be much work to extend this to other label sets.

### Name Tagging

For Icelandic NER the included IceBERT-NER model is used. For english a NER model fine tuned on BERT large from huggingface is used with spacy as fallback if sentence length is too long for the model to process. Note that this results in downloading of data beyond 1Gb.

The script accepts a tab separated English to Icelandic file, e.g.

```
Einar Jónsson was visited by Guðrún.	Guðrún fór í heimsókn til Einars Jónssonar.
Anna got a gift from Pétur, Páll and Alexei.    Anna fékk gjöf frá Alexei, Pétri og Páli.
```

```bash
python nertagger.py --language is --input testdata/en_is.tsv --output testdata/is.ner
python nertagger.py --language en --input testdata/en_is.tsv --output testdata/en.ner
```

Which writes to file

```
Guðrún fór í heimsókn til Einars Jónssonar .	B-Person O O O O B-Person I-Person O
Anna fékk gjöf frá Alexei , Pétri og Páli .	B-Person O O O B-Person O B-Person O B-Person O
```

and for English (the last column is sp if the spacy fallback was used)

```
Einar Jónsson was visited by Guðrún .	I-PER I-PER O O O I-PER O	hf
Anna got a gift from Pétur , Páll and Alexei .  I-PER O O O O I-PER O I-PER O I-PER O I-PER O O O I-PER O I-PER O I-PER O	hf
```

Note the different tagsets used, this is dealt with by the aligner.


### Analyzing and pairing

(This can be skipped) The next step aligns the two tagged files, and optionally prints some statistics. This step is run automatically by the filtering but can be ran on its own.

```bash
python aligner.py --is_ent testdata/is.ner --en_ent testdata/en.ner --output testdata/alignment.tsv
```

The columns are `ner_tagger_1, source_1, ner_tagger_2, source_2, match_code, max_distance (1-JarWink), alignment spans`

```
is		hf		1	0.06999999999999995	0:1:Person-5:6:PER 5:7:Person-0:2:PER
is		hf		1	0.12	0:1:Person-0:1:PER 4:5:Person-9:10:PER 6:7:Person-5:6:PER 8:9:Person-7:8:PER
```

### Filtering and POS tagging
This step parses the named files, aligns entities and pos tags them.

```bash
python postagger.py --is_ent testdata/is.ner --en_ent testdata/en.ner --output testdata/en_is.pos.tsv
```

The resulting file contains tags indicating which entity ID and part of speech (POS) a given name has in the Icelandic side.

```
<e:0:nkee-s:>Einar Jónsson</e0> was visited by <e:1:nven-s:>Guðrún</e1> .	<e:1:nven-s:>Guðrún</e1> fór í heimsókn til <e:0:nkee-s:>Einars Jónssonar</e0> .
<e:0:nven-s:>Anna</e0> got a gift from <e:1:nkeþ-s:>Pétur</e1> , <e:2:nkeþ-s:>Páll</e2> and <e:3:nkeþ-s:>Alexei</e3> .	<e:0:nven-s:>Anna</e0> fékk gjöf frá <e:3:nkeþ-s:>Alexei</e3> , <e:1:nkeþ-s:>Pétri</e1> og <e:2:nkeþ-s:>Páli</e2> .
```

### Substituting

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
Sigríður Einarsdóttir was visited by Ingvar Blöndal .	Ingvar Blöndal fór í heimsókn til Sigríðar Einarsdóttur .
Sigríður got a gift from Ingvar , Jón and Jón Jónsson .	Sigríður fékk gjöf frá Jóni Jónssyni , Ingvari og Jóni .
```

