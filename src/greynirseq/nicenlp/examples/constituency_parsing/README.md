# Instructions for Miðeind neural parser (experimental)

The docker container can be also be located on dockerhub (`docker pull mideind/neural-parser`). To load the saved image into docker use:

```
docker load -i neural-parser.tar
```

The input to the parser is a text file (`${INPUT_FILE}`) where each line contains a sentence that will be parsed.
The output file will be located in `${OUTPUT_DIR}/output.txt` . To run the parser use the following:

```
docker run --volume ${INPUT_FILE}:/data/input.txt --volume ${OUTPUT_DIR}:/data/ mideind/neural-parser
```

---

# Leiðbeiningar fyrir tauganetsþáttara Miðeindar

Athugið að einnig má finna gáminn á dockerhub (`docker pull mideind/neural-parser`). Til að hlaða vistuðum gámi inn í docker skal nota:

```
docker load -i neural-parser.tar
```

Inntakið í þáttarann er texta skrá (`${INPUT_FILE}`) þar sem hver lína geymir eina málsgrein. Eftir keyrslu má finna úttakið í skránni `${OUTPUT_DIR}/output.txt` .
Til að keyra þáttarann skal nota:

```
docker run --volume ${INPUT_FILE}:/data/input.txt --volume ${OUTPUT_DIR}:/data/ mideind/neural-parser
```
