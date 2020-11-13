# Error codes in iceErrorCorpus that are considered out of scope
# for GreynirCorrect, at this stage at least
OUT_OF_SCOPE = {
    "agreement-pro",  # samræmi fornafns við undanfara  grammar ...vöðvahólf sem sé um dælinguna. Hann dælir blóðinu > Það dælir blóðinu
    "aux",  # meðferð vera og verða, hjálparsagna   wording mun verða eftirminnilegt > mun vera eftirminnilegt
    "bracket4square",  # svigi fyrir hornklofa  punctuation (Portúgal) > [Portúgal]
    "collocation-idiom",  # fast orðasamband með ógagnsæja merkingu collocation hélt hvorki vindi né vatni > hélt hvorki vatni né vindi
    "collocation",  # fast orðasamband  collocation fram á þennan dag > fram til þessa dags
    "comma4conjunction",  # komma fyrir samtengingu punctuation ...fara með vald Guðs, öll löggjöf byggir... > ...fara með vald Guðs og öll löggjöf byggir...
    "comma4dash",  # komma fyrir bandstrik  punctuation , > -
    "comma4ex",  # komma fyrir upphrópun    punctuation Viti menn, almúginn... > Viti menn! Almúginn...
    "comma4period",  # komma fyrir punkt    punctuation ...kynnast nýju fólki, er á þrítugsaldri > ...kynnast nýju fólki. Hann er á þrítugsaldri
    "comma4qm",  # komma fyrir spurningarmerki  punctuation Höfum við réttinn, eins og að... > Höfum við réttinn? Eins og að...
    "conjunction4comma",  # samtenging fyrir kommu  punctuation ...geta orðið þröngvandi og erfitt getur verið... > ...geta orðið þröngvandi, erfitt getur verið...
    "conjunction4period",  # samtenging fyrir punkt punctuation ...tónlist ár hvert og tónlistarstefnurnar eru orðnar... > ...tónlist ár hvert. Tónlistarstefnurnar eru orðnar...
    "context",  # rangt orð í samhengi  other
    "dash4semicolon",  # bandstrik fyrir semíkommu  punctuation núna - þetta > núna; þetta
    "def4ind",  # ákveðið fyrir óákveðið    grammar skákinni > skák
    "dem-pro",  # hinn í stað fyrir sá; sá ekki til eða ofnotað grammar hinn > sá
    "dem4noun",  # ábendingarfornafn í stað nafnorðs    grammar hinn > maðurinn
    "dem4pers",  # ábendingarfornafn í stað persónufornafns grammar þessi > hún
    "extra-comma",  # auka komma    punctuation stríð, við náttúruna > stríð við náttúruna
    "extra-number",  # tölustöfum ofaukið   other   139,0 > 139
    "extra-period",  # auka punktur punctuation á morgun. Og ... > á morgun og...
    "extra-punctuation",  # auka greinarmerki   punctuation ... að > að
    "extra-space",  # bili ofaukið  spacing 4 . > 4.
    "extra-symbol",  # tákn ofaukið other   Dalvík + gaf... > Dalvík gaf...
    "extra-word",  # orði ofaukið   insertion   augun á mótherja > augu mótherja
    "extra-words",  # orðum ofaukið insertion   ...ég fer að hugsa... > ...ég hugsa...
    "foreign-error",  # villa í útlendu orði    foreign Supurbowl > Super Bowl
    "fw4ice",  # erlent orð þýtt yfir á íslensku    style   Elba > Saxelfur
    "gendered",  # kynjað mál, menn fyrir fólk  exclusion   menn hugsa oft > fólk hugsar oft
    "ice4fw",  # íslenskt orð notað í stað erlends      Demókrata öldungarþings herferðarnefndina > Democratic Senatorial Campaign Committee
    "ind4def",  # óákveðið fyrir ákveðið    grammar gítartakta > gítartaktana
    "ind4sub",  # framsöguháttur fyrir vh.  grammar Þrátt fyrir að konfúsíanismi er upprunninn > Þrátt fyrir að konfúsíanismi sé upprunninn
    "indef-pro",  # óákveðið fornafn    grammar enginn > ekki neinn
    "it4nonit",  # skáletrað fyrir óskáletrað       Studdi Isma'il > Studdi Isma'il
    "loan-syntax",  # lánuð setningagerð    style   ég vaknaði upp > ég vaknaði
    "missing-commas",  # kommur vantar utan um innskot  punctuation Hún er jafn verðmæt ef ekki verðmætari en háskólapróf > Hún er verðmæt, ef ekki verðmætari, en háskólapróf
    "missing-conjunction",  # samtengingu vantar    punctuation í Noregi suður að Gíbraltarsundi > í Noregi og suður að Gíbraltarsundi
    "missing-ex",  # vantar upphrópunarmerki    punctuation Viti menn ég komst af > Viti menn! Ég komst af
    "missing-quot",  # gæsalöpp vantar  punctuation „I'm winning > „I'm winning“
    "missing-quots",  # gæsalappir vantar   punctuation I'm winning > „I'm winning“
    "missing-semicolon",  # vantar semíkommu    punctuation Haukar Björgvin Páll > Haukar; Björgvin Páll
    # "missing-space",        # vantar bil  spacing eðlis-og efnafræði > eðlis- og efnafræði
    "missing-square",  # vantar hornklofi   punctuation þeir > [þeir]
    "missing-symbol",  # tákn vantar    punctuation 0 > 0%
    "missing-word",  # orð vantar   omission    í Donalda > í þorpinu Donalda
    "missing-words",  # fleiri en eitt orð vantar   omission    því betri laun > því betri laun hlýtur maður
    "nonit4it",  # óskáletrað fyrir skáletrað       orðið qibt > orðið qibt
    "noun4dem",  # nafnorð í stað ábendingarfornafns    grammar stærsta klukkan > sú stærsta
    "noun4pro",  # nafnorð í stað fornafns  grammar menntun má nálgast > hana má nálgast
    "past4pres",  # sögn í þátíð í stað nútíðar grammar þegar hún leigði spólur > þegar hún leigir spólur
    "period4comma",  # punktur fyrir kommu  punctuation meira en áður. Hella meira í sig > meira en áður, hella meira í sig
    "period4conjunction",  # punktur fyrir samtengingu  punctuation ...maður vill gera. Vissulega > ...maður vill gera en vissulega
    "period4ex",  # punktur fyrir upphrópun punctuation Viti menn. > Viti menn!
    "pers4dem",  # persónufornafn í staðinn fyrir ábendingarf.  grammar það > þetta
    "pres4past",  # sögn í nútíð í stað þátíðar grammar Þeir fara út > Þeir fóru út
    "pro4noun",  # fornafn í stað nafnorðs  grammar þau voru spurð > parið var spurt
    "pro4reflexive",  # nafnorð í stað afturbeygðs fornafns grammar gefur orku til fólks í kringum það > gefur orku til fólks í kringum sig
    "pro4reflexive",  # persónufornafn í stað afturbeygðs fn.   grammar Fólk heldur að það geri það hamingjusamt > Fólk heldur að það geri sig hamingjusamt
    "punctuation",  # greinarmerki  punctuation hún mætti og hann var ekki tilbúinn > hún mætti en hann var ekki tilbúinn
    "qm4ex",  # spurningarmerki fyrir upphrópun punctuation Algjört hrak sjálf? > Algjört hrak sjálf!
    "reflexive4noun",  # afturbeygt fornafn í stað nafnorðs grammar félagið hélt aðalfund þess > félagið hélt aðalfund sinn
    "reflexive4pro",  # afturbeygt fornafn í stað persónufornafns   grammar gegnum líkama sinn > gegnum líkama hans
    "simple4cont",  # nútíð í stað vera að + nafnh. grammar ók > var að aka
    "square4bracket",  # hornklofi fyrir sviga  punctuation [börnin] > (börnin)
    "style",  # stíll   style   urðu ekkert frægir > urðu ekki frægir
    "sub4ind",  # viðtengingarh. fyrir fh.  grammar Stjórnvöld vildu auka rétt borgara og geri þeim kleift > Stjórnvöld vildu auka rétt borgara og gera þeim kleift
    "unicelandic",  # óíslenskuleg málnotkun    style   ...fer eftir persónunni... > ...fer eftir manneskjunni...
    "upper4lower-proper",  # stór stafur í sérnafni þar sem hann á ekki að vera capitalization  Mál og Menning > Mál og menning
    "wording",  # orðalag   wording ...gerðum allt í raun... > ...gerðum í raun allt...
    "xxx",  # unclassified  unclassified
    "zzz",  # to revisit    unannotated
}


# Give each error type a fixed number. This is useful for running neural network classifiers on the error corpus.
ERROR_NUMBERS = {
    "lower4upper-initial": 1,
    "lower4upper-proper": 2,
    "lower4upper-acro": 3,
    "upper4lower-common": 4,
    "upper4lower-proper": 5,
    "upper4lower-noninitial": 6,
    "caps4low": 7,
    "collocation": 8,
    "collocation-idiom": 9,
    "agreement": 10,
    "agreement-concord": 11,
    "agreement-pred": 12,
    "agreement-pro": 13,
    "ind4def": 14,
    "def4ind": 15,
    "ind4sub": 16,
    "sub4ind": 17,
    "verb-inflection": 18,
    "nominal-inflection": 19,
    "plural4singular": 20,
    "singular4plural": 21,
    "v3": 22,
    "v3-subordinate": 23,
    "compound-collocation": 24,
    "compound-nonword": 25,
    "nonword": 26,
    "missing-letter": 27,
    "missing-word": 28,
    "missing-words": 29,
    "swapped-letters": 30,
    "comma4period": 31,
    "comma4qm": 32,
    "comma4colon": 33,
    "double-punctuation": 34,
    "extra-abbreviation": 35,
    "extra-dash": 36,
    "iteration-colon": 37,
    "missing-colon": 38,
    "missing-comma": 39,
    "missing-commas": 40,
    "missing-period": 41,
    "missing-qm": 42,
    "missing-conjunction": 43,
    "missing-quot": 44,
    "missing-quots": 45,
    "misplaced-quot": 46,
    "wrong-quots": 47,
    "extra-quot": 48,
    "extra-quots": 49,
    "punctuation": 50,
    "extra-punctuation": 51,
    "extra-comma": 52,
    "extra-period": 53,
    "period4comma": 54,
    "period4colon": 55,
    "period4conjunction": 56,
    "conjunction4period": 57,
    "conjunction4comma": 58,
    "comma4conjunction": 59,
    "period4qm": 60,
    "period-plus-conjunction": 61,
    "comma-plus-conjunction": 62,
    "abbreviation-period": 63,
    "comma4ex": 64,
    "period4ex": 65,
    "semicolon4colon": 66,
    "extra-semicolon": 67,
    "ordinal-period": 68,
    "merged-words": 69,
    "split-compound": 70,
    "split-word": 71,
    "split-words": 72,
    "missing-dash": 73,
    "zzz": 74,
    "xxx": 75,
    "extra-word": 76,
    "extra-words": 77,
    "extra-accent": 78,
    "extra-letter": 79,
    "missing-accent": 80,
    "wording": 81,
    "ngnk": 82,
    "i4y": 83,
    "y4i": 84,
    "í4ý": 85,
    "ý4í": 86,
    "aux": 87,
    "conjunction-drop": 88,
    "adjective-inflection": 89,
    "n4nn": 90,
    "nn4n": 91,
    "fw": 92,
    "foreign-error": 93,
    "dative-sub": 94,
    "dir4loc": 95,
    "loc4dir": 96,
    "gendered": 97,
    "number4word": 98,
    "word4number": 99,
    "style": 100,
    "unicelandic": 101,
    "bad-contraction": 102,
    "new-passive": 103,
    "taboo-word": 104,
    "missing-space": 105,
    "extra-space": 106,
    "loan-syntax": 107,
    "each": 108,
    "missing-bracket": 109,
    "extra-bracket": 110,
    "noun4pro": 111,
    "pro4noun": 112,
    "reflexive4noun": 113,
    "pro4reflexive": 114,
    "u4y": 115,
    "conjunction": 116,
    "extra-conjunction": 117,
    "pronun-writing": 118,
    "semicolon4comma": 119,
    "conjunction4qm": 120,
    "wrong-prep": 121,
    "pres4past": 122,
    "past4pres": 123,
    "letter-rep": 124,
    "missing-slash": 125,
    "pro4reflexive": 126,
    "reflexive4pro": 127,
    "comma4bracket": 128,
    "qm4comma": 129,
    "kv4hv": 130,
    "hv4kv": 131,
    "bracket4slash": 132,
    "að4af": 133,
    "af4að": 134,
    "missing-ex": 135,
    "words4abbreviation": 136,
    "abbreviation4words": 137,
    "qm4ex": 138,
    "qm4period": 139,
    "pers4dem": 140,
    "dem-pro": 141,
    "indef-pro": 142,
    "marked4unmarked": 143,
    "adj4adv": 144,
    "adv4adj": 145,
    "have": 146,
    "context": 147,
    "cont4simple": 148,
    "missing-inf-part": 149,
    "fw4ice": 150,
    "bracket4square": 151,
    "square4bracket": 152,
    "dash4comma": 153,
    "date-period": 154,
    "comma4semicolon": 155,
    "word4dash": 156,
    "dash4word": 157,
    "missing-semicolon": 158,
    "symbol4word": 159,
    "want": 160,
    "abbreviation": 161,
    "dem4pers": 162,
    "nom4acc-sub": 163,
    "acc4nom-sub": 164,
    "slash4or": 165,
    "date-abbreviation": 166,
    "simple4cont": 167,
    "name-error": 168,
    "dash4period": 169,
    "ex4comma": 170,
    "colon4period": 171,
    "colon4comma": 172,
    "ex4period": 173,
    "extra-colon": 174,
    "nonit4it": 175,
    "bracket4comma": 176,
    "extra-inf-part": 177,
    "extra-qm": 178,
    "comma4dash": 179,
    "dash4semicolon": 180,
    "wrong-dash": 181,
    "it4nonit": 182,
    "hypercorr": 183,
    "dash4colon": 184,
    "extra-symbol": 185,
    "dots4comma": 186,
    "comma4dots": 187,
    "missing-symbol": 188,
    "dots4period": 189,
    "extra-number": 190,
    "extra-square": 191,
    "bracket4period": 192,
    "word4symbol": 193,
    "nonsup4sup": 194,
    "sup4nonsup": 195,
    "semicolon4period": 196,
    "period4semicolon": 197,
    "period4dash": 198,
    "gen-escape": 199,
    "symbol4number": 200,
    "missing-square": 201,
    "slash4dash": 202,
    "wrong-accent": 203,
    "dem4noun": 204,
    "noun4dem": 205,
    "ice4fw": 206,
    "extra-commas": 207,
    "number4symbol": 208,
    "conjunction4semicolon": 209,
    "repeat-word": 210,
    "repeat-word-split": 211,
    "number-fail": 212,
}
