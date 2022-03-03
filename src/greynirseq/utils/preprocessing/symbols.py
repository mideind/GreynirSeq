# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

SYMBOL_WHITELIST = set(
    [
        " ",  # 0x20        SPACE
        "!",  # 0x21        EXCLAMATION MARK
        "#",  # 0x23        NUMBER SIGN
        "$",  # 0x24        DOLLAR SIGN
        "%",  # 0x25        PERCENT SIGN
        "&",  # 0x26        AMPERSAND
        "'",  # 0x27        APOSTROPHE
        "(",  # 0x28        LEFT PARENTHESIS
        ")",  # 0x29        RIGHT PARENTHESIS
        "*",  # 0x2a        ASTERISK
        "+",  # 0x2b        PLUS SIGN
        ",",  # 0x2c        COMMA
        "-",  # 0x2d        HYPHEN-MINUS
        ".",  # 0x2e        FULL STOP
        "/",  # 0x2f        SOLIDUS
        "0",  # 0x30        DIGIT ZERO
        "1",  # 0x31        DIGIT ONE
        "2",  # 0x32        DIGIT TWO
        "3",  # 0x33        DIGIT THREE
        "4",  # 0x34        DIGIT FOUR
        "5",  # 0x35        DIGIT FIVE
        "6",  # 0x36        DIGIT SIX
        "7",  # 0x37        DIGIT SEVEN
        "8",  # 0x38        DIGIT EIGHT
        "9",  # 0x39        DIGIT NINE
        ":",  # 0x3a        COLON
        ";",  # 0x3b        SEMICOLON
        "<",  # 0x3c        LESS-THAN SIGN
        "=",  # 0x3d        EQUALS SIGN
        ">",  # 0x3e        GREATER-THAN SIGN
        "?",  # 0x3f        QUESTION MARK
        "@",  # 0x40        COMMERCIAL AT
        "A",  # 0x41        LATIN CAPITAL LETTER A
        "B",  # 0x42        LATIN CAPITAL LETTER B
        "C",  # 0x43        LATIN CAPITAL LETTER C
        "D",  # 0x44        LATIN CAPITAL LETTER D
        "E",  # 0x45        LATIN CAPITAL LETTER E
        "F",  # 0x46        LATIN CAPITAL LETTER F
        "G",  # 0x47        LATIN CAPITAL LETTER G
        "H",  # 0x48        LATIN CAPITAL LETTER H
        "I",  # 0x49        LATIN CAPITAL LETTER I
        "J",  # 0x4a        LATIN CAPITAL LETTER J
        "K",  # 0x4b        LATIN CAPITAL LETTER K
        "L",  # 0x4c        LATIN CAPITAL LETTER L
        "M",  # 0x4d        LATIN CAPITAL LETTER M
        "N",  # 0x4e        LATIN CAPITAL LETTER N
        "O",  # 0x4f        LATIN CAPITAL LETTER O
        "P",  # 0x50        LATIN CAPITAL LETTER P
        "Q",  # 0x51        LATIN CAPITAL LETTER Q
        "R",  # 0x52        LATIN CAPITAL LETTER R
        "S",  # 0x53        LATIN CAPITAL LETTER S
        "T",  # 0x54        LATIN CAPITAL LETTER T
        "U",  # 0x55        LATIN CAPITAL LETTER U
        "V",  # 0x56        LATIN CAPITAL LETTER V
        "W",  # 0x57        LATIN CAPITAL LETTER W
        "X",  # 0x58        LATIN CAPITAL LETTER X
        "Y",  # 0x59        LATIN CAPITAL LETTER Y
        "Z",  # 0x5a        LATIN CAPITAL LETTER Z
        "[",  # 0x5b        LEFT SQUARE BRACKET
        "]",  # 0x5d        RIGHT SQUARE BRACKET
        "^",  # 0x5e        CIRCUMFLEX ACCENT
        "a",  # 0x61        LATIN SMALL LETTER A
        "b",  # 0x62        LATIN SMALL LETTER B
        "c",  # 0x63        LATIN SMALL LETTER C
        "d",  # 0x64        LATIN SMALL LETTER D
        "e",  # 0x65        LATIN SMALL LETTER E
        "f",  # 0x66        LATIN SMALL LETTER F
        "g",  # 0x67        LATIN SMALL LETTER G
        "h",  # 0x68        LATIN SMALL LETTER H
        "i",  # 0x69        LATIN SMALL LETTER I
        "j",  # 0x6a        LATIN SMALL LETTER J
        "k",  # 0x6b        LATIN SMALL LETTER K
        "l",  # 0x6c        LATIN SMALL LETTER L
        "m",  # 0x6d        LATIN SMALL LETTER M
        "n",  # 0x6e        LATIN SMALL LETTER N
        "o",  # 0x6f        LATIN SMALL LETTER O
        "p",  # 0x70        LATIN SMALL LETTER P
        "q",  # 0x71        LATIN SMALL LETTER Q
        "r",  # 0x72        LATIN SMALL LETTER R
        "s",  # 0x73        LATIN SMALL LETTER S
        "t",  # 0x74        LATIN SMALL LETTER T
        "u",  # 0x75        LATIN SMALL LETTER U
        "v",  # 0x76        LATIN SMALL LETTER V
        "w",  # 0x77        LATIN SMALL LETTER W
        "x",  # 0x78        LATIN SMALL LETTER X
        "y",  # 0x79        LATIN SMALL LETTER Y
        "z",  # 0x7a        LATIN SMALL LETTER Z
        "{",  # 0x7b        LEFT CURLY BRACKET
        "|",  # 0x7c        VERTICAL LINE
        "}",  # 0x7d        RIGHT CURLY BRACKET
        "~",  # 0x7e        TILDE
        "£",  # 0xa3        POUND SIGN
        "¥",  # 0xa5        YEN SIGN
        # "®",  # 0xae        REGISTERED SIGN
        # "¯",  # 0xaf        MACRON
        # "±",  # 0xb1        PLUS-MINUS SIGN
        "´",  # 0xb4        ACUTE ACCENT
        # "µ",  # 0xb5        MICRO SIGN
        "Á",  # 0xc1        LATIN CAPITAL LETTER A WITH ACUTE
        "Æ",  # 0xc6        LATIN CAPITAL LETTER AE
        "É",  # 0xc9        LATIN CAPITAL LETTER E WITH ACUTE
        "Í",  # 0xcd        LATIN CAPITAL LETTER I WITH ACUTE
        "Ð",  # 0xd0        LATIN CAPITAL LETTER ETH
        "Ó",  # 0xd3        LATIN CAPITAL LETTER O WITH ACUTE
        "Ö",  # 0xd6        LATIN CAPITAL LETTER O WITH DIAERESIS
        "Ú",  # 0xda        LATIN CAPITAL LETTER U WITH ACUTE
        "Ý",  # 0xdd        LATIN CAPITAL LETTER Y WITH ACUTE
        "Þ",  # 0xde        LATIN CAPITAL LETTER THORN
        "á",  # 0xe1        LATIN SMALL LETTER A WITH ACUTE
        "æ",  # 0xe6        LATIN SMALL LETTER AE
        "é",  # 0xe9        LATIN SMALL LETTER E WITH ACUTE
        "í",  # 0xed        LATIN SMALL LETTER I WITH ACUTE
        "ð",  # 0xf0        LATIN SMALL LETTER ETH
        "ó",  # 0xf3        LATIN SMALL LETTER O WITH ACUTE
        "ö",  # 0xf6        LATIN SMALL LETTER O WITH DIAERESIS
        "ú",  # 0xfa        LATIN SMALL LETTER U WITH ACUTE
        "ý",  # 0xfd        LATIN SMALL LETTER Y WITH ACUTE
        "þ",  # 0xfe        LATIN SMALL LETTER THORN
        "‘",  # 0x2018      LEFT SINGLE QUOTATION MARK
        "’",  # 0x2019      RIGHT SINGLE QUOTATION MARK
        "‚",  # 0x201a      SINGLE LOW-9 QUOTATION MARK
        "‛",  # 0x201b      SINGLE HIGH-REVERSED-9 QUOTATION MARK
        "“",  # 0x201c      LEFT DOUBLE QUOTATION MARK
        "”",  # 0x201d      RIGHT DOUBLE QUOTATION MARK
        "„",  # 0x201e      DOUBLE LOW-9 QUOTATION MARK
        # "‟",  # 0x201f      DOUBLE HIGH-REVERSED-9 QUOTATION MARK
        "•",  # 0x2022      BULLET
        # "‧",  # 0x2027      HYPHENATION POINT
        # "≤",  # 0x2264      LESS-THAN OR EQUAL TO
        # "≥",  # 0x2265      GREATER-THAN OR EQUAL TO
        '"',  # 0x22        QUOTATION MARK
        "–",  # 0x2013        EN-DASH
        "—",  # 0x2014        EM-DASH
    ]
)

RARE_SYMBOLS = (
    "äüăÀìšŽčêłçţßžÄàåşÅøôâãČāńėŠīćżųÜñęąĂﬃõëîțěřħőȚēÈūŢċØŁťśľđșňïġůĄĮŘļò"
    "ȝģŚËÂįņűœźÔȘÏĒÕŲĪŅÇķÌĻƍȜĘÎÑĆŐŻĀÃĜďûŤŮİĳĊĦȕùŒȠĖŪĚǻĞɨĤĐÊĺƛÒŞȞɟƗƒʊɢệƝɧĶı"
    "ĠɥȡɪɚɴɛƫʌǿȀÿǼÙŸɱÛĲȅȂȑȓȗȁĈǹɫȇɬɝǾǋȟƖũɡȐĽȖƠƥɹȢǺȣȻƮɳɣȍƯȪĹơƦ➍➌➐➋➑➎➊"
)

GREEK_SYMBOLS = "αμονηιρεςστ·ΑλβκίΕγυΣΔάωπδΤόΚΠΜώφήΟχθΡέΙΝΛύΥΩΦΗΓΒξΘΊΧΨζ΄ΆψΌϊΞΈΎΪΐϑ;"

CYRILLIC_SYMBOLS = "аинеотрслвкСдИяАпНзцугъЕмбРДТКЛОфПчжМВЯхБЗГщЖшУйЪЧФЦЙЩХюЮШөЄьІы"

MATH_SYMBOLS = "𝐶𝑡𝑖𝐾𝑛𝑇𝐵𝐹𝑓𝑒𝑅𝑘𝑉𝐼𝐂𝑊𝑑𝐿𝐑𝑚𝐻𝑀𝑦𝑤𝑣𝑏𝐈𝑥𝑐𝐃𝑃𝑋𝐏𝑟𝐴𝑔𝑎𝑝𝑠𝐺𝑗𝜌𝑆"

HEBREW_SYMBOLS = "פדיאטלבעתשמכהסצגרחקזנו"

THAI_SYMBOLS = "ยบรถณกฑ"

ARABIC_SYMBOLS = "".join(chr(i) for i in range(0x0600, 0x06FF + 1))

UNNAMED_SYMBOLS = (
    "\uf106\uf8e7\x94\x81\uf0fc\uf0a3\uf8ec\uf06d\uf8f8\uf8fe\uf8ed\x89\uf0b4"
    "\uf8fd\x97\x84\uf8f7\uf8f6\uf8eb\x90\uf8fc\x93\uf8fa\uf6da\x9f\x86\x9c"
    "\uf020\x8a\x9a\uf8ef\uf8fb\uf0e8\uf03d\x83\uf8f9\uf8ee\uf0b1\uf8f0\uf0be"
    "\x8b\uf0d7\x8d"
)

BAD_SYMBOLS_OS2018 = "âğųôŨşčėëÄĀîŞĘÈęÔÅŽı°àÕåÂŌōèŊžˇŋį�ØÀ¤û¿˛ãīŖøï¨š`ŲÜũĄäüõĶČñņ…"

MISC_SYMBOLS = (
    "\uf0b3\u200c\uf06d\uf0a3\uf0b0\uf0af\uf0b1\uf061\uf0ad\uf0e8\uf0ab\uf0ae"
    "\uf062\uf0a5\uf03e\uf05d\uf05b\uf025\uf0b7\uf0a7\uf022\uf023\uf03d\uf031"
    "\uf044\uf030\uf02a\uf07b\uf07d\uf02b\uf07e\uf0da\uf0ac\uf02f\uf037\uf066"
    "\uf081\uf082\uf084\uf0d7\uf083\uf0e0\uf04c\uf024\uf02e\uf065\uf064\uf085"
    "\uf032\uf020\uf077\uf086\uf0fa\uf0f9\uf0e9\uf0d9\uf06b\u200b\uf0b8\uf06c"
    "\uf0bd\uf0a2\u200e\uf029\uf088\uf06f\uf034\uf087\uf06a\uf071\uf0df\uf0dc"
    "\uf0f1\uf0d8\uf063\uf049\uf0a6\uf0ea\uf041\uf076\uf03a\uf09f\uf02c\uf03c"
    "\uf067\uf02d\uf074\uf0b4"
    "→‡¶·⁺¡◊−─″˂―∙ﬁ©«»˚³♦ﬂ∆¼ª‰′‒‐┼●ʼ♯▲⏐≧¦‑♠ớ̶⁄̴׀¾˃¬∼○¸║ƙ›¢↘̈"
    "↑↓²↔×§½_º（）"
)

BANNED_SYMBOLS = (
    RARE_SYMBOLS
    + GREEK_SYMBOLS
    + CYRILLIC_SYMBOLS
    + MATH_SYMBOLS
    + THAI_SYMBOLS
    + HEBREW_SYMBOLS
    + BAD_SYMBOLS_OS2018
    + MISC_SYMBOLS
    + ARABIC_SYMBOLS
    + "\\"
)

QUOTE_LIKE = "\"'„“”«»‛ʼ″´`′‘’`,"


class ICE_QUOTE:
    class PRIMARY:
        LEFT = "„"  # 0x201e  DOUBLE LOW-9 QUOTATION MARK
        RIGHT = "“"  # 0x201c  LEFT DOUBLE QUOTATION MARK
        BOTH = [LEFT, RIGHT]

    class SECONDARY:
        LEFT = "‚"  # 0x201a  SINGLE LOW-9 QUOTATION MARK
        RIGHT = "‘"  # 0x2018  LEFT SINGLE QUOTATION MARK
        BOTH = [LEFT, RIGHT]

    ALL = PRIMARY.BOTH + SECONDARY.BOTH


PUNCTUATION_SYMBOLS = (
    "'ʼ∞¥≈€∂‧Ω÷‐℉†℃‛″£™∙§«»@¯^!½³²˜−{$¼¹≠}º‗®‑#¡´&`|·≥―′¿<≤~?±" + '…\\>”_+][°–=*"‘%„“;:-•(),…–`-—!’?;“”:.,'
)
NON_STANDARD_SPACES = (
    "\u0009",  # CHARACTER TABULATION
    "\u00a0",  # NO-BREAK SPACE
    "\u115F",  # Hangul filler
    "\u1160",  # Hangul filler
    "\u1680",  # OGHAM SPACE MARK
    *tuple(chr(i) for i in range(0x2000, 0x200A + 1)),  # Various spaces
    "\u202F",  # NARROW NO-BREAK SPACE
    "\u205F",  # MEDIUM MATHEMATICAL SPACE
    "\u3000",  # IDEOGRAPHIC SPACE
    "\u2800",  # BRAILLE PATTERN BLANK
    "\u3164",  # HANGUL JUNGSEONG FILLER
    "\uFFA0",  # HALFWIDTH HANGUL LETTER HAN
)
SEPARATORS = (
    "\u000A",  # LINE FEED (LF)
    "\u000B",  # LINE TABULATION (VT)
    "\u000C",  # FORM FEED (FF)
    "\u000D",  # CARRIAGE RETURN (CR)
    "\u0085",  # NEXT LINE (NEL)
    "\u2028",  # LINE SEPARATOR
    "\u2029",  # PARAGRAPH SEPARATOR
)
DASHES = (*tuple(chr(i) for i in range(0x2010, 0x2015 + 1)),)  # Different types of dashes

SUBSTITUTE_FOR_NULL = (
    *tuple(chr(i) for i in range(0x0000, 0x0008 + 1)),  # C0 control codes - without separators
    *tuple(chr(i) for i in range(0x000E, 0x001F + 1)),  # C0 control codes - without separators
    *tuple(chr(i) for i in range(0x0080, 0x009F + 1)),  # C1 control codes
    "\u00ad",  # 0xad SOFT HYPHEN
    "\u034F",  # COMBINING GRAPHEME JOINER
    "\u17B4",  # Khmer vowel sign aq
    "\u17B5",  # Khmer vowel sign aa
    "\u180E",  # MONGOLIAN VOWEL SEPARATOR
    *tuple(chr(i) for i in range(0x200B, 0x200F + 1)),  # Format characters
    *tuple(chr(i) for i in range(0x202A, 0x202E + 1)),  # More format characters
    *tuple(chr(i) for i in range(0x2060, 0x206F + 1)),  # More format characters and invisible separators
    *tuple(chr(i) for i in range(0xFFF0, 0xFFFF + 1)),  # Specials
    "\ufe0f",  # Variation Selector-16 - used before emojis
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE
)
