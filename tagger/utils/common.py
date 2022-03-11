pad = '<pad>'
unk = '<unk>'
bos = '<bos>'
eos = '<eos>'

language_specific_punct_replace = {
    "de" : {
        "$": "Euro",
        "%": "Prozent"
    },
    "en" : {
        "$": "#",
        "%": "Prozent"
    },
    "es" : {
        "$": "euros",
        "%": "ciento"
    },
    "fr" : {
        "$": "$",
        "%": "$"
    },
    "id" : {
        "$": "$000",
        "%": "persen"
    },
    "it" : {
        "%": "$"
    },
    "pt-br" : {
        "$": "reais",
        "%": "cento"
    },
    "sv" : {
        "%": "procent"
    }
}

punct_replace = {
    "`": "'",
    "``": "''",
    "```": "'''"
}
