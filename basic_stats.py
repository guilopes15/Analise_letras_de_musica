import spacy
import polars as pl
# https://spacy.io/usage


nlp = spacy.load('pt_core_news_sm')


def tokens(text: str) -> int:
    doc = nlp(text)
    return len([t.text for t in doc if not t.is_punct])


def types(text: str) -> int:
    doc = nlp(text)
    return len(set([t.text for t in doc if not t.is_punct]))


def lemmas(text: str) -> int: 
    doc = nlp(text)
    return len(set([t.lemma_ for t in doc if not t.is_punct]))


def ttr(cols) -> float | None: #token-type ratio
    tokens = cols['tokens']
    types = cols['types']
    
    if tokens is None or types is None:
        return None
    
    return (types / tokens) * 100


def ltor(cols) -> float | None: #token-type ratio
    tokens = cols['tokens']
    lemmas = cols['lemmas']
    
    if tokens is None or lemmas is None:
        return None
    
    return (lemmas / tokens) * 100


def ltyr(cols) -> float | None: #token-type ratio
    types = cols['types']
    lemmas = cols['lemmas']
    
    if types is None or lemmas is None:
        return None
    
    return (lemmas / types) * 100


df = pl.read_csv('deadfish_limpo.csv')

new_df = df.with_columns(
    (
        pl.col('letra').alias('tokens').map_elements(tokens, return_dtype=int)
    ),
    (
        pl.col('letra').alias('types').map_elements(types, return_dtype=int)
    ),
    (
        pl.col('letra').alias('lemmas').map_elements(lemmas, return_dtype=int)
    ),
).with_columns(
    (
        pl.struct(['types', 'tokens'])
        .alias('ttr')
        .map_elements(ttr, return_dtype=float)
    ),
    (
        pl.struct(['lemmas', 'tokens'])
        .alias('ltor')
        .map_elements(ltor, return_dtype=float)
    ),
    (
        pl.struct(['lemmas', 'types'])
        .alias('ltyr')
        .map_elements(ltyr, return_dtype=float)
    ),
)

new_df.write_csv('deadfish_stats.csv')

