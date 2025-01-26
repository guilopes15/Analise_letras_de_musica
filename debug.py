import polars as pl
from cleaner import df


printable = df.with_columns(
    pl.col('letra')
    .alias('printable')
    .map_elements(lambda x: x.isprintable(), return_dtype=bool)
)

debug = printable.filter(pl.col('printable') == False)
print(len(df), len(debug))
print(repr(debug[0]['letra'][0]))


