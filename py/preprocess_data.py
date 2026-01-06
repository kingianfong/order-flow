# %%


from pathlib import Path


import polars as pl


DATA_DIR = Path(__file__).parent.parent / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PREPROCSSED_DATA_DIR = DATA_DIR / 'preprocessed'


# %%


def extract_syms() -> list[str]:
    result = []
    for partition in RAW_DATA_DIR.iterdir():
        name = partition.name
        assert partition.is_dir() and name.startswith('sym='), f'{partition=}'
        _, sym = name.split('=')
        result.append(sym)
    return result


# effectively groups trades by timestamp and side
def preprocess(sym: str) -> None:
    out_path = PREPROCSSED_DATA_DIR / f'sym={sym}'
    try:
        pl.scan_parquet(out_path).tail().collect()
        return
    except:
        pass

    print(f'preprocessing {sym=}')
    lf = (
        pl.scan_parquet(RAW_DATA_DIR)
        .filter(pl.col('sym') == pl.lit(sym))
        .drop('sym')
        .rename(dict(is_buyer_maker='is_sell'))
        .with_columns(
            pl.col('time').cast(pl.Datetime('ms')),
            is_buy=~pl.col('is_sell'),
        )
        .sort('time')
        .group_by('time', maintain_order=True)
        .agg(
            date=pl.col('date').last(),
            total_count=pl.len(),
            total_qty=pl.col('qty').sum(),
            total_quote_qty=pl.col('quote_qty').sum(),
            vwap=pl.col('quote_qty').sum() / pl.col('qty').sum(),

            buy_count=pl.col('is_buy').sum(),
            sell_count=pl.col('is_sell').sum(),
            buy_qty=pl.when('is_buy').then('qty').sum(),
            sell_qty=pl.when('is_sell').then('qty').sum(),
        )
    )

    lf.sink_parquet(
        pl.PartitionByKey(
            out_path,
            by=pl.col('date'),
            include_key=False,
        ),
        compression='zstd',
        compression_level=10,  # arbitrary
        mkdir=True,
    )


if __name__ == '__main__':
    syms = extract_syms()
    print(f'{syms=}')
    for sym in syms:
        preprocess(sym=sym)


# %%
