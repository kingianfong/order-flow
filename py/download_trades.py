# %%

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import io
import itertools
import zipfile


import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests


# %%


OUTPUT_DIR = Path(__file__).parent.parent / 'data/raw'

SYMBOLS = [
    'BTCUSDT',
    'ETHUSDT',
]


DATES = pd.date_range(
    start=datetime.date(2025, 9, 1),
    end=datetime.date(2025, 10, 15),

    # high volume dates for testing:
    # start=datetime.date(2025, 10, 9),
    # end=datetime.date(2025, 10, 11),
)


# binance um futures trades only
def create_url(sym: str, date: datetime.date) -> str:
    type = 'trades'
    date_str = date.strftime('%Y-%m-%d')
    prefix = 'https://data.binance.vision/data/futures/um/daily'
    return f'{prefix}/{type}/{sym}/{sym}-{type}-{date_str}.zip'


def download_and_save(sym_date: tuple[str, datetime.date]) -> None:
    sym, date = sym_date
    date_str = date.strftime('%Y-%m-%d')
    csv_name = f'{sym.upper()}-trades-{date_str}.csv'

    out_path = OUTPUT_DIR / f'sym={sym}/date={date_str}'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pq.read_table(out_path, use_threads=False)
        return
    except:
        pass

    url = create_url(sym=sym, date=date)

    print(f'downloading for {sym} {date_str}')
    response = requests.get(url)
    response.raise_for_status()
    bytes = response.content

    with zipfile.ZipFile(io.BytesIO(bytes)) as z:
        assert [csv_name,] == z.namelist(), f'{csv_name=}, {z.namelist()=}'
        csv_bytes = z.read(csv_name)

    table = pacsv.read_csv(io.BytesIO(csv_bytes))

    print(f'writing to {out_path}')
    pq.write_to_dataset(
        table,
        out_path,
        use_threads=False,
        basename_template='{i}.parquet',
        existing_data_behavior='delete_matching',
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html
        use_dictionary=False,
        compression='zstd',
        compression_level=14,  # chosen with for loop (including 2025-10-10)
        column_encoding=dict(
            id='DELTA_BINARY_PACKED',
            time='DELTA_BINARY_PACKED',
        ),
    )
    print(f'wrote to {out_path}')


def main():
    print(f'downloading to {OUTPUT_DIR}')
    OUTPUT_DIR.mkdir(parents=False, exist_ok=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(
            download_and_save,
            itertools.product(SYMBOLS, DATES)
        )


if __name__ == '__main__':
    main()


# %%
