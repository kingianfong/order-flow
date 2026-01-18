# %%

from pathlib import Path
import datetime

from jinja2 import Template
from pandas.io.formats.style import Styler
import pandas as pd
import polars as pl


PROJ_ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJ_ROOT_DIR / 'results'
TEMPLATE_PATH = PROJ_ROOT_DIR / 'README.template.md'
README_PATH = PROJ_ROOT_DIR / 'README.md'

VAL_START_DATE = datetime.date(2025, 10, 1)


# %%

_fig_counter = 1


def image(rel_path: str, caption: str) -> str:
    global _fig_counter
    count = _fig_counter
    _fig_counter += 1

    template = Template("""\
<p align="center">
  <img src="{{ path }}" width="{{ width }}">
  <br>Fig {{ count }}: {{ caption }}
</p>""")

    return template.render(
        count=count,
        path=(RESULTS_DIR / rel_path).relative_to(PROJ_ROOT_DIR),
        width='90%',
        caption=caption,
    )


_table_counter = 1


def table(styler: Styler, caption: str) -> str:
    global _table_counter
    count = str(_table_counter)
    _table_counter += 1

    return (
        styler
        .set_caption(f'Table {count}: {caption}')
        .set_table_styles(
            [
                {
                    'selector': '',
                    'props': [
                        ('caption-side', 'bottom'),
                        ('margin-left', 'auto'),
                        ('margin-right', 'auto'),
                    ],
                },
            ],
        )
        # set uuid to ensure determinism for version control
        .to_html(
            table_uuid=str(count),
            doctype_html=False,
        )
    )


# %%


def per_model_results(prefix: str) -> str:
    template = Template(
        """\
<details>
  <summary><b>Model: {{ prefix }}</b></summary>
  <details style="margin-left: 20px;">
    <summary>Parameters</summary>
    {{ params }}
  </details>
  <details style="margin-left: 20px;">
    <summary>Diagnostics</summary>
    {{ diagnostics }}
  </details>
{% if prefix != "constant" %}
  <details style="margin-left: 20px;">
    <summary>Inverse Hessian</summary>
    {{ inv_hessian }}
  </details>
{% endif %}
{% if prefix in ["rbf", "rbf_hawkes"] %}
  <details style="margin-left: 20px;">
    <summary>Seasonality bases</summary>
    {{ bases }}
  </details>
{% endif %}
  <details style="margin-left: 20px;">
    <summary>Predictions</summary>
    {{ predictions }}
  </details>
</details>
""",
        trim_blocks=True,
        lstrip_blocks=False,
    )
    render_kwargs = dict(
        prefix=prefix,
        params=table(
            styler=(
                pd.read_csv(
                    RESULTS_DIR / f'{prefix}/params.csv', index_col=0)
                .style
            ),
            caption=f'{prefix} parameters'
        ),
        diagnostics=table(
            styler=(
                pd.read_csv(RESULTS_DIR / f'{prefix}/diagnostics.csv',
                            index_col=0)
                .style
                .format(
                    dict(
                        mean='{:.4f}',
                        hess_se='{:.4f}',
                        robust_se='{:.4f}',
                        z_score='{:.2f}',
                        p_value='{:.2%}',
                        se_ratio='{:.4f}',
                    )
                )
            ),
            caption=f'{prefix} diagnostics'
        ),
        inv_hessian=image(f'{prefix}/inv_hessian.png',
                          f'{prefix} inverse Hessian'),
        predictions=image(f'{prefix}/counts.png',
                          f'{prefix} predictions'),
    )
    if prefix.startswith('rbf'):
        render_kwargs['bases'] = image(f'{prefix}/bases.png',
                                       f'{prefix} bases')
    return template.render(**render_kwargs)


def generate_readme():
    template = Template(TEMPLATE_PATH.read_text())
    rendered = template.render(
        intro=image(
            'overall/val1.png',
            'Comparison of model outputs',
        ),
        rbf_hawkes_bases=image(
            'rbf_hawkes/bases.png',
            'Baseline fluctuation in trades based on time of day',
        ),
        rbf_hawkes_inv_hess=image(
            'rbf_hawkes/inv_hessian.png',
            'Inverse Hessian to identify potential identifiability issues',
        ),
        data_head=table(
            (
                pl.scan_parquet(PROJ_ROOT_DIR / 'data/raw')
                .filter(pl.col('sym') == 'BTCUSDT')
                .drop('date', 'sym')
                .head()
                .collect()
                .to_pandas()
                .set_index('id')
                .style
            ),
            caption='Raw data',
        ),
        data_stats=table(
            styler=(
                pl.scan_parquet(PROJ_ROOT_DIR / 'data/raw')
                .with_columns(pl.col('time').cast(pl.Datetime('ms')))
                .group_by(
                    subset=(pl.when(pl.col('date') < VAL_START_DATE)
                            .then(pl.lit('train'))
                            .otherwise(pl.lit('validation'))),
                )
                .agg(
                    n_events=pl.len(),
                    n_unique_timestamps=pl.col('time').n_unique(),
                )
                .collect()
                .to_pandas().set_index('subset').sort_index()
                .style.format('{:,}')
            ),
            caption='Sample counts'
        ),
        loglik_mean=table(
            styler=(
                pl.read_csv(RESULTS_DIR / 'overall/loglik_mean.csv')
                .with_columns(subset=pl.when(pl.col('is_train'))
                              .then(pl.lit('train')).otherwise(pl.lit('validation')))
                .drop('is_train')
                .to_pandas()
                .set_index('subset')
                .rename(columns=lambda x: x.replace('_loglik', ''))
                .style
                .format('{:.2f}')
            ),
            caption='Mean log likelihood per event',
        ),
        loglik_relative=table(
            styler=(
                pl.read_csv(RESULTS_DIR / 'overall/loglik_relative.csv')
                .with_columns(subset=pl.when(pl.col('is_train'))
                              .then(pl.lit('train')).otherwise(pl.lit('validation')))
                .drop('is_train')
                .to_pandas()
                .set_index('subset')
                .rename(columns=lambda x: x.replace('_loglik', ''))
                .style
                .format('{:.2%}')
            ),
            caption='Mean log likelihood per event relative to constant baseline',
        ),
        qq_val=image(
            'overall/qq_val.png',
            'QQ plots of normalised residuals',
        ),
        ts_val2=image(
            'overall/val2.png',
            '2025-10-10',
        ),
        ts_val3=image(
            'overall/val3.png',
            '2025-10-10 zoomed in',
        ),
        all_model_results='\n\n'.join(map(per_model_results, [
            'constant',
            'rbf',
            'hawkes',
            'rbf_hawkes',
            'pl_hawkes',
        ])),
    )

    with open(README_PATH, 'w') as f:
        f.write(rendered)


if __name__ == '__main__':
    generate_readme()


# %%
