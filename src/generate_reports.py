# %%

from pathlib import Path

from jinja2 import Template
from pandas.io.formats.style import Styler
import pandas as pd
import polars as pl


PROJ_ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJ_ROOT_DIR / 'results'
TEMPLATE_PATH = PROJ_ROOT_DIR / 'README.template.md'
README_PATH = PROJ_ROOT_DIR / 'README.md'


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
                    'selector': 'caption',
                    'props': [
                        ('caption-side', 'bottom'),
                    ],
                },
                {
                    'selector': '',
                    'props': [
                        ('margin-left', 'auto'),
                        ('margin-right', 'auto'),
                    ],
                },
            ],
            overwrite=False,
        )
        # set uuid to ensure determinism for version control
        .to_html(table_uuid=str(count))
    )


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
                .background_gradient(axis=1)
                .format('{:.2f}')
            ),
            caption='Mean log likelihood per sample',
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
                .background_gradient(axis=1)
                .format('{:.2%}')
            ),
            caption='Mean log likelihood per sample relative to constant baseline',
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
    )

    with open(README_PATH, 'w') as f:
        f.write(rendered)


if __name__ == '__main__':
    generate_readme()


# %%
