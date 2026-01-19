# %%

import json
from pathlib import Path
import datetime

from jinja2 import Template
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


def table(df: pd.DataFrame, caption: str) -> str:
    global _table_counter
    count = str(_table_counter)
    _table_counter += 1

    template = Template("""\
<div align="center">

{{ md_table }}

Table {{ count }}: {{ caption }}
</div>
""")

    return template.render(
        count=count,
        md_table=df.to_markdown(tablefmt='github'),
        caption=caption,
    )


# %%

def convert_formats(df: pd.DataFrame, formatter: str | dict[str, str]) -> pd.DataFrame:
    if isinstance(formatter, dict):
        for col, fmt in formatter.items():
            df[col] = df[col].map(fmt.format)
        return df
    if isinstance(formatter, str):
        for col in df.columns:
            df[col] = df[col].map(formatter.format)
        return df
    assert False, type(formatter)


def per_model_results(prefix: str, formula: str) -> str:
    template = Template(
        """\
#### {{ prefix }}: {{ n_params }} param(s), training duration: {{ elapsed }}s ({{ seconds_per_eval }}s/eval)

$$
\\begin{align}
{{ formula }}
\\end{align}
$$

<details>
<summary>Parameters</summary>
{{ params }}
</details>
<details>
<summary>Diagnostics</summary>
{{ diagnostics }}
</details>
{% if prefix != "constant" %}
<details>
<summary>Inverse Hessian</summary>
{{ inv_hessian }}
</details>
<details>
<summary>Convergence</summary>
{{ convergence }}
</details>
{% endif %}
{% if prefix in ["rbf", "rbf_hawkes"] %}
<details>
<summary>Seasonal Basis Functions</summary>
{{ bases }}
</details>
{% endif %}
<details>
<summary>Predictions</summary>
{{ predictions }}
</details>
""")
    with open(RESULTS_DIR / f'{prefix}/convergence_stats.json') as f:
        conv_stats = json.load(f)
    n_evals = conv_stats['n_iter'] + conv_stats['n_linesearch']
    elapsed = conv_stats['elapsed_seconds']
    seconds_per_eval = elapsed / n_evals

    render_kwargs = dict(
        prefix=prefix,
        n_params=conv_stats['n_params'],
        elapsed='{:.2f}'.format(elapsed),
        seconds_per_eval='{:.2f}'.format(seconds_per_eval),
        formula=formula,

        params=table(
            df=(
                pd.read_csv(RESULTS_DIR / f'{prefix}/params.csv',
                            index_col=0)
            ),
            caption=f'{prefix} parameters'
        ),
        diagnostics=table(
            df=(
                pd.read_csv(RESULTS_DIR / f'{prefix}/diagnostics.csv',
                            index_col=0)
                .pipe(
                    convert_formats,
                    dict(
                        mean='{:.4f}',
                        hess_se='{:.4f}',
                        robust_se='{:.4f}',
                        z_score='{:.2f}',
                        p_value='{:.2%}',
                        se_ratio='{:.4f}',
                    ),
                )
            ),
            caption=f'{prefix} diagnostics'
        ),
    )
    if prefix != 'constant':
        render_kwargs['inv_hessian'] = image(f'{prefix}/inv_hessian.png',
                                             f'{prefix} inverse Hessian')
        render_kwargs['convergence'] = image(f'{prefix}/optim_outputs.png',
                                             f'{prefix} convergence')
    if prefix.startswith('rbf'):
        render_kwargs['bases'] = image(f'{prefix}/bases.png',
                                       f'{prefix} bases')
    render_kwargs['predictions'] = image(f'{prefix}/counts.png',
                                         f'{prefix} predictions')
    return template.render(**render_kwargs)


def generate_all_model_results() -> str:
    per_model: list[str] = []
    formulae = dict(
        constant='\\lambda(t) = \\mu',
        rbf='\\lambda(t) = \\text{softplus}( a + \\sum_k w_k \\phi_k(t) )',
        hawkes=(
            '& \\lambda(t) = \\mu + \\sum_{t_i < t} \\Phi(t - t_i) \\\\'
            '\n& \\Phi(t) = g \\omega e^{-\\omega t}'
        ),
        rbf_hawkes=(
            '& \\lambda(t) = \\text{softplus}( a + \\sum_k w_k \\phi_k(t) ) + \\sum_{t_i < t} \\Phi(t - t_i) \\\\'
            '\n& \\Phi(t) = g \\omega e^{-\\omega t}'
        ),
        pl_hawkes=(
            '& \\lambda(t) = \\mu + \\sum_{t_i < t} \\Phi(t - t_i) \\\\'
            '\n& \\Phi(t) = \\frac {g \\omega \\beta} {( 1 + \\omega t) ^ {1 + \\beta}}'
        ),
    )
    for prefix, formula in formulae.items():
        result = per_model_results(prefix=prefix, formula=formula)
        per_model.append(result)
    return '\n\n'.join(per_model)


def generate_readme():
    template = Template(TEMPLATE_PATH.read_text())

    loglik_mean = (
        pl.read_csv(RESULTS_DIR / 'overall/loglik_mean.csv')
        .with_columns(subset=pl.when(pl.col('is_train'))
                      .then(pl.lit('train')).otherwise(pl.lit('validation')))
        .drop('is_train')
    )
    loglik_diff = (
        loglik_mean
        .with_columns(
            (pl.selectors.ends_with('_loglik') - pl.col('constant_loglik'))
            / pl.col('constant_loglik')
        )
    )

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
            'Inverse Hessian: high absolute values suggest potential'
            + ' identifiability issues',
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
            ),
            caption='Raw data',
        ),
        data_stats=table(
            df=(
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
                .pipe(convert_formats, '{:,}')
            ),
            caption='Sample counts'
        ),
        loglik_mean=table(
            df=(
                loglik_mean
                .to_pandas()
                .set_index('subset')
                .rename(columns=lambda x: x.replace('_loglik', ''))
                .pipe(convert_formats, '{:.2f}')
            ),
            caption='Mean log likelihood per timestamp',
        ),
        loglik_diff=table(
            df=(
                loglik_diff
                .to_pandas()
                .set_index('subset')
                .rename(columns=lambda x: x.replace('_loglik', ''))
                .pipe(convert_formats, '{:.2f}')
            ),
            caption='Mean log likelihood per timestamp minus constant baseline',
        ),
        qq_val=image(
            'overall/qq_val.png',
            'QQ plots of Pearson residuals of binned counts',
        ),
        ts_val2=image(
            'overall/val2.png',
            '2025-10-10',
        ),
        ts_val3=image(
            'overall/val3.png',
            '2025-10-10 zoomed in',
        ),
        all_model_results=generate_all_model_results(),
    )

    with open(README_PATH, 'w') as f:
        f.write(rendered)


if __name__ == '__main__':
    generate_readme()


# %%
