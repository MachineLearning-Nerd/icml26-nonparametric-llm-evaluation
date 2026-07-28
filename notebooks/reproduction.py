import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Nonparametric LLM evaluation: an evidence-first reproduction

    ![Full-scale Chatbot Arena interval-width result](https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-rHndxbqWyh-nonparametric-llm-eval/master/reports/reproduction/images/claim6_headline.png)

    **Headline:** at the paper's literal `n=32,980`, `K=20` Arena scale,
    plugin confidence intervals are only 1.4–4.0% as wide as the
    cross-fitted EIF intervals. The disabled-EIF control has ratio 1.0 and
    is rejected.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What is being estimated?

    Preference logs show only selected model pairs, so a naive plugin
    estimator treats learned outcome probabilities as if they were known.
    DMLRank instead targets

    \[
    \theta = \mathbb{E}\{F(\mu(X))\},
    \]

    where `F` can be Borda, a Bradley–Terry projection, or Rank
    Centrality. Its efficient-influence-function correction propagates
    uncertainty from selectively observed labels into the final ranking.
    """)
    return


@app.cell
def _(mo):
    gars = {
        "Borda": {
            "plugin_width": 0.0021965099510308592,
            "debiased_width": 0.054907107174906614,
            "ratio": 0.04018219206377316,
        },
        "Bradley–Terry": {
            "plugin_width": 0.006613367651389132,
            "debiased_width": 0.28842007813300474,
            "ratio": 0.023531251149869956,
        },
        "Rank Centrality": {
            "plugin_width": 0.00011351841708914952,
            "debiased_width": 0.007963063781496369,
            "ratio": 0.014133656373289397,
        },
    }
    choice = mo.ui.dropdown(
        options=list(gars),
        value="Borda",
        label="Inspect a GARS",
    )
    choice
    return choice, gars


@app.cell
def _(choice, gars, mo):
    selected = gars[choice.value]
    mo.md(
        f"""
        ### {choice.value}

        | Quantity | Precomputed value |
        |---|---:|
        | Plugin median simultaneous width | `{selected["plugin_width"]:.6f}` |
        | Debiased EIF median width | `{selected["debiased_width"]:.6f}` |
        | Plugin / EIF | `{100 * selected["ratio"]:.2f}%` |
        | EIF-disabled control | `100.00%` — rejected |

        The formal contract was fixed before the run: plugin / EIF must be
        below 50% for **every** GARS. The independent checker recomputes the
        median from all 20 per-model widths.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Claim-by-claim outcome

    | Claim | Evidence | Verdict | Confidence |
    |---|---|---|---|
    | 1 · GARS special cases | Algebraic residuals ≤ `4.44e-16`; wrong symmetrization separated | VERIFIED | HIGH |
    | 2 · Efficient cross-fitted EIF | Pathwise error `2.78e-17`; omitted IPW rejected; independent asymptotic derivation | VERIFIED | MEDIUM |
    | 3 · Table 1 numbers | Two source interpretations + metric audit + falsification route | BLOCKED | LOW |
    | 4 · A-optimal formula | KKT derivation, budget `4.44e-16`, SLSQP `5.86e-8` | VERIFIED | HIGH |
    | 5 · Table 2 numbers | Oracle, pilot, acquired-data, and falsification routes | BLOCKED | LOW |
    | 6 · Full Arena intervals | Literal scale, three GARS, independent checker and corruption | VERIFIED | MEDIUM |

    `BLOCKED` is deliberate: the exact Table 1/2 author seeds, coefficient
    draws, metric conventions, and raw finite-run realizations are not
    public. A nearby clean-room mismatch is not an exact falsification.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproducible path

    Every experiment inherited the same command:

    ```text
    uv sync --frozen && uv run python repro/run_all.py
    ```

    The lockfile SHA-256 is
    `95f8cd0826a479d495738fba954dc4e4a8712ce13614c21e2769667d3714c165`.
    The final evidence run used Hugging Face `cpu-upgrade`, exposed 64
    logical CPUs, and completed in 527.503 seconds.

    The notebook intentionally embeds completed results. It does not ask
    Molab users to rerun expensive inference. See the
    [illustrated report](https://github.com/MachineLearning-Nerd/icml26-repro-rHndxbqWyh-nonparametric-llm-eval/blob/master/reports/reproduction/report.md)
    for implementation details, raw-evidence links, source ambiguity, and
    experiment branches.
    """)
    return


if __name__ == "__main__":
    app.run()
