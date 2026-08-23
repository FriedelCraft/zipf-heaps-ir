
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import r2_score, mean_squared_error

from .utils import get_logger

logger = get_logger(__name__)


def generate_checkpoint_targets(
    first_checkpoint_tokens: int,
    checkpoints_per_decade: float,
    max_checkpoints: int = 200,
) -> Iterator[int]:
    
    if first_checkpoint_tokens < 1:
        raise ValueError("first_checkpoint_tokens must be >= 1")
    if checkpoints_per_decade <= 0:
        raise ValueError("checkpoints_per_decade must be > 0")

    growth_factor = 10 ** (1.0 / checkpoints_per_decade)
    target = float(first_checkpoint_tokens)
    emitted = 0
    last_emitted = 0
    while emitted < max_checkpoints:
        target_int = round(target)
        if target_int > last_emitted:
            yield target_int
            last_emitted = target_int
            emitted += 1
        target *= growth_factor


@dataclass
class HeapsFitResult:
    K: float
    beta: float
    intercept_log_k: float   # = log(K); kept for transparency/debugging
    r_squared: float
    rmse: float
    n_observations: int

    def to_metrics_dict(self) -> dict:
        """Column names matching this project's requested metrics CSV schema."""
        return {
            "K": self.K,
            "beta": self.beta,
            "R_squared": self.r_squared,
            "RMSE": self.rmse,
            "intercept_log_K": self.intercept_log_k,
            "number_of_checkpoints_used_for_fit": self.n_observations,
        }


def fit_heaps_law(
    token_counts: np.ndarray,
    vocabulary_sizes: np.ndarray,
    min_observations: int = 5,
) -> HeapsFitResult | None:
    """
    Fit V(N) = K * N^beta via ordinary least squares on
    log(V) = log(K) + beta * log(N), using scipy.stats.linregress +
    sklearn r2_score/mean_squared_error — the same approach used by the
    companion zipf_analysis_project's Zipf fit, for consistency.

    Returns None if there are too few checkpoints to fit reliably rather
    than raising.
    """
    n = len(token_counts)
    if n < min_observations:
        logger.warning(
            f"Only {n} Heaps checkpoint(s) available for fitting "
            f"(< min_observations={min_observations}); skipping fit."
        )
        return None

    log_n = np.log(token_counts.astype(float))
    log_v = np.log(vocabulary_sizes.astype(float))

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_n, log_v)
    predicted = intercept + slope * log_n
    r2 = r2_score(log_v, predicted)
    rmse = float(np.sqrt(mean_squared_error(log_v, predicted)))

    return HeapsFitResult(
        K=float(np.exp(intercept)),
        beta=float(slope),
        intercept_log_k=float(intercept),
        r_squared=float(r2),
        rmse=rmse,
        n_observations=n,
    )



def compute_heaps_residuals(growth_df: pd.DataFrame, fit: HeapsFitResult) -> pd.DataFrame:
    """
    residual = observed log(vocabulary_size) - predicted log(vocabulary_size),
    using the fitted line. Operates only on the (small) checkpoint table.
    """
    if growth_df.empty:
        return pd.DataFrame(columns=[
            "token_count", "vocabulary_size", "log_token_count",
            "log_vocabulary_size", "predicted_vocabulary_size",
            "predicted_log_vocabulary_size", "residual",
        ])

    df = growth_df.copy()
    df["log_token_count"] = np.log(df["token_count"].astype(float))
    df["log_vocabulary_size"] = np.log(df["vocabulary_size"].astype(float))
    df["predicted_log_vocabulary_size"] = fit.intercept_log_k + fit.beta * df["log_token_count"]
    df["predicted_vocabulary_size"] = np.exp(df["predicted_log_vocabulary_size"])
    df["residual"] = df["log_vocabulary_size"] - df["predicted_log_vocabulary_size"]
    return df[[
        "token_count", "vocabulary_size", "log_token_count", "log_vocabulary_size",
        "predicted_vocabulary_size", "predicted_log_vocabulary_size", "residual",
    ]]


def generate_heaps_summary(fit: HeapsFitResult | None, growth_df: pd.DataFrame) -> list[str]:
    """
    Produce a neutral, structured plain-text summary that explicitly
    separates:

        EMPIRICAL RESULT     — what was actually observed (checkpoints)
        FITTED MODEL VALUES  — the regression line (an approximation)
        QUALITATIVE IMPLICATION — plain-language interpretation of beta

    Never uses beta as a multiplier or input to a cost figure; that
    connection, if made at all, belongs in the separate
    cost_infrastructure_project as narrative, not here as arithmetic.
    """
    lines: list[str] = []

    lines.append("EMPIRICAL RESULT:")
    if growth_df.empty:
        lines.append("  No vocabulary growth checkpoints were recorded.")
    else:
        first = growth_df.iloc[0]
        last = growth_df.iloc[-1]
        lines.append(
            f"  Observed vocabulary growth across {len(growth_df)} checkpoints, "
            f"from {int(first['token_count']):,} tokens "
            f"({int(first['vocabulary_size']):,} unique tokens observed) to "
            f"{int(last['token_count']):,} tokens "
            f"({int(last['vocabulary_size']):,} unique tokens observed)."
        )
        lines.append(
            "  These are direct measurements taken during a single streaming "
            "pass over the corpus, not model output."
        )

    lines.append("")
    lines.append("FITTED MODEL VALUES:")
    if fit is None:
        lines.append(
            "  Heaps' Law fit could not be computed (insufficient checkpoints "
            "at or above the configured min_tokens_for_fit threshold)."
        )
    else:
        lines.append(
            f"  V(N) = K * N^beta, fitted via log-log ordinary least squares: "
            f"K = {fit.K:.4f}, beta = {fit.beta:.4f}, "
            f"R^2 = {fit.r_squared:.4f}, RMSE (log scale) = {fit.rmse:.4f}, "
            f"based on {fit.n_observations} checkpoint(s)."
        )
        lines.append(
            "  This is a FITTED APPROXIMATION, not an exact description of the "
            "observed data — inspect R^2 and the residual plot for how well it "
            "actually tracks the checkpoints before treating it as reliable."
        )

        lines.append("")
        lines.append("QUALITATIVE IMPLICATION:")
        if fit.beta < 1.0:
            lines.append(
                f"  The estimated exponent (beta = {fit.beta:.4f}) is below 1, "
                f"consistent with SUBLINEAR vocabulary growth: as more tokens "
                f"are processed, vocabulary size grows more slowly than the "
                f"token count itself. This is the typical Heaps' Law pattern — "
                f"new distinct words become progressively rarer as more text is "
                f"seen, though growth has not necessarily plateaued within the "
                f"observed range."
            )
        else:
            lines.append(
                f"  The estimated exponent (beta = {fit.beta:.4f}) is at or "
                f"above 1, meaning vocabulary size is growing roughly as fast "
                f"as, or faster than, the token count itself within the "
                f"observed range — i.e. new tokens are still being introduced "
                f"at a largely undiminished rate."
            )
        lines.append(
            "  Practically, this suggests that continued data acquisition may "
            "still introduce new vocabulary and improve long-tail coverage — "
            "the growth curve has not been observed to plateau within this "
            "corpus. This is a qualitative reading of the trend, not a "
            "quantitative prediction of how much new vocabulary future text "
            "will contain."
        )

    lines.append("")
    lines.append("NOT DONE HERE / NOT ALLOWED:")
    lines.append(
        "  beta is NOT used as a multiplier or input in any cost or "
        "infrastructure calculation in this project. Any connection between "
        "vocabulary growth and infrastructure cost is qualitative narrative "
        "only, made separately in the cost_infrastructure_project's "
        "integration summary — never arithmetic here or there."
    )

    return lines

def build_cost_project_compatible_row(
    fit: HeapsFitResult | None, growth_df: pd.DataFrame, corpus_version: str,
) -> pd.DataFrame:
    
    if growth_df.empty:
        token_count = None
        vocabulary_size = None
    else:
        last = growth_df.iloc[-1]
        token_count = int(last["token_count"])
        vocabulary_size = int(last["vocabulary_size"])

    row = {
        "corpus_version": corpus_version,
        "token_count": token_count,
        "vocabulary_size": vocabulary_size,
        "K": fit.K if fit is not None else None,
        "beta": fit.beta if fit is not None else None,
        "r_squared": fit.r_squared if fit is not None else None,
        "rmse": fit.rmse if fit is not None else None,
    }
    return pd.DataFrame([row])


def run_heaps_analysis(
    checkpoints: list[dict] | pd.DataFrame,
    output_dir: Path,
    min_tokens_for_fit: int = 100_000,
    min_observations_for_fit: int = 5,
) -> dict:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(checkpoints, list):
        growth_df = pd.DataFrame(checkpoints, columns=["token_count", "vocabulary_size"])
    else:
        growth_df = checkpoints.copy()

    if not growth_df.empty:
        growth_df = (
            growth_df.sort_values("token_count")
            .drop_duplicates(subset="token_count")
            .reset_index(drop=True)
        )

    growth_df.to_csv(output_dir / "heaps_vocabulary_growth.csv", index=False)

    fit_df = (
        growth_df[growth_df["token_count"] >= min_tokens_for_fit].reset_index(drop=True)
        if not growth_df.empty else growth_df
    )

    fit = fit_heaps_law(
        fit_df["token_count"].values if not fit_df.empty else np.array([]),
        fit_df["vocabulary_size"].values if not fit_df.empty else np.array([]),
        min_observations=min_observations_for_fit,
    )

    metrics_columns = [
        "K", "beta", "R_squared", "RMSE", "intercept_log_K",
        "number_of_checkpoints", "number_of_checkpoints_used_for_fit",
    ]
    if fit is not None:
        metrics_row = fit.to_metrics_dict()
        metrics_row["number_of_checkpoints"] = len(growth_df)
        pd.DataFrame([metrics_row])[metrics_columns].to_csv(
            output_dir / "heaps_metrics.csv", index=False
        )
        residuals_df = compute_heaps_residuals(fit_df, fit)
    else:
        pd.DataFrame(
            [{"K": None, "beta": None, "R_squared": None, "RMSE": None,
              "intercept_log_K": None, "number_of_checkpoints": len(growth_df),
              "number_of_checkpoints_used_for_fit": len(fit_df)}]
        )[metrics_columns].to_csv(output_dir / "heaps_metrics.csv", index=False)
        residuals_df = pd.DataFrame(
            columns=["token_count", "vocabulary_size", "log_token_count",
                     "log_vocabulary_size", "predicted_vocabulary_size",
                     "predicted_log_vocabulary_size", "residual"]
        )

    residuals_df.to_csv(output_dir / "heaps_residuals.csv", index=False)

    summary_lines = generate_heaps_summary(fit, growth_df)
    with open(output_dir / "heaps_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    for line in summary_lines:
        logger.info(line)

    return {
        "growth_df": growth_df,
        "fit_df": fit_df,
        "fit": fit,
        "residuals_df": residuals_df,
        "summary_lines": summary_lines,
    }
