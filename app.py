import math
from typing import List, Tuple

import pandas as pd
import streamlit as st


# -------------------------------
# Helpers
# -------------------------------
def parse_fractional_odds(token: str) -> float:
    """
    Convert fractional odds like '8/1' or '5/2' to decimal odds.
    Decimal odds = 1 + (numerator / denominator)
    """
    num, den = token.split("/", 1)
    return 1.0 + (float(num) / float(den))


def parse_numeric_token(token: str, use_comma_decimal: bool) -> float:
    """
    Parse a token that might be:
      - decimal with dot or comma (e.g. '3.5' or '3,5')
      - fractional odds 'a/b'
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty token")

    if "/" in token:
        return parse_fractional_odds(token)

    if use_comma_decimal:
        token = token.replace(",", ".")
    return float(token)


def parse_pasted_list(raw: str, use_comma_decimal: bool) -> List[float]:
    """
    Split by comma, semicolon, whitespace. Keep tokens with numbers or a '/'.
    """
    # Normalize separators
    tmp = raw.replace(";", " ").replace(",", " ").replace("\n", " ").replace("\t", " ")
    tokens = [t for t in tmp.split(" ") if t.strip() != ""]
    if not tokens:
        return []

    values = []
    for tok in tokens:
        values.append(parse_numeric_token(tok, use_comma_decimal))
    return values


def implied_probabilities_from_decimal_odds(odds: List[float]) -> Tuple[List[float], float]:
    """
    Convert decimal odds to implied probabilities, then return:
      - raw implied probs p_i = 1/odds_i
      - overround (sum(p_i) - 1.0)
    """
    raw = [1.0 / o for o in odds]
    overround = sum(raw) - 1.0
    return raw, overround


def normalize(probs: List[float]) -> List[float]:
    s = sum(probs)
    if s <= 0:
        raise ValueError("Sum of probabilities must be positive.")
    return [p / s for p in probs]


def shannon_entropy_bits(probs: List[float]) -> float:
    # H = -sum p log2 p, with convention 0 log 0 = 0
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log2(p)
    return h


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Race Entropy", page_icon="🐎", layout="centered")

st.title("🐎 Race Entropy (Shannon)")
st.write(
    "Paste horse **cotes** (decimal odds like `3.5`, or fractional like `8/1`) "
    "or probabilities. We’ll normalize them and compute Shannon entropy."
)

with st.sidebar:
    st.header("Input settings")
    input_mode = st.radio(
        "How do you want to enter values?",
        ["Paste list", "Interactive fields"],
        index=0,
    )
    value_type = st.radio(
        "What are you entering?",
        ["Odds (decimal/fractional)", "Probabilities (0–1)"],
        index=0,
        help="If you choose odds, we convert to implied probabilities and normalize.",
    )
    use_comma = st.checkbox(
        "Use comma as decimal separator (e.g., 3,5)", value=True
    )
    normalize_choice = st.radio(
        "Normalization for entropy",
        ["Auto-normalize (recommended)", "Do not normalize (use as-is)"],
        index=0,
        help="When using odds, bookmaker margin (overround) makes summed implied probabilities > 1. "
             "Auto-normalize removes that margin. If you provide probabilities that do not sum to 1, "
             "auto-normalize fixes that too.",
    )

    show_table = st.checkbox("Show table", value=True)
    show_chart = st.checkbox("Show bar chart", value=True)

# Collect inputs
values: List[float] = []

if input_mode == "Paste list":
    example = "3.5 5 7 10 15"
    placeholder = example if value_type.startswith("Odds") else "0.28 0.20 0.16 0.11 0.07 ..."
    raw = st.text_area(
        "Enter values (separated by spaces, commas, or semicolons).",
        placeholder=placeholder,
        height=120,
    )
    if raw.strip():
        try:
            values = parse_pasted_list(raw, use_comma_decimal=use_comma)
        except Exception as e:
            st.error(f"Could not parse input: {e}")

else:
    n = st.number_input("Number of horses", min_value=1, max_value=20, value=8, step=1)
    values = []
    cols_per_row = 4
    for i in range(n):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        idx_in_row = i % cols_per_row
        default = 6.0 if value_type.startswith("Odds") else round(1.0 / n, 4)
        v = cols[idx_in_row].number_input(
            f"Horse {i+1}",
            value=float(default),
            min_value=0.0001 if value_type.startswith("Odds") else 0.0,
            step=0.05 if value_type.startswith("Odds") else 0.01,
            key=f"horse_{i}",
        )
        values.append(float(v))

# Validate and compute
if not values:
    st.info("Enter at least one value to begin.")
    st.stop()

# Disallow invalid odds
if value_type.startswith("Odds"):
    bad = [o for o in values if o < 1.0]
    if bad:
        st.error("All decimal odds must be ≥ 1. Fractional odds like '8/1' are OK (they convert to ≥ 2).")
        st.stop()

try:
    if value_type.startswith("Odds"):
        # From odds to implied probabilities
        raw_probs, overround = implied_probabilities_from_decimal_odds(values)
        chosen_probs = raw_probs[:]
        source_note = "Implied from odds (p = 1/odds)."

    else:
        # Direct probabilities
        raw_probs = values[:]
        overround = None  # not applicable
        chosen_probs = raw_probs[:]
        source_note = "Provided directly as probabilities."

    # Normalize if requested
    if normalize_choice.startswith("Auto-normalize"):
        probs = normalize(chosen_probs)
        norm_note = "Normalized to sum = 1."
    else:
        probs = chosen_probs
        norm_note = "No normalization applied."
except Exception as e:
    st.error(f"Computation error: {e}")
    st.stop()

# Entropy measures
H_bits = shannon_entropy_bits(probs)
H_nats = H_bits * math.log(2.0)  # convert bits -> nats
effective_n = 2 ** H_bits        # “effective number of equally likely horses”

# Metrics header
st.subheader("Results")

m1, m2, m3 = st.columns(3)
m1.metric("Shannon entropy (bits)", f"{H_bits:.4f}")
m2.metric("Shannon entropy (nats)", f"{H_nats:.4f}")
m3.metric("Effective # of horses (2^H)", f"{effective_n:.3f}")

# Overround / margin (only when odds provided)
if value_type.startswith("Odds"):
    raw_sum = sum(1.0 / o for o in values)
    margin_pct = (raw_sum - 1.0) * 100.0
    st.caption(
        f"**Implied prob sum (raw)**: {raw_sum:.4f}  |  **Overround**: {margin_pct:+.2f}%  "
        f"({source_note} {norm_note})"
    )
else:
    s = sum(values)
    st.caption(
        f"**Sum of provided probabilities**: {s:.4f}  ({source_note} {norm_note})"
    )

# Table
df = None
if show_table:
    records = []
    for i, v in enumerate(values, start=1):
        row = {"Horse": i}
        if value_type.startswith("Odds"):
            row["Input (odds)"] = v
            row["Implied p (raw)"] = 1.0 / v
        else:
            row["Input p (raw)"] = v
        row["p used for entropy"] = probs[i - 1]
        row["p used (%)"] = 100.0 * probs[i - 1]
        records.append(row)
    df = pd.DataFrame.from_records(records)
    st.dataframe(df, use_container_width=True)

# Chart
if show_chart:
    chart_df = pd.DataFrame(
        {"Horse": [f"{i}" for i in range(1, len(probs) + 1)],
         "Probability (%)": [100.0 * p for p in probs]}
    ).set_index("Horse")
    st.bar_chart(chart_df, use_container_width=True)

# Download CSV
if df is None:
    # Build a minimal DF if user hid the table
    df = pd.DataFrame({
        "Horse": list(range(1, len(probs) + 1)),
        "p used for entropy": probs,
        "p used (%)": [100.0 * p for p in probs],
    })
csv = df.to_csv(index=False)
st.download_button(
    "⬇️ Download table as CSV",
    data=csv,
    file_name="race_entropy_table.csv",
    mime="text/csv",
)

# Footer
st.write("---")
st.caption(
    "Notes:\n"
    "- Decimal **cotes** (e.g., 3.5) convert to implied p = 1/odds. Fractional odds (e.g., 8/1) are supported and convert to decimal automatically.\n"
    "- Bookmaker margin (overround) makes raw implied probabilities sum to > 1. **Auto-normalize** removes this margin for entropy.\n"
    "- Shannon entropy H (bits) uses log base 2. The effective number of horses is 2^H."
)

if __name__ == "__main__":
    pass
