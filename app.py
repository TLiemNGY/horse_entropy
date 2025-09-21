import math
from typing import List, Tuple

import pandas as pd
import streamlit as st


# -------------------------------
# Helpers
# -------------------------------
def parse_fractional_odds(token: str) -> float:
    num, den = token.split("/", 1)
    return 1.0 + (float(num) / float(den))

def parse_numeric_token(token: str, use_comma_decimal: bool) -> float:
    token = token.strip()
    if not token:
        raise ValueError("Empty token")
    if "/" in token:
        return parse_fractional_odds(token)
    if use_comma_decimal:
        token = token.replace(",", ".")
    return float(token)

def parse_pasted_list(raw: str, use_comma_decimal: bool) -> List[float]:
    tmp = raw.replace(";", " ").replace(",", " ").replace("\n", " ").replace("\t", " ")
    tokens = [t for t in tmp.split(" ") if t.strip() != ""]
    if not tokens:
        return []
    return [parse_numeric_token(tok, use_comma_decimal) for tok in tokens]

def implied_probabilities_from_decimal_odds(odds: List[float]) -> Tuple[List[float], float]:
    raw = [1.0 / o for o in odds]
    return raw, (sum(raw) - 1.0)

def normalize(probs: List[float]) -> List[float]:
    s = sum(probs)
    if s <= 0:
        raise ValueError("Sum of probabilities must be positive.")
    return [p / s for p in probs]

def shannon_entropy_nats(probs: List[float]) -> float:
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p)
    return h


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Race Entropy", page_icon="🐎", layout="centered")

st.title("🐎 Race Entropy (Shannon, nats)")
st.write(
    "Collez les **cotes** (décimales `3.5` ou fractionnelles `8/1`) "
    "ou des probabilités. On normalise puis on calcule l’entropie de Shannon en **nats**."
)

with st.sidebar:
    st.header("Paramètres")
    input_mode = st.radio("Mode de saisie", ["Liste collée", "Champs interactifs"], index=0)
    value_type = st.radio(
        "Type de valeurs",
        ["Cotes (décimales/fractionnelles)", "Probabilités (0–1)"],
        index=0,
        help="Avec des cotes, on passe en probabilités implicites puis on normalise."
    )
    use_comma = st.checkbox("Virgule comme séparateur décimal (ex. 3,5)", value=True)
    normalize_choice = st.radio(
        "Normalisation pour l’entropie",
        ["Auto-normaliser (recommandé)", "Ne pas normaliser"],
        index=0,
        help=("Avec des cotes, la marge du bookmaker fait que la somme des probabilités dépasse 1. "
              "L’auto-normalisation enlève cette marge. Si vos probabilités ne somment pas à 1, "
              "on les renormalise aussi.")
    )
    show_table = st.checkbox("Afficher le tableau", value=True)
    show_chart = st.checkbox("Afficher l’histogramme", value=True)

# Saisie
values: List[float] = []

if input_mode == "Liste collée":
    example = "3.5 5 7 10 15"
    placeholder = example if value_type.startswith("Cotes") else "0.28 0.20 0.16 0.11 0.07 ..."
    raw = st.text_area("Entrez les valeurs (séparées par espaces, virgules ou points-virgules).",
                       placeholder=placeholder, height=120)
    if raw.strip():
        try:
            values = parse_pasted_list(raw, use_comma_decimal=use_comma)
        except Exception as e:
            st.error(f"Impossible de parser l’entrée : {e}")
else:
    n = st.number_input("Nombre de chevaux", min_value=1, max_value=20, value=8, step=1)
    values = []
    cols_per_row = 4
    for i in range(n):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        idx = i % cols_per_row
        default = 6.0 if value_type.startswith("Cotes") else round(1.0 / n, 4)
        v = cols[idx].number_input(
            f"Cheval {i+1}",
            value=float(default),
            min_value=0.0001 if value_type.startswith("Cotes") else 0.0,
            step=0.05 if value_type.startswith("Cotes") else 0.01,
            key=f"horse_{i}",
        )
        values.append(float(v))

if not values:
    st.info("Saisissez au moins une valeur pour commencer.")
    st.stop()

if value_type.startswith("Cotes"):
    bad = [o for o in values if o < 1.0]
    if bad:
        st.error("Toutes les cotes décimales doivent être ≥ 1. Les cotes fractionnelles (ex. '8/1') sont OK.")
        st.stop()

# Probabilités utilisées
try:
    if value_type.startswith("Cotes"):
        raw_probs, overround = implied_probabilities_from_decimal_odds(values)
        chosen_probs = raw_probs[:]
        source_note = "Déduites des cotes (p = 1/cote)."
    else:
        raw_probs = values[:]
        overround = None
        chosen_probs = raw_probs[:]
        source_note = "Fournies directement (probabilités)."

    if normalize_choice.startswith("Auto-normaliser"):
        probs = normalize(chosen_probs)
        norm_note = "Normalisées pour sommer à 1."
    else:
        probs = chosen_probs
        norm_note = "Sans normalisation."
except Exception as e:
    st.error(f"Erreur de calcul : {e}")
    st.stop()

# Entropie + effectif + classement fav/2e
H_nats = shannon_entropy_nats(probs)
effective_n = math.exp(H_nats)
nb_partants = len(probs)

if len(probs) >= 2:
    ranked = sorted(list(enumerate(probs, start=1)), key=lambda x: x[1], reverse=True)
    (fav_idx, p1), (sec_idx, p2) = ranked[0], ranked[1]
    fav_margin = p1 - p2
else:
    fav_idx = sec_idx = None
    p1 = p2 = None
    fav_margin = None

# -------------------------------
# RÉSULTATS — Ordre et présentation demandés
# -------------------------------
st.subheader("Résultats")

# 1) Entropie (grosse métrique) + 2) Marge (grosse métrique)
c1, c2 = st.columns(2)
c1.metric("Entropie de Shannon (nats)", f"{H_nats:.4f}")
if fav_margin is not None:
    c2.metric("Marge fav vs 2e (p1 − p2)", f"{fav_margin:.4f}")
else:
    c2.metric("Marge fav vs 2e (p1 − p2)", "N/A")

# Bandeau d’incertitude (optionnel)
if H_nats > 2.36:
    st.info("🔵 Course **très incertaine** (H > 2.36 nats).")

# Sous-ligne en plus petit : nombre réel puis effectif
st.caption(f"**Nombre de partants (réel)** : {nb_partants}  |  **Nombre effectif de chevaux (exp(H))** : {effective_n:.3f}")

# Détail favori / 2e en petit aussi
if fav_margin is not None:
    st.caption(
        f"Favori : Cheval {fav_idx} ({100*p1:.2f}%)  |  2e : Cheval {sec_idx} ({100*p2:.2f}%)"
    )

# -------------------------------
# Paragraphes explicatifs (un par métrique)
# -------------------------------
st.markdown("### Interprétation des métriques")
st.markdown(
    "**Entropie de Shannon (nats)** — Mesure l’incertitude globale de la course : "
    "H = −∑ p·ln p. Plus H est élevé, plus la répartition des chances est homogène. "
    "Pour n partants équiprobables, H = ln(n). Un seuil pratique : au-delà de 2.36 nats, la course est considérée **très incertaine**."
)
st.markdown(
    "**Marge favori vs 2e (p1 − p2)** — Différence entre la probabilité du favori et celle du 2e, "
    "calculée sur les probabilités **utilisées** (après normalisation si activée). "
    "Plus cette marge est grande, plus le favori se détache. Proche de 0 ⇒ lutte serrée en tête."
)
st.markdown(
    "**Nombre de partants (réel)** — Compte brut des chevaux considérés dans le calcul. "
    "Il ne dit rien sur l’équilibre de la course, seulement sur sa taille."
)
st.markdown(
    "**Nombre effectif de chevaux (exp(H))** — Nombre de chevaux **équiprobables** qui donneraient la **même incertitude** que la course réelle. "
    "Toujours ≤ au nombre réel : s’il y a un gros favori, le nombre effectif baisse ; s’ils sont proches, il se rapproche du réel."
)

# Notes sur la somme/overround
if value_type.startswith("Cotes"):
    raw_sum = sum(1.0 / o for o in values)
    margin_pct = (raw_sum - 1.0) * 100.0
    st.caption(f"**Somme des proba implicites (brutes)** : {raw_sum:.4f}  |  **Overround** : {margin_pct:+.2f}% "
               f"({source_note} {norm_note})")
else:
    s = sum(values)
    st.caption(f"**Somme des probabilités fournies** : {s:.4f}  ({source_note} {norm_note})")

# Tableau
df = None
if show_table:
    records = []
    for i, v in enumerate(values, start=1):
        row = {"Cheval": i}
        if value_type.startswith("Cotes"):
            row["Entrée (cote)"] = v
            row["p implicite (brut)"] = 1.0 / v
        else:
            row["p fournie (brute)"] = v
        row["p utilisée"] = probs[i - 1]
        row["p utilisée (%)"] = 100.0 * probs[i - 1]
        records.append(row)
    df = pd.DataFrame.from_records(records)
    st.dataframe(df, use_container_width=True)

# Histogramme
if show_chart:
    chart_df = pd.DataFrame(
        {"Cheval": [f"{i}" for i in range(1, len(probs) + 1)],
         "Probabilité (%)": [100.0 * p for p in probs]}
    ).set_index("Cheval")
    st.bar_chart(chart_df, use_container_width=True)

# Download CSV
if df is None:
    df = pd.DataFrame({
        "Cheval": list(range(1, len(probs) + 1)),
        "p utilisée": probs,
        "p utilisée (%)": [100.0 * p for p in probs],
    })
csv = df.to_csv(index=False)
st.download_button("⬇️ Télécharger le tableau en CSV", data=csv,
                   file_name="race_entropy_table.csv", mime="text/csv")

# Footer
st.write("---")
st.caption(
    "Notes :\n"
    "- Les **cotes** (ex. 3.5) sont converties en p = 1/cote. Les cotes **fractionnelles** (ex. 8/1) sont supportées.\n"
    "- L’**overround** du bookmaker gonfle la somme des probabilités > 1. L’option d’**auto-normalisation** l’enlève.\n"
    "- Entropie en **nats** : H = -∑ p ln p. Nombre effectif = **exp(H)**.\n"
    "- *Marge fav vs 2e* : calculée sur les probabilités **utilisées** (après normalisation si activée)."
)

if __name__ == "__main__":
    pass
