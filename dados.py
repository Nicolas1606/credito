"""Gerador reproduzível. Nenhum registro pertence a uma pessoa real."""
import numpy as np
import pandas as pd

FEATURES = ["score", "comprometimento", "ltv", "valor"]
CHAVES = ["contrato", "safra"]
ALVO = "atraso90_12m"


def gerar():
    rng = np.random.default_rng(2026)
    months = pd.date_range("2011-01-01", "2024-01-01", freq="MS")
    dates = months.repeat(300)
    n = len(dates)
    stress = (dates.year >= 2022).astype(float)
    score = rng.normal(680 - 9 * stress, 62, n).clip(350, 850).round()
    dti = (rng.beta(2.4, 4.8, n) * 85 + 2 * stress).round(1)
    ltv = (rng.beta(5, 2, n) * 95 + 5).round(1)
    amount = rng.lognormal(11.3, .58, n).clip(8000, 800000).round(0)
    logit = -3.05 - .017 * (score - 680) + .038 * (dti - 28) + .023 * (ltv - 74)
    logit += .7 * ((ltv > 86) & (dti > 38)) + .38 * stress
    event = rng.random(n) < 1 / (1 + np.exp(-logit))
    # O evento é o primeiro atraso de 90 dias dentro dos 12 meses seguintes.
    offset = rng.integers(3, 13, size=n)
    df = pd.DataFrame({"contrato": [f"S{i:06}" for i in range(n)], "safra": dates,
                       "score": score, "comprometimento": dti, "ltv": ltv, "valor": amount})
    df["mes_evento"] = (dates.to_period("M") + offset).to_timestamp().where(event)
    # Falta de informação na proposta, independente do rótulo.
    df.loc[rng.random(n) < .035, "comprometimento"] = np.nan
    return df


def validar_propostas(df):
    missing = set(CHAVES + FEATURES) - set(df.columns)
    if missing or df.empty:
        raise ValueError(f"Lote vazio ou colunas ausentes: {sorted(missing)}")
    if df.contrato.isna().any() or df.contrato.duplicated().any():
        raise ValueError("Cada contrato deve ter um identificador único e preenchido.")
    df = df.copy()
    df["safra"] = pd.to_datetime(df.safra, errors="raise").dt.to_period("M").dt.to_timestamp()
    if df.safra.isna().any():
        raise ValueError("Safra ausente.")
    for col, low, high in [("score", 300, 850), ("comprometimento", 0, 100),
                            ("ltv", 0, 200), ("valor", .01, np.inf)]:
        s = pd.to_numeric(df[col], errors="raise")
        if ((s.notna()) & (~s.between(low, high) | ~np.isfinite(s))).any():
            raise ValueError(f"Valor inválido em {col}.")
        df[col] = s
    if df[FEATURES].isna().all(axis=1).any():
        raise ValueError("Proposta sem nenhuma variável de risco.")
    return df


def preparar(raw, corte):
    df = validar_propostas(raw)
    event = pd.to_datetime(df["mes_evento"], errors="raise").dt.to_period("M").dt.to_timestamp()
    if (event <= df.safra).any():
        raise ValueError("Evento anterior ou simultâneo à safra.")
    maturity = df.safra + pd.DateOffset(months=12)
    df[ALVO] = (event <= maturity).astype(int)
    return df.loc[maturity <= pd.Timestamp(corte), CHAVES + FEATURES + [ALVO]].copy()

