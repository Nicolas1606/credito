"""A política é definida em validação e permanece fixa no teste."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

RISCO_MAX = .045
REVISAO_MAX = .10
MIN_APROVADOS = 100


def intervalo(k, n):
    if not n:
        return np.nan, np.nan
    p, z = k / n, 1.96
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    margin = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0., center-margin), min(1., center+margin)


def decidir(p, aprovar, revisar):
    return np.select([p <= aprovar, p <= revisar], ["aprovar", "revisar"], default="recusar")


def resumir(y, decisoes):
    a = decisoes == "aprovar"
    n, k = int(a.sum()), int(np.asarray(y)[a].sum())
    lo, hi = intervalo(k, n)
    return dict(total=len(y), aprovados=n, eventos=k, aprovacao=n/len(y),
                revisao=float(np.mean(decisoes == "revisar")), risco=k/n if n else np.nan,
                inferior95=lo, superior95=hi)


def selecionar(y, p):
    rows = []
    for a in np.unique(np.quantile(p, np.arange(1, 100)/100)):
        # Usa a maior faixa de revisão que cabe no orçamento, respeitando empates.
        options = np.unique(np.r_[a, np.quantile(p, np.arange(1, 101)/100)])
        r = max(v for v in options if v >= a and np.mean((p > a) & (p <= v)) <= REVISAO_MAX)
        rows.append(dict(corte_aprovar=a, corte_revisar=r, **resumir(y, decidir(p,a,r))))
    frontier = pd.DataFrame(rows)
    ok = frontier[(frontier.superior95 <= RISCO_MAX) & (frontier.aprovados >= MIN_APROVADOS)]
    if ok.empty:
        raise ValueError("Nenhuma política viável na validação. Verifique a fronteira e as premissas.")
    best = ok.sort_values("aprovados").iloc[-1]
    return best.corte_aprovar, best.corte_revisar, frontier


def qualidade(y, p):
    return {"roc_auc": roc_auc_score(y,p), "average_precision": average_precision_score(y,p),
            "brier": brier_score_loss(y,p), "prevalencia": np.mean(y), "pd_media": np.mean(p)}


def psi(ref, current):
    edges = np.unique(np.r_[-np.inf, np.quantile(ref, np.arange(1,10)/10), np.inf])
    def shares(x):
        counts = np.histogram(x, edges)[0] + .5
        return counts / counts.sum()
    a, b = shares(ref), shares(current)
    return float(np.sum((b-a)*np.log(b/a)))

