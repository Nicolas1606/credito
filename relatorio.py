"""Cria a figura a partir dos CSVs de uma execução."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from politica import RISCO_MAX


def render(folder):
    monitor = pd.read_csv(folder/"monitoramento.csv",parse_dates=["safra"])
    result = pd.read_csv(folder/"politicas.csv").query("periodo == 'teste'").set_index("politica")
    fig, (a,b) = plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    result[["aprovacao","revisao"]].mul(100).plot.bar(ax=a,color=["#12675d","#e0aa59"],rot=0)
    a.set(xlabel="Política",ylabel="Contratos (%)",title="Decisões no período de teste")
    a.legend(["Aprovação","Revisão"])
    b.plot(monitor.safra,100*monitor.risco,color="#125c74",label="Atraso observado")
    b.fill_between(monitor.safra,100*monitor.inferior95,100*monitor.superior95,alpha=.18)
    b.axhline(100*RISCO_MAX,color="#a44235",ls="--",label="Teto da política")
    b.set(title="Risco entre aprovados por safra",ylabel="Contratos com atraso (%)")
    b.tick_params(axis="x",labelrotation=35)
    b.legend()
    fig.savefig(folder/"painel.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("pasta",type=Path)
    render(p.parse_args().pasta)

