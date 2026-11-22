"""Acompanha uma nova safra, com métricas imediatas ou com resultados maduros."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from credito import prever
from dados import preparar, validar_propostas, ALVO
from politica import decidir, psi, resumir, RISCO_MAX, REVISAO_MAX


def monitorar(raw, bundle, referencia, corte=None):
    frame = validar_propostas(raw)
    if corte is not None:
        frame = preparar(frame,corte)
    if frame.empty:
        raise ValueError("Nenhum contrato elegível para o corte.")
    out = []
    for date,g in frame.groupby("safra"):
        p = prever(bundle["model"],bundle["cal"],g)
        d = decidir(p,bundle["a"],bundle["r"])
        review = (d=="revisar").mean()
        shift = psi(referencia,p)
        row = dict(safra=date,versao=bundle["version"],total=len(g),aprovacao=(d=="aprovar").mean(),
                   revisao=review,pd_media=p.mean(),psi=shift,alerta_distribuicao=shift>.20,
                   alerta_capacidade=review>REVISAO_MAX)
        if corte is not None:
            row.update(resumir(g[ALVO],d))
            row["alerta_risco"] = row["risco"] > RISCO_MAX
        out.append(row)
    return pd.DataFrame(out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execucao",type=Path)
    parser.add_argument("lote",type=Path)
    parser.add_argument("saida",type=Path)
    parser.add_argument("--corte",help="Com desfechos observados: YYYY-MM-01")
    args=parser.parse_args()
    bundle=joblib.load(args.execucao/"modelo.joblib")
    referencia=pd.read_csv(args.execucao/"referencia_pd.csv").pd_12m
    report=monitorar(pd.read_csv(args.lote,dtype={"contrato":str}),bundle,referencia,args.corte)
    report.to_csv(args.saida,index=False,mode="x")

