"""Converte um par anual de arquivos TXT Standard/Sample em CSV normalizado."""
import argparse
from pathlib import Path
import pandas as pd


def converter(originacao, performance):
    origin = pd.read_csv(originacao,sep="|",header=None,usecols=[0,1,9,10,11,19],dtype=str)
    origin.columns=["score","primeiro_pagamento","comprometimento","valor","ltv","contrato"]
    if origin.contrato.duplicated().any() or origin.contrato.isna().any():
        raise ValueError("Contrato ausente ou repetido na originação.")
    first = pd.to_datetime(origin.pop("primeiro_pagamento"),format="%Y%m").dt.to_period("M")
    origin["safra"] = (first-1).dt.to_timestamp()
    for col, sentinel in [("score",9999),("ltv",999),("comprometimento",999)]:
        origin[col] = pd.to_numeric(origin[col]).replace(sentinel,float("nan"))
    origin["valor"] = pd.to_numeric(origin.valor)
    starts = pd.Series(first.array,index=origin.contrato)
    pieces=[]
    for block in pd.read_csv(performance,sep="|",header=None,usecols=[0,1,3,8],dtype=str,chunksize=200000):
        block.columns=["contrato","data","status","baixa"]
        block = block[block.contrato.isin(starts.index)].copy()
        month = pd.to_datetime(block.data,format="%Y%m").dt.to_period("M")
        start = block.contrato.map(starts)
        block["idade"] = month.astype("int64") - start.astype("int64")
        block["data"] = month.dt.to_timestamp()
        pieces.append(block[block.idade.between(0,11)])
    hist = pd.concat(pieces,ignore_index=True)
    if hist.duplicated(["contrato","idade"]).any():
        raise ValueError("Mês duplicado no desempenho.")
    status = pd.to_numeric(hist.status,errors="coerce")
    bad = hist[status.ge(3)].groupby("contrato").data.min()
    clean = hist[status.isin([0,1,2])]
    # A liquidação voluntária encerra o risco somente com histórico anterior completo.
    payoff = clean[clean.baixa.eq("01")].set_index("contrato").idade
    good = clean.groupby("contrato").size()
    full = good[good==12].index
    through_payoff = clean.join(payoff.rename("fim"),on="contrato")
    prior = through_payoff[through_payoff.idade<=through_payoff.fim].groupby("contrato").size()
    closed = prior[prior.eq(payoff.reindex(prior.index)+1)].index
    valid = origin.contrato.isin(full.union(closed).union(bad.index))
    output = origin.loc[valid].copy()
    output["mes_evento"] = output.contrato.map(bad)
    return output, {"originais":len(origin),"incluidos":int(valid.sum()),
                    "censurados_ou_incompletos":int((~valid).sum())}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("originacao",type=Path)
    p.add_argument("performance",type=Path)
    p.add_argument("saida",type=Path)
    args=p.parse_args()
    frame,info=converter(args.originacao,args.performance)
    frame.to_csv(args.saida,index=False,mode="x")
    print(info)

