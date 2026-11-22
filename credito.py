"""CLI de treinamento, backtest, simulação e pontuação em lote."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dados import ALVO, CHAVES, FEATURES, gerar, preparar, validar_propostas
from politica import decidir, psi, qualidade, resumir, selecionar, RISCO_MAX, REVISAO_MAX

PERIODOS = {"treino": (2011, 2015), "calibracao": (2017, 2017),
            "politica": (2019, 2019), "teste": (2021, 2022)}
CORTE = "2024-01-01"


def log_odds(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def prever(model, calibrator, frame):
    raw = model.predict_proba(frame[FEATURES])[:,1]
    return calibrator.predict_proba(log_odds(raw))[:,1] if calibrator is not None else raw


def train(raw, folder, source):
    data = preparar(raw, CORTE)
    sets = {name: data[data.safra.dt.year.between(start,end)] for name,(start,end) in PERIODOS.items()}
    for name, frame in sets.items():
        if frame[ALVO].nunique() < 2:
            raise ValueError(f"Período {name} precisa de observações das duas classes.")
    t, c, v, test = [sets[k] for k in PERIODOS]
    data.groupby(data.safra.dt.to_period("M"))[ALVO].agg(["count","sum","mean"]).to_csv(folder/"safras.csv")
    data[FEATURES].describe().T.assign(ausentes=data[FEATURES].isna().sum()).to_csv(folder/"perfil.csv")
    models = {"logistica": make_pipeline(SimpleImputer(strategy="median",add_indicator=True),
                        StandardScaler(), LogisticRegression(max_iter=500)),
              "arvores": HistGradientBoostingClassifier(max_iter=90, max_leaf_nodes=10,
                        min_samples_leaf=45, learning_rate=.06, early_stopping=False,random_state=26)}
    rows, candidates = [], {}
    for name, model in models.items():
        model.fit(t[FEATURES],t[ALVO])
        cal = LogisticRegression(C=1000).fit(log_odds(prever(model,None,c)),c[ALVO])
        for label, method in [("original",None),("calibrado",cal)]:
            key = f"{name}_{label}"
            candidates[key] = (model, method)
            rows.append(dict(modelo=key, **qualidade(v[ALVO],prever(model,method,v))))
    comparison = pd.DataFrame(rows).sort_values("brier")
    comparison.to_csv(folder/"modelos_validacao.csv",index=False)
    name = comparison.iloc[0].modelo
    model, cal = candidates[name]
    pv = prever(model,cal,v)
    pd.DataFrame({"pd_12m":pv}).to_csv(folder/"referencia_pd.csv",index=False)
    a,r,front = selecionar(v[ALVO],pv)
    front.to_csv(folder/"fronteira_validacao.csv",index=False)
    source_hash = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    version = f"credito-v1-{source_hash[:12]}"
    bundle = dict(model=model,cal=cal,a=a,r=r,version=version,source=source)
    joblib.dump(bundle,folder/"modelo.joblib")
    # A regra de referência usa apenas o score na concessão.
    def reference(frame):
        score = frame.score.fillna(0)
        return np.select([score>=720,score>=710],["aprovar","revisar"],default="recusar")
    pt = prever(model,cal,test)
    decisions = decidir(pt,a,r)
    comparison_policy = []
    for period,frame,p in [("validacao",v,pv),("teste",test,pt)]:
        for policy,d in [("referencia",reference(frame)),("selecionada",decidir(p,a,r))]:
            comparison_policy.append(dict(periodo=period,politica=policy,**resumir(frame[ALVO],d)))
    pd.DataFrame(comparison_policy).to_csv(folder/"politicas.csv",index=False)
    audited = test[CHAVES+[ALVO]].assign(pd_12m=pt,decisao=decisions)
    audited.to_csv(folder/"teste_por_contrato.csv",index=False)
    months=[]
    for date,g in audited.groupby("safra"):
        s = resumir(g[ALVO],g.decisao)
        shift = psi(pv,g.pd_12m)
        months.append(dict(safra=date,**s,pd_media=g.pd_12m.mean(),psi=shift,
            alerta_risco=bool(s["risco"]>RISCO_MAX),alerta_capacidade=bool(s["revisao"]>REVISAO_MAX),
            alerta_distribuicao=bool(shift>.20)))
    pd.DataFrame(months).to_csv(folder/"monitoramento.csv",index=False)
    calibration = audited.assign(faixa=pd.cut(pt, np.r_[0,.025,.05,.075,.10,.15,.25,1],include_lowest=True))
    calibration.groupby("faixa",observed=True).agg(n=(ALVO,"size"), previsto=("pd_12m","mean"),
                                  observado=(ALVO,"mean")).to_csv(folder/"calibracao_teste.csv")
    selected = comparison_policy[-1]
    release = bool(selected["superior95"]<=RISCO_MAX and selected["revisao"]<=REVISAO_MAX)
    summary = dict(fonte=source,hash_dados=source_hash,versao=version,modelo=name,
                   corte_observacao=CORTE,registros=len(raw),maduros=len(data),
                   fora_das_janelas=len(data)-sum(map(len,sets.values())),
                   amostras={k:len(g) for k,g in sets.items()},corte_aprovar=a,corte_revisar=r,
                   teste_modelo=qualidade(test[ALVO],pt),teste_politica=selected,
                   restricoes_atendidas_no_teste=release)
    (folder/"resultado.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    test[CHAVES+FEATURES].head(12).to_csv(folder/"exemplo_lote.csv",index=False)
    return summary


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd",required=True)
    tr = subs.add_parser("treinar")
    tr.add_argument("--historico",type=Path)
    tr.add_argument("--saida",type=Path,required=True)
    sc = subs.add_parser("pontuar")
    sc.add_argument("--modelo",type=Path,required=True)
    sc.add_argument("--lote",type=Path,required=True)
    sc.add_argument("--saida",type=Path,required=True)
    sim = subs.add_parser("simular")
    sim.add_argument("--execucao",type=Path,required=True)
    sim.add_argument("--corte-aprovar",type=float,required=True)
    sim.add_argument("--corte-revisar",type=float,required=True)
    args = parser.parse_args()
    if args.cmd=="treinar":
        if args.saida.exists():
            parser.error("Escolha uma nova pasta de execução para preservar o histórico.")
        args.saida.parent.mkdir(parents=True,exist_ok=True)
        raw = pd.read_csv(args.historico) if args.historico else gerar()
        with TemporaryDirectory(dir=args.saida.parent) as tmp:
            stage = Path(tmp)/"execucao"
            stage.mkdir()
            result = train(raw,stage,"CSV fornecido" if args.historico else "simulação sintética")
            stage.rename(args.saida)
        print(json.dumps(result,indent=2,ensure_ascii=False))
    elif args.cmd=="pontuar":
        if args.saida.exists():
            parser.error("Escolha um novo arquivo de saída.")
        bundle = joblib.load(args.modelo)
        frame = validar_propostas(pd.read_csv(args.lote,dtype={"contrato":str}))
        p = prever(bundle["model"],bundle["cal"],frame)
        result = frame[CHAVES].assign(pd_12m=p,decisao=decidir(p,bundle["a"],bundle["r"]),
                         versao=bundle["version"],origem=bundle["source"])
        result.to_csv(args.saida,index=False,mode="x")
        print(f"{len(result)} propostas pontuadas em {args.saida}")
    else:
        if not 0 <= args.corte_aprovar <= args.corte_revisar <= 1:
            parser.error("Exija 0 <= corte de aprovação <= corte de revisão <= 1.")
        df = pd.read_csv(args.execucao/"teste_por_contrato.csv")
        print("Cenário exploratório sobre o teste já aberto.")
        print(resumir(df[ALVO],decidir(df.pd_12m,args.corte_aprovar,args.corte_revisar)))


if __name__=="__main__":
    cli()

