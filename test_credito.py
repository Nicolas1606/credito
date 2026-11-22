import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from dados import gerar, preparar, validar_propostas, ALVO, FEATURES
from politica import decidir, selecionar, intervalo, psi
from credito import PERIODOS
from importar_freddie import converter


class TestCredito(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = gerar().head(6)

    def test_maturacao(self):
        raw = self.raw.copy()
        raw["safra"] = pd.Timestamp("2023-01-01")
        raw["mes_evento"] = pd.to_datetime(["2023-05-01",None,"2024-02-01",None,None,"2024-01-01"])
        self.assertTrue(preparar(raw,"2023-12-01").empty)
        self.assertEqual(preparar(raw,"2024-01-01")[ALVO].tolist(),[1,0,0,0,0,1])

    def test_duplicado_e_faixas(self):
        with self.assertRaises(ValueError):
            validar_propostas(pd.concat([self.raw,self.raw]))
        with self.assertRaises(ValueError):
            validar_propostas(self.raw.assign(score=1000))

    def test_sem_informacao(self):
        with self.assertRaises(ValueError):
            validar_propostas(self.raw.assign(**{k:np.nan for k in FEATURES}))

    def test_intervalo_temporal(self):
        bounds=list(PERIODOS.values())
        for (_,end),(start,_) in zip(bounds,bounds[1:]):
            self.assertGreaterEqual(start-end,2)

    def test_cortes(self):
        self.assertEqual(decidir(np.array([.05,.051,.10,.11]),.05,.1).tolist(),
                         ["aprovar","revisar","revisar","recusar"])

    def test_limites_politica(self):
        p=np.linspace(.001,.8,500)
        y=np.r_[np.zeros(400),np.ones(100)]
        a,r,_=selecionar(y,p)
        d=decidir(p,a,r)
        self.assertLessEqual(np.mean(d=="revisar"),.1)
        sub=y[d=="aprovar"]
        self.assertLessEqual(intervalo(sub.sum(),len(sub))[1],.045)
        with self.assertRaises(ValueError):
            selecionar(np.ones(500),p)

    def test_drift(self):
        x=np.arange(100)
        self.assertAlmostEqual(psi(x,x),0)
        self.assertGreater(psi(x,x+100),.2)

    def test_adaptador(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            ids=["evento","faltante","completo","quitado"]
            orig=[]
            for key in ids:
                row=[""]*20
                for i,val in [(0,"730"),(1,"201801"),(9,"32"),(10,"90000"),(11,"80"),(19,key)]:
                    row[i]=val
                orig.append("|".join(row))
            (root/"o").write_text("\n".join(orig))
            perf=[]
            for key,n in zip(ids,[5,10,12,5]):
                for m in range(1,n+1):
                    status="3" if key=="evento" and m==5 else "0"
                    code="01" if key=="quitado" and m==5 else ""
                    perf.append("|".join([key,f"2018{m:02}","80000",status,"","","","",code]))
            (root/"p").write_text("\n".join(perf))
            df,info=converter(root/"o",root/"p")
            self.assertEqual(info["censurados_ou_incompletos"],1)
            self.assertEqual(set(df.contrato),{"evento","completo","quitado"})
            self.assertEqual(preparar(df,"2019-01-01")[ALVO].sum(),1)


if __name__=="__main__":
    unittest.main()

