# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 15:52:26 2018

@author: vilson.ferreira
"""

# Carga dos dados
import pandas as pd

Dados = pd.read_json("D:/Totvs_Challenge/challenge.json")

# Eliminação de nulos (sem critérios analisados, foco em rodar o modelo)
Dados = Dados.dropna()

# Seleção das variáveis indicadas
DadosModelo = Dados[['branch_id', 'customer_code', 'group_code', 'item_code',
                     'item_total_price', 'quantity', 'sales_channel', 'segment_code', 
                     'seller_code', 'total_price', 'unit_price']]

ClassesModelo = Dados["is_churn"]

# Separação dos dados de treino e teste
NumTreino = int(DadosModelo.shape[0]*0.8)

DadosTreino = DadosModelo[:NumTreino]
ClassesTreino = ClassesModelo[:NumTreino]
DadosTeste = DadosModelo[NumTreino:]
ClassesTeste = ClassesModelo[NumTreino:]

# Treinamento do modelo
from sklearn import linear_model
from sklearn import metrics

Modelo = linear_model.LogisticRegression(penalty="l2")
Modelo.fit(DadosTreino, ClassesTreino)

# Teste do modelo
Predicao = Modelo.predict(DadosTeste)

# Métricas
metrics.mean_absolute_error(ClassesTeste, Predicao)
metrics.mean_squared_error(ClassesTeste, Predicao)
metrics.explained_variance_score(ClassesTeste, Predicao)
metrics.accuracy_score(ClassesTeste, Predicao)