# ======================================
# Importações de bibliotecas
# ======================================

from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from pymongo import MongoClient
import os
import logging
from dotenv import load_dotenv

# Blueprint
agua_bp = Blueprint("agua_bp", __name__)

# Log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Variáveis do .env
load_dotenv()




# ======================================
# Funções
# ======================================
def conectar_mongodb():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("DB_NAME")
    collection_name = os.getenv("COLLECTION_NAME")
    client = MongoClient(mongo_uri)
    db = client[db_name]
    return db[collection_name]

def carregar_dados(collection):
    dados = list(collection.find())
    return pd.DataFrame(dados)

def preparar_dados(df):
    colunas_numericas = [
        "cloro_residual", "cor_agua_bruta", "cor_agua_tratada",
        "fluoreto", "nitrato", "ph_agua_bruta", "ph_agua_tratada",
        "turbidez_agua_bruta", "turbidez_agua_tratada"
    ]
    df = df.dropna(subset=colunas_numericas + ["qualidade"])
    X = df[colunas_numericas].copy()
    y = df["qualidade"].copy()
    return X, y

def definir_cv(y_train, n_train):
    min_class_count = np.min(np.bincount(y_train))
    if min_class_count >= 2:
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = min(2, n_train)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cv

def gerar_lista_k(n_train, desired_k=30):
    odd_ks = list(range(1, n_train + 1, 2))
    if len(odd_ks) < desired_k:
        even_ks = [k for k in range(2, n_train + 1, 2)]
        ks = (odd_ks + even_ks)[:desired_k]
    else:
        ks = odd_ks[:desired_k]
    return ks

def definir_modelos(k_list):
    return {
        "KNN": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())
            ]),
            "param_grid": {
                'knn__n_neighbors': k_list,
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['euclidean', 'manhattan']
            }
        },
        "DecisionTree": {
            "pipeline": Pipeline([
                ('tree', DecisionTreeClassifier(random_state=42))
            ]),
            "param_grid": {
                'tree__criterion': ['gini', 'entropy'],
                'tree__max_depth': [None, 3, 5, 7, 9],
                'tree__min_samples_split': [2, 5, 10],
                'tree__min_samples_leaf': [1, 2, 4]
            }
        },
        "NaiveBayes": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),
                ('nb', GaussianNB())
            ]),
            "param_grid": {
                'nb__var_smoothing': np.logspace(-9, -1, 9)
            }
        }
    }


# ======================================
# Treinos
# ======================================
def treinar_melhor_modelo():
    collection = conectar_mongodb()
    df = carregar_dados(collection)
    X, y = preparar_dados(df)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.33, random_state=42, stratify=y_enc
    )

    cv = definir_cv(y_train, X_train.shape[0])
    k_list = gerar_lista_k(X_train.shape[0])
    modelos = definir_modelos(k_list)

    resultados = []
    for nome, cfg in modelos.items():
        grid = GridSearchCV(
            cfg["pipeline"],
            cfg["param_grid"],
            cv=cv,
            scoring='f1_weighted',
            refit=True,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        resultados.append({"nome": nome, "f1": grid.best_score_, "grid": grid})

    melhor = max(resultados, key=lambda x: x["f1"])
    modelo_final = melhor["grid"].best_estimator_
    modelo_final.fit(X_train, y_train)
    return modelo_final, le

modelo_final, le = treinar_melhor_modelo()
log.info("Modelo treinado")


# ======================================
# Endpoint
# ======================================
@agua_bp.route("/prever", methods=["POST"])
def prever_qualidade():
    try:
        dados = request.get_json()
        colunas = [
            "cloro_residual", "cor_agua_bruta", "cor_agua_tratada",
            "fluoreto", "nitrato", "ph_agua_bruta", "ph_agua_tratada",
            "turbidez_agua_bruta", "turbidez_agua_tratada"
        ]
        X_novo = pd.DataFrame([dados], columns=colunas)
        pred_enc = modelo_final.predict(X_novo)[0]
        pred = le.inverse_transform([pred_enc])[0]
        return jsonify({"qualidade": pred})
    except Exception as e:
        log.error(f"Erro na previsão: {e}")
        return jsonify({"erro": str(e)}), 400
