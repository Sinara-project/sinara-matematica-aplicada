# ===========================================================
#  CLASSIFICAÇÃO DA QUALIDADE DA ÁGUA
#  Modelos Utilizados: KNN, Decision Tree, Naive Bayes
#  Versão Didática com comentários detalhados
# ===========================================================

# ===== 1. IMPORTAÇÕES =====
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    GridSearchCV, train_test_split, StratifiedKFold, KFold
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer, f1_score, accuracy_score, recall_score, precision_score, 
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import matplotlib.pyplot as plt
import logging
import time
from dotenv import load_dotenv
import os

# ===========================================================
#  2. CARREGAR VARIÁVEIS DE AMBIENTE
# ===========================================================
load_dotenv()  # Carrega variáveis do arquivo .env

# ===========================================================
#  3. CONFIGURAÇÃO DE LOG
# ===========================================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)  # Logger para mensagens informativas e de erro

# ===========================================================
#  4. FUNÇÃO: CONECTAR AO MONGODB
# ===========================================================
def conectar_mongodb():
    """
    Conecta ao MongoDB usando variáveis do .env e retorna a coleção.
    Importante: certifique-se de que MONGO_URI, DB_NAME e COLLECTION_NAME estão definidos no .env
    """
    mongo_uri = os.getenv("MONGO_URI")           # Lê a URI do MongoDB
    db_name = os.getenv("DB_NAME")               # Lê o nome do banco de dados
    collection_name = os.getenv("COLLECTION_NAME")  # Lê o nome da coleção

    client = MongoClient(mongo_uri)              # Cria o cliente MongoDB
    db = client[db_name]                         # Seleciona o banco
    return db[collection_name]                   # Retorna a coleção específica

# ===========================================================
#  5. FUNÇÃO: CARREGAR DADOS DO MONGODB
# ===========================================================
def carregar_dados(collection):
    """
    Lê todos os documentos da coleção e converte para DataFrame pandas.
    Cada documento do MongoDB vira uma linha do DataFrame.
    """
    dados = list(collection.find())  # Recupera todos os documentos
    return pd.DataFrame(dados)       # Converte para DataFrame

# ===========================================================
#  6. FUNÇÃO: PREPARAR DADOS
# ===========================================================
def preparar_dados(df):
    """
    Remove linhas com valores faltantes (NaN) e separa features do alvo.
    Features: colunas numéricas (medidas da água)
    Alvo: 'qualidade' (boa, ruim, etc.)
    """
    colunas_numericas = [
        "cloro_residual", "cor_agua_bruta", "cor_agua_tratada",
        "fluoreto", "nitrato", "ph_agua_bruta", "ph_agua_tratada",
        "turbidez_agua_bruta", "turbidez_agua_tratada"
    ]
    df = df.dropna(subset=colunas_numericas + ["qualidade"])  # Remove linhas com valores faltantes
    X = df[colunas_numericas].copy()                           # Features
    y = df["qualidade"].copy()                                 # Alvo
    return X, y

# ===========================================================
#  7. FUNÇÃO: DEFINIR VALIDAÇÃO CRUZADA
# ===========================================================
def definir_cv(y_train, n_train):
    """
    Define a estratégia de validação cruzada.
    Se todas as classes têm pelo menos 2 amostras, usa StratifiedKFold para manter proporção.
    Caso contrário, usa KFold simples.
    """
    min_class_count = np.min(np.bincount(y_train))  # Conta quantas amostras existem na menor classe
    if min_class_count >= 2:
        n_splits = min(5, min_class_count)          # Máximo de 5 folds, não excedendo o tamanho da menor classe
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        n_splits = min(2, n_train)                  # Para datasets pequenos
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    log.info(f"Validação cruzada: {n_splits} folds (min_class_count={min_class_count})")
    return cv

# ===========================================================
#  8. FUNÇÃO: GERAR LISTA DE K PARA KNN
# ===========================================================
def gerar_lista_k(n_train, desired_k=30):
    """
    Gera lista de valores de K para o KNN.
    Prioriza números ímpares para evitar empates no voto da maioria.
    """
    odd_ks = list(range(1, n_train + 1, 2))  # Lista de ímpares
    if len(odd_ks) < desired_k:
        even_ks = [k for k in range(2, n_train + 1, 2)]  # Lista de pares se precisar
        ks = (odd_ks + even_ks)[:desired_k]              # Combina ímpares e pares
    else:
        ks = odd_ks[:desired_k]  # Limita ao desired_k
    log.info(f"Total de k testados: {len(ks)}")
    return ks

# ===========================================================
#  9. FUNÇÃO: DEFINIR MODELOS E GRIDS
# ===========================================================
def definir_modelos(k_list):
    """
    Define os modelos, pipelines e as grades de hiperparâmetros para o GridSearchCV.
    - KNN: normaliza dados e testa K, weights e métricas.
    - DecisionTree: testa critérios, profundidade máxima e parâmetros de divisão.
    - NaiveBayes: aplica normalização e testa var_smoothing.
    """
    return {
        "KNN": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),        # Normaliza dados
                ('knn', KNeighborsClassifier())      # Classificador KNN
            ]),
            "param_grid": {                         # Hiperparâmetros do KNN
                'knn__n_neighbors': k_list,        # Vizinhos K
                'knn__weights': ['uniform', 'distance'],  # Uniform = todos têm mesmo peso, distance = pesos pela distância
                'knn__metric': ['euclidean', 'manhattan'] # Distância
            }
        },
        "DecisionTree": {
            "pipeline": Pipeline([
                ('tree', DecisionTreeClassifier(random_state=42)) # Classificador Decision Tree
            ]),
            "param_grid": {                   # Hiperparâmetros
                'tree__criterion': ['gini', 'entropy'],  # Critério de divisão
                'tree__max_depth': [None, 3, 5, 7, 9],   # Profundidade máxima
                'tree__min_samples_split': [2, 5, 10],   # Minimo para dividir nó
                'tree__min_samples_leaf': [1, 2, 4]      # Minimo de amostras em folha
            }
        },
        "NaiveBayes": {
            "pipeline": Pipeline([
                ('scaler', StandardScaler()),       # Normaliza dados
                ('nb', GaussianNB())                # Classificador Naive Bayes Gaussiano
            ]),
            "param_grid": {
                'nb__var_smoothing': np.logspace(-9, -1, 9)  # Pequenas variações para estabilidade
            }
        }
    }

# ===========================================================
# 10. FUNÇÃO PRINCIPAL CORRIGIDA
# ===========================================================
def main():
    # ---- Conectar e carregar dados ----
    collection = conectar_mongodb()        
    df = carregar_dados(collection)        
    log.info(f"Total de registros: {len(df)}")

    # ---- Preparar dados ----
    X, y = preparar_dados(df)              

    # ---- Codificar variável alvo ----
    le = LabelEncoder()                     
    y_enc = le.fit_transform(y)
    log.info(f"Classes codificadas: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ---- Dividir treino/teste (33% para teste) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.33, random_state=42, stratify=y_enc
    )

    # ---- Definir validação cruzada e lista de k ----
    cv = definir_cv(y_train, X_train.shape[0])
    k_list = gerar_lista_k(X_train.shape[0])

    # ---- Definir métricas ----
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'precision': make_scorer(precision_score, average='weighted')
    }

    # ---- Definir modelos e grids ----
    modelos = definir_modelos(k_list)
    resultados = []

    # ---- Treinar e avaliar modelos ----
    for nome, cfg in modelos.items():
        log.info(f"\n===== Treinando modelo: {nome} =====")
        inicio = time.time()
        grid = GridSearchCV(
            cfg["pipeline"], 
            cfg["param_grid"], 
            cv=cv,
            scoring=scoring,
            refit='f1',
            n_jobs=-1,
            verbose=0  # reduz saída detalhada
        )
        grid.fit(X_train, y_train)
        duracao = time.time() - inicio

        # Contar número de testes de forma genérica (funciona com múltiplas métricas)
        n_testes = len(next(iter(grid.cv_results_.values())))

        # Armazenar resultados
        resultados.append({
            "modelo": nome,
            "melhor_f1": grid.best_score_,
            "melhores_params": grid.best_params_,
            "n_testes": n_testes,
            "tempo": duracao,
            "grid": grid
        })

        # Mostrar resultados no terminal
        print(f"Modelo: {nome}")
        print(f"Tempo de treinamento: {duracao:.1f}s")
        print(f"Quantidade de combinações testadas: {n_testes}")
        print(f"Melhor F1-score: {grid.best_score_:.4f}")
        print(f"Melhores parâmetros: {grid.best_params_}\n")

    # ---- Selecionar melhor modelo ----
    melhor = max(resultados, key=lambda x: x["melhor_f1"])
    print("===== MELHOR MODELO =====")
    print(f"Modelo: {melhor['modelo']}")
    print(f"F1-score: {melhor['melhor_f1']:.4f}")
    print(f"Parâmetros: {melhor['melhores_params']}\n")

    # ---- Treinar modelo final ----
    modelo_final = melhor["grid"].best_estimator_
    modelo_final.fit(X_train, y_train)

    # ---- Avaliar no conjunto de teste ----
    y_pred = modelo_final.predict(X_test)
    print("=== RELATÓRIO FINAL NO TESTE ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ---- Matriz de confusão no terminal ----
    cm = confusion_matrix(y_test, y_pred)
    print("=== MATRIZ DE CONFUSÃO ===")
    # Cabeçalho
    print(" " * 12 + " ".join([f"{cls:>10}" for cls in le.classes_]))
    for i, row in enumerate(cm):
        print(f"{le.classes_[i]:<12}" + " ".join([f"{v:>10}" for v in row]))

# ===========================================================
# 11. EXECUÇÃO
# ===========================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Erro durante a execução: {e}")
