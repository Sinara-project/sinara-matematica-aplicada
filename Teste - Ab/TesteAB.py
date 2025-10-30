# =========================================
# Importação de bibliotecas
# =========================================

import numpy as np
from scipy import stats


# =========================================
# Dados --- Baseado no teste do Maze entre 2 interfaces de login, 
# registro de ponto e respondendo formulários dentro do aplicativo.
# =========================================
A = np.array([47.38,57.55,97.47,33.85,39.13])
B = np.array([48.11,47.58,120.36,175.79,69.83])

# Hipóteses:
# H₀: μA = μB  → Não há diferença significativa entre as interfaces
# H₁: μA ≠ μB  → Há diferença significativa entre as interfaces

# =========================================
# Teste t de Student (amostras independentes)
# =========================================

# Executa o teste t
t_value, p_value = stats.ttest_ind(A, B, equal_var=True)  

# Tamanhos das amostras
n_A, n_B = len(A), len(B)

# Graus de liberdade
degrees_of_freedom = n_A + n_B - 2

# Nível de significância
alpha = 0.01 # como tem muito pouco dado decidi diminuir o nivem de significância para 1%, pois assim o teste vai ser mais preciso

# Valor crítico (teste bicaudal)
t_critical = stats.t.ppf(1 - alpha/2, degrees_of_freedom)

# Exibe resultados
print(f"Valor t calculado: {t_value:.4f}")
print(f"Graus de liberdade: {degrees_of_freedom}")
print(f"Valor crítico de t (α = {alpha}): ±{t_critical:.4f}")
print(f"p-valor: {p_value:.6f}")


# =========================
# Conclusão
# =========================
if abs(t_value) > t_critical or p_value < alpha:
    print("✅ Rejeitamos H₀: há diferença estatisticamente significativa entre as interfaces.")
else:
    print("❌ Não rejeitamos H₀: não há diferença estatisticamente significativa entre as interfaces.")

# Interpretação
mean_A = np.mean(A)
mean_B = np.mean(B)
print(f"\nMédia Interface A: {mean_A:.2f} segundos")
print(f"Média Interface B: {mean_B:.2f} segundos")

if mean_A < mean_B:
    print("➡ A Interface A apresentou melhor desempenho (menor tempo médio).")
elif mean_A > mean_B:
    print("➡ A Interface B apresentou melhor desempenho (menor tempo médio).")
else:
    print("➡ As duas interfaces tiveram desempenho médio igual.")
