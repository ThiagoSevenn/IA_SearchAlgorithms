import numpy as np
import matplotlib.pyplot as plt
import statistics

PROBLEMAS = {}  # será preenchido depois das funções objetivo

# FUNÇÃO AUXILIAR - Verifica e aplica restrição de caixa aos candidatos
def check_bounds(candidate, bounds):
    lowers = np.array([b[0] for b in bounds])
    uppers = np.array([b[1] for b in bounds])
    return np.clip(candidate, lowers, uppers)

# FUNÇÕES OBJETIVO (vetorizadas / escala contínua)
# Função 1: soma dos quadrados (mínimização)
def func_1(sol):
    return np.sum(sol**2)

# Função 2: duas gaussianas (maximização neste caso no uso original)
def func_2(sol):
    x1, x2 = sol
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))

# Função 3: Ackley (mínimização clássica)
def func_3(sol):
    x1, x2 = sol
    part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    part2 = -np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
    return part1 + part2 + 20 + np.e

# Função 4: Rastrigin (mínimização)
def func_4(sol):
    return np.sum(sol**2 - 10 * np.cos(2 * np.pi * sol) + 10)

# Função 5: função composta com cosseno, pico gaussiano e termo pequeno
def func_5(sol):
    x1, x2 = sol
    return (x1 * np.cos(x1)) / 20.0 + 2 * np.exp(-(x1**2) - (x2 - 1)**2) + 0.01 * x1 * x2

# Função 6: mistura de senos (maximização no uso original)
def func_6(sol):
    x1, x2 = sol
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

# Dicionário de problemas com suas funções
PROBLEMAS = {
    1: {"func": func_1, "bounds": [(-100, 100)]*2, "min": True, "name": "Prob 1: x^2 + y^2"},
    2: {"func": func_2, "bounds": [(-2, 4), (-2, 5)], "min": False, "name": "Prob 2: Picos"},
    3: {"func": func_3, "bounds": [(-8, 8)]*2, "min": True, "name": "Prob 3: Ackley"},
    4: {"func": func_4, "bounds": [(-5.12, 5.12)]*2, "min": True, "name": "Prob 4: Rastrigin"},
    5: {"func": func_5, "bounds": [(-10, 10)]*2, "min": False, "name": "Prob 5: Complexa"},
    6: {"func": func_6, "bounds": [(-1, 3), (-1, 3)], "min": False, "name": "Prob 6: Senos"}
}

# ALGORITMOS DE BUSCA (HC, LRS, GRS)
# run_optimization executa uma única execução do algoritmo escolhido
def run_optimization(algo_name, obj_func, bounds, max_iter=1000, patience=50, minimize=True, **kwargs):
    dim = len(bounds)
    bounds_arr = np.array(bounds)

    # Hill Climbing: começa na borda inferior de cada dimensão (determinístico).
    # LRS/GRS: começam com solução aleatória dentro dos bounds.
    if algo_name == "Hill Climbing":
        current_sol = np.array([b[0] for b in bounds])
    else:
        current_sol = np.random.uniform(bounds_arr[:, 0], bounds_arr[:, 1], dim)

    current_score = obj_func(current_sol)
    best_sol = current_sol.copy()
    best_score = current_score

    history = [best_score]
    no_improv = 0
    iterations_run = 0

    for i in range(max_iter):
        iterations_run = i + 1
        candidate = None

        # - Hill Climbing: pequena perturbação (epsilon)
        if algo_name == "Hill Climbing":
            epsilon = kwargs.get('epsilon', 0.1)
            candidate = current_sol + np.random.uniform(-epsilon, epsilon, dim)
        # LRS: amostra normal centrada em current_sol com desvio sigma
        elif algo_name == "LRS":
            sigma = kwargs.get('sigma', 0.1)
            candidate = np.random.normal(current_sol, sigma, dim)
        # - GRS: amostra completamente aleatória nos bounds
        elif algo_name == "GRS":
            candidate = np.random.uniform(bounds_arr[:, 0], bounds_arr[:, 1], dim)
        else:
            raise ValueError(f"Algoritmo desconhecido: {algo_name}")

        # Aplica restrição de caixa para garantir validade do candidato e determina se houve melhora
        candidate = check_bounds(candidate, bounds)
        score = obj_func(candidate)
        improved = (score < best_score) if minimize else (score > best_score)

        if improved:
            best_sol = candidate
            best_score = score
            if algo_name != "GRS":
                current_sol = candidate
            no_improv = 0
        else:
            no_improv += 1

        history.append(best_score)

        # Critério de parada por paciência
        if no_improv >= patience:
            history.extend([best_score] * (max_iter - i - 1)) # Preenche o histórico
            break

    return best_sol, best_score, history, iterations_run

# ROUTINA PRINCIPAL DA ETAPA 1
def main_etapa1():
    print(">>> Iniciando ETAPA 1: Otimização Contínua")
    print("Executando 100 rodadas para cada um dos 18 cenários (6 problemas x 3 algoritmos)...")

    algos = [
        ("Hill Climbing", {"epsilon": 0.1}),
        ("LRS", {"sigma": 0.1}),
        ("GRS", {})
    ]

    # Estrutura de armazenamento:
    # data[prob_id][algo_name] = {'scores': [], 'coords': [], 'hists': [], 'iters': []}
    data = {pid: {a[0]: {'scores': [], 'coords': [], 'hists': [], 'iters': []} for a in algos} for pid in PROBLEMAS}

    # TESTE COM DIFERENTES VALORES: Varredura de ε (HC) e σ (LRS) usando Prob 1
    print("\n>>> Testando diferentes valores de ε (Hill Climbing) e σ (LRS)...")

    epsilons = [0.1, 0.05, 0.02, 0.01]
    sigmas = [0.5, 0.3, 0.1, 0.05]

    # tupla (valor, score)
    melhor_epsilon = None  
    melhor_sigma = None    

    prob_teste = PROBLEMAS[1]
    func_teste = prob_teste["func"]
    bounds_teste = prob_teste["bounds"]
    minimize_teste = prob_teste["min"]

    # Varre epsilons para HC (comportamento de minimização assumido por Prob 1)
    for eps in epsilons:
        sol, score, _, _ = run_optimization(
            "Hill Climbing",
            func_teste,
            bounds_teste,
            minimize=minimize_teste,
            epsilon=eps
        )
        if melhor_epsilon is None or score < melhor_epsilon[1]:
            melhor_epsilon = (eps, score)

    # Varre sigmas para LRS
    for sg in sigmas:
        sol, score, _, _ = run_optimization(
            "LRS",
            func_teste,
            bounds_teste,
            minimize=minimize_teste,
            sigma=sg
        )
        if melhor_sigma is None or score < melhor_sigma[1]:
            melhor_sigma = (sg, score)

    print(f"Melhor ε (Hill Climbing): {melhor_epsilon[0]}  → Score: {melhor_epsilon[1]:.6f}")
    print(f"Melhor σ (LRS): {melhor_sigma[0]}  → Score: {melhor_sigma[1]:.6f}")
    print(">>> Varredura concluída.\n")

    # LOOP PRINCIPAL DE EXECUÇÃO (100 rodadas por cenário)
    runs_per_combo = 100
    for pid, pdata in PROBLEMAS.items():
        for algo_name, params in algos:
            for _ in range(runs_per_combo):
                sol, score, hist, it_count = run_optimization(
                    algo_name, pdata['func'], pdata['bounds'],
                    minimize=pdata['min'], **params
                )
                data[pid][algo_name]['scores'].append(score)
                data[pid][algo_name]['coords'].append(sol)
                data[pid][algo_name]['hists'].append(hist)
                data[pid][algo_name]['iters'].append(it_count)

    # GERAÇÃO DO ARQUIVO TXT COM AS ESTATÍSTICAS
    filename = "resultados_estatisticos.txt"
    with open(filename, "w", encoding="utf-8") as f:
        header = f"{'Prob':<20} | {'Algoritmo':<15} | {'Melhor':<12} | {'Média Score':<12} | {'Moda Solução (Freq)':<22} | {'Iter(Méd)':<10} | {'Params':<15}"
        sep = "=" * len(header)

        def log(msg):
            print(msg)        
            f.write(msg + "\n") 

        log("\n" + sep)
        log(f"{'TABELA DE RESULTADOS ESTATÍSTICOS (100 RODADAS)':^{len(header)}}")
        log(sep)
        log(header)
        log("-" * len(header))

        for pid, pdata in PROBLEMAS.items():
            for algo_name, params in algos:
                scores = data[pid][algo_name]['scores']
                iters = data[pid][algo_name]['iters']

                # Estatísticas principais
                best_val = np.min(scores) if pdata['min'] else np.max(scores)
                mean_val = np.mean(scores)
                mean_iter = np.mean(iters)

                # Moda (com arredondamento para reduzir sensibilidade)
                rounded = [round(s, 4) for s in scores]
                try:
                    mode_val = statistics.mode(rounded)
                    freq = rounded.count(mode_val)
                    mode_str = f"{mode_val} ({freq}x)"
                except statistics.StatisticsError:
                    mode_str = "Disperso"

                # Formata parâmetros para exibição
                param_str = str(params).replace("{", "").replace("}", "").replace("'", "")
                if not param_str:
                    param_str = "N/A"

                line = f"{pdata['name']:<20} | {algo_name:<15} | {best_val:<12.4f} | {mean_val:<12.4f} | {mode_str:<22} | {mean_iter:<10.1f} | {param_str:<15}"
                log(line)
            log("-" * len(header))

        print(f"\nArquivo salvo: {filename}")

    # PLOTAGEM (3 FIGURAS): convergência média, boxplots e scatter final
    print("\nGerando gráficos (Convergência, Boxplots, Scatter)...")

    # FIGURA 1: CONVERGÊNCIA (Média das 100 rodadas)
    fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle("ETAPA 1 - Convergência (Média de 100 Rodadas)")
    axs1 = axs1.ravel()

    for i, pid in enumerate(PROBLEMAS):
        ax = axs1[i]
        for algo_name, _ in algos:
            all_hists = data[pid][algo_name]['hists']
            # calcula comprimento máximo e faz o padding com o último valor de cada histórico.
            if len(all_hists) == 0:
                continue
            max_len = max(len(h) for h in all_hists)
            padded = np.zeros((len(all_hists), max_len))
            for j, h in enumerate(all_hists):
                h_arr = np.array(h)
                if len(h_arr) < max_len:
                    # pad com o último valor disponível
                    pad_val = h_arr[-1]
                    padded[j, :] = np.concatenate([h_arr, np.full(max_len - len(h_arr), pad_val)])
                else:
                    padded[j, :] = h_arr[:max_len]
            mean_hist = np.mean(padded, axis=0)
            ax.plot(mean_hist, label=algo_name, linewidth=1.5)

        ax.set_title(PROBLEMAS[pid]['name'])
        ax.set_xlabel("Iterações")
        ax.set_ylabel("Fitness (Média)")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # FIGURA 2: BOXPLOTS (Distribuição dos Scores Finais)
    fig2, axs2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle("ETAPA 1 - Boxplots (Distribuição de 100 Soluções)")
    axs2 = axs2.ravel()

    for i, pid in enumerate(PROBLEMAS):
        ax = axs2[i]
        scores_list = [data[pid][a[0]]['scores'] for a in algos]
        labels = [a[0] for a in algos]
        ax.boxplot(scores_list, tick_labels=labels)
        ax.set_title(PROBLEMAS[pid]['name'])
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # FIGURA 3: SCATTER FINAL (Dispersão no Espaço de Busca)
    fig3, axs3 = plt.subplots(2, 3, figsize=(18, 10))
    fig3.suptitle("ETAPA 1 - Scatter Final (Onde as 100 rodadas terminaram)")
    axs3 = axs3.ravel()
    colors = {'Hill Climbing': 'red', 'LRS': 'blue', 'GRS': 'green'}

    for i, pid in enumerate(PROBLEMAS):
        ax = axs3[i]
        for algo_name, _ in algos:
            coords = np.array(data[pid][algo_name]['coords'])
            if coords.size == 0:
                continue
            ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.6,
                       c=colors.get(algo_name, 'black'), label=algo_name)

        bounds = PROBLEMAS[pid]['bounds']
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_title(PROBLEMAS[pid]['name'])
        if i == 0:
            ax.legend() 

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()

if __name__ == "__main__":
    main_etapa1()