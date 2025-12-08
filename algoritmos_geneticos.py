import numpy as np
import matplotlib.pyplot as plt
import random
import time
import statistics
import os

# PARTE 1: 8 RAINHAS (SIMULATED ANNEALING)
# Classe que resolve o problema das 8 rainhas usando Simulated Annealing
class EightQueensSA:
    def __init__(self, temp_initial=1000, alpha=0.95):
        self.n = 8
        self.temp_initial = temp_initial
        self.alpha = alpha

    # Conta o número de pares de rainhas que se atacam (função objetivo a minimizar)
    def count_attacks(self, board):
        h = 0
        # Percorre pares de colunas e conta ataques por linha ou diagonal
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                    h += 1
        return h

    # Gera um vizinho modificando a linha de uma coluna aleatória (garante mudança)
    def get_neighbor(self, board):
        neighbor = board.copy()
        col = np.random.randint(0, self.n)
        new_row = np.random.randint(0, self.n)
        # Garante que a nova linha seja diferente da atual (evita vizinho idêntico)
        while new_row == neighbor[col]:
            new_row = np.random.randint(0, self.n)
        neighbor[col] = new_row
        return neighbor

    # Executa o SA por até max_iter iterações e retorna melhor solução encontrada + histórico
    def solve(self, max_iter=2000):
        # Solução inicial aleatória (cada coluna recebe uma linha aleatória)
        current_sol = np.random.randint(0, self.n, self.n)
        current_h = self.count_attacks(current_sol)
        best_sol, best_h = current_sol.copy(), current_h

        temp = self.temp_initial
        # Históricos de energia e temperatura para plot
        hist_energy = []  
        hist_temp = []    

        start_t = time.time()

        for i in range(max_iter):
            hist_energy.append(current_h)
            hist_temp.append(temp)

            # Se já encontramos solução sem ataques, interrompe
            if best_h == 0:
                break

            neighbor = self.get_neighbor(current_sol)
            h_nb = self.count_attacks(neighbor)
            delta = h_nb - current_h

            # Se o vizinho for melhor, aceita sempre
            if delta < 0:
                current_sol, current_h = neighbor, h_nb
                # Atualiza melhor global se necessário
                if current_h < best_h:
                    best_sol, best_h = current_sol.copy(), current_h
            else:
                # Aceita pior com probabilidade exp(-delta/temp) (criterio Metropolis)
                if temp > 1e-10 and random.random() < np.exp(-delta / temp):
                    current_sol, current_h = neighbor, h_nb

            # Decaimento exponencial da temperatura
            temp *= self.alpha

        elapsed = time.time() - start_t
        return best_sol, best_h, hist_energy, hist_temp, elapsed

# Plota o dashboard do 8 Rainhas com 4 painéis
def plot_8queens_dashboard(sol, attacks, h_ene, h_temp, solutions_over_time):
    fig = plt.figure(figsize=(14, 9))
    # Figura 1: suptitle com o formato solicitado
    fig.suptitle("ETAPA 2 - Dashboard 8 Rainhas (Simulated Annealing)", fontsize=14)

    # 1. Energia vs Iteração
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(h_ene, label="Energia (Ataques)", color='red')
    ax1.set_title("Energia (Ataques) vs Iteração")
    ax1.set_xlabel("Iterações")
    ax1.set_ylabel("Conflitos")
    ax1.grid(True)
    ax1.legend()

    # 2. Temperatura vs Iteração
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(h_temp, label="Temperatura", color='orange')
    ax2.set_title("Decaimento da Temperatura")
    ax2.set_xlabel("Iterações")
    ax2.set_ylabel("Temperatura")
    ax2.grid(True)
    ax2.legend()

    # 3. Tabuleiro
    ax3 = fig.add_subplot(2, 2, 3)
    # Monta um tabuleiro xadrezado para visual
    board = np.zeros((8, 8))
    board[1::2, ::2] = 1
    board[::2, 1::2] = 1
    ax3.imshow(board, cmap='gray', interpolation='nearest')
    # Desenha as rainhas com um símbolo; cor verde se sem ataques, senão vermelho
    for c, r in enumerate(sol):
        ax3.text(c, r, '♕', fontsize=24, ha='center', va='center',
                 color='green' if attacks == 0 else 'red')
    ax3.set_title(f"Solução Final (Ataques: {attacks})")
    ax3.axis('off')

    # 4. Evolução do número de soluções encontradas
    ax4 = fig.add_subplot(2, 2, 4)
    if solutions_over_time:
        times, counts = zip(*solutions_over_time)
        ax4.plot(times, counts, marker='o')
        ax4.set_title("Descoberta das 92 Soluções")
        ax4.set_xlabel("Tempo (s)")
        ax4.set_ylabel("Soluções Únicas")
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, "Nenhuma solução acumulada", ha='center', va='center')
        ax4.set_title("Descoberta das 92 Soluções (vazio)")
        ax4.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# PARTE 2: CAIXEIRO VIAJANTE 3D (GENETIC ALGORITHM)
# Função para carregar dados do CAIXEIRO VIAJANTE a partir de CSV
def load_tsp_data(filename="CaixeiroGruposGA.csv", group_id=1):
    if not os.path.exists(filename):
        print(f"AVISO: Arquivo '{filename}' não encontrado. Usando dados aleatórios.")
        return None

    try:
        # Leitura do CSV
        data = np.loadtxt(filename, delimiter=',')
        
        # Colunas: [X, Y, Z, Grupo]
        coords = data[:, :3]
        groups = data[:, 3]
        
        # Filtra: Grupo escolhido OU Grupo 0 (Origem)
        mask = (groups == group_id) | (groups == 0)
        points = coords[mask]
        
        print(f"Arquivo carregado: {len(points)} pontos (Grupo {group_id} + Origem)")
        return points
        
    except Exception as e:
        print(f"Erro ao ler CSV: {e}. Usando aleatório.")
        return None

# Classe que implementa um GA para CAIXEIRO VIAJANTE em 3D com matriz de distâncias precomputada
class TSP3D_GA:
    def __init__(self, points=None, num_points=40, pop_size=100, mutation_rate=0.01, elitism=True):
        self.pop_size = pop_size
        self.mut_rate = mutation_rate
        self.elitism = elitism
        
        if points is not None:
            self.points = points
            self.num_points = len(points)
        else:
            # Gera pontos aleatórios no cubo [-30,30]^3 se não passar pontos externos
            self.points = np.random.uniform(-30, 30, (num_points, 3))
            self.num_points = num_points
            
        # Precompute matrix de distâncias para acelerar fitness
        diff = self.points[:, np.newaxis, :] - self.points[np.newaxis, :, :]
        self.dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Avalia vetor de permutações (pop) e retorna array de distâncias totais (menor = melhor)
    def fitness_vec(self, pop):
        scores = []
        for ind in pop:
            d = 0.0
            for i in range(self.num_points - 1): # soma distâncias entre cidades na ordem da permutação
                d += self.dist_matrix[ind[i], ind[i + 1]]
            d += self.dist_matrix[ind[-1], ind[0]]
            scores.append(d)
        return np.array(scores)

    # Crossover OX1: preserva segmento do pai1 e insere na ordem restante do pai2
    def crossover(self, p1, p2):
        size = self.num_points
        start, end = sorted(random.sample(range(size), 2))
        child = np.full(size, -1)
        child[start:end + 1] = p1[start:end + 1]
        mask = ~np.isin(p2, child)
        child[child == -1] = p2[mask]
        return child

    # Executa o GA por generations gerações e retorna melhor rota/distância/histórico e geração de convergência
    def run(self, generations=200):
        # População inicial: permutações aleatórias das cidades
        pop = np.array([np.random.permutation(self.num_points) for _ in range(self.pop_size)])
        history = []
        best_route, best_dist = None, float('inf')
        conv_gen = generations  # geração onde ocorreu a melhor atualização (inicial: última)
        patience = 0

        for g in range(generations):
            scores = self.fitness_vec(pop)
            min_idx = np.argmin(scores)

            # Atualiza melhor global
            if scores[min_idx] < best_dist:
                best_dist = scores[min_idx]
                best_route = pop[min_idx].copy()
                conv_gen = g
                patience = 0
            else:
                patience += 1

            history.append(best_dist)

            # Construção da nova população com elitismo e torneio como seleção
            new_pop = []
            if self.elitism:
                sorted_idx = np.argsort(scores)
                # Adiciona 2 melhores diretamente (elitismo de 2)
                new_pop.extend([pop[sorted_idx[0]], pop[sorted_idx[1]]])

            # Preenche restante da população
            while len(new_pop) < self.pop_size:
                # Torneio de 3 para escolher cada pai
                parents = []
                for _ in range(2):
                    cands = np.random.choice(self.pop_size, 3, replace=False)
                    best_c = cands[np.argmin(scores[cands])]
                    parents.append(pop[best_c])

                child = self.crossover(parents[0], parents[1])

                # Mutação por swap com probabilidade mut_rate
                if random.random() < self.mut_rate:
                    i, j = random.sample(range(self.num_points), 2)
                    child[i], child[j] = child[j], child[i]

                new_pop.append(child)

            pop = np.array(new_pop)

        return best_route, best_dist, history, conv_gen

# Plota o dashboard do CAIXEIRO VIAJANTE com 3 painéis
def plot_tsp_dashboard(route, points, h_elit, h_no_elit, conv_gens):
    fig = plt.figure(figsize=(15, 10))
    # Figura 5: suptitle principal
    fig.suptitle("ETAPA 2 - Dashboard Caixeiro Viajante 3D (GA)", fontsize=14)

    # 1. Rota 3D
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ord_pts = points[route]
    ord_pts = np.vstack([ord_pts, ord_pts[0]])
    ax1.plot(ord_pts[:, 0], ord_pts[:, 1], ord_pts[:, 2], marker='o', c='blue')
    ax1.scatter(ord_pts[0, 0], ord_pts[0, 1], ord_pts[0, 2], c='red', s=100, marker='*', label="Início/Fim")
    
    # Destaca Origem (0,0,0) se existir nos pontos
    origin_idx = np.where((points[:,0]==0) & (points[:,1]==0) & (points[:,2]==0))[0]
    if len(origin_idx) > 0:
        ox, oy, oz = points[origin_idx[0]]
        ax1.text(ox, oy, oz, " Origem", color='black', fontsize=10)

    ax1.set_title(f"Melhor Rota (Dist: {h_elit[-1]:.2f})")
    ax1.legend()

    # 2. Comparação Elitismo
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(h_elit, label="Com Elitismo", color='green')
    ax2.plot(h_no_elit, label="Sem Elitismo", linestyle='--', color='gray')
    ax2.set_title("Evolução do Custo")
    ax2.set_xlabel("Geração")
    ax2.set_ylabel("Distância")
    ax2.legend()
    ax2.grid(True)

    # 3. Histograma: gerações de convergência
    ax3 = fig.add_subplot(2, 2, 3)
    if conv_gens:
        ax3.hist(conv_gens, bins=10, alpha=0.7, color='purple')
        ax3.set_title("Histograma: Geração de Convergência")
        ax3.set_xlabel("Geração")
        ax3.set_ylabel("Frequência")
    else:
        ax3.text(0.5, 0.5, "Sem dados de convergência", ha='center', va='center')
        ax3.set_title("Histograma: Geração de Convergência (vazio)")
        ax3.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ROUTINA PRINCIPAL DA ETAPA 2
def main_etapa2():
    print(">>> Iniciando ETAPA 2: Otimização Combinatória")

    # --- 8 RAINHAS ---
    print("\n[8 RAINHAS] Buscando 92 soluções (aguarde)...")
    sa = EightQueensSA(temp_initial=1000, alpha=0.95)
    sols = set()
    sols_time = []
    t_start = time.time()

    while len(sols) < 92:
        sol, h, _, _, _ = sa.solve(max_iter=1000)
        
        # Se encontrou uma solução válida (h=0)
        if h == 0:
            tup = tuple(sol)
            # Se ela é nova (ainda não está no conjunto)
            if tup not in sols:
                sols.add(tup)
                sols_time.append((time.time() - t_start, len(sols)))
                print(f"  -> Solução única encontrada: {len(sols)}/92")
        
        # Timeout de segurança (40s)
        if time.time() - t_start > 40: 
            print(f"Timeout de 40s atingido. Encontradas {len(sols)} soluções.")
            break

    # Rodada de demonstração para gráficos e prints
    sol_demo, h_demo, h_ene, h_temp, t_exec = sa.solve(max_iter=3000)

    print(f"\n--- Prints Necessários (8 Rainhas) ---")
    print(f"Uma solução: {sol_demo}")
    print(f"f(x) (Energia): {h_demo}")
    print(f"Tempo Execução (rodada demo): {t_exec:.4f}s")
    print(f"Parâmetros SA: Temp={sa.temp_initial}, Alpha={sa.alpha}")

    plot_8queens_dashboard(sol_demo, h_demo, h_ene, h_temp, sols_time)

    # --- CAIXEIRO VIAJANTE 3D ---
    print("\n[CAIXEIRO VIAJANTE 3D] Executando GA...")
    
    tsp_points = load_tsp_data("CaixeiroGruposGA.csv", group_id=1)

    # Configuração dos GAs (com e sem elitismo)
    ga = TSP3D_GA(points=tsp_points, num_points=35, pop_size=100, elitism=True)
    ga_no = TSP3D_GA(points=ga.points, pop_size=100, elitism=False)

    # Execução sem elitismo para comparação
    _, _, h_no_elit, _ = ga_no.run(generations=200)

    # Estatísticas com elitismo (30 rodadas)
    gens_conv = []
    best_r, best_d, h_elit = None, float('inf'), []

    total_rodadas = 30 # Define quantas vezes vai rodar
    print(f"\nRodando {total_rodadas} vezes para estatísticas...")

    for i in range(total_rodadas):
        # Executa o GA
        r, d, h, g = ga.run(generations=200)
        print(f"  -> Rodada {i+1}/{total_rodadas} concluída. (Distância encontrada: {d:.4f})")
        gens_conv.append(g)
        
        # Verifica se é o melhor global
        if d < best_d:
            best_d = d
            best_r = r
            h_elit = h

    try:
        mode_gen = statistics.mode(gens_conv)
    except statistics.StatisticsError:
        mode_gen = "Dispersa"

    print(f"\n--- Prints Necessarios (CAIXEIRO VIAJANTE) ---")
    print(f"Melhor Rota (prefixo): {best_r[:10]}... (Total {len(best_r)})")
    print(f"Distância: {best_d:.4f}")
    print(f"Parâmetros GA: Pop={ga.pop_size}, Mut={ga.mut_rate}, Elitismo=True")
    print(f"Geração de Convergência (Moda): {mode_gen}")

    plot_tsp_dashboard(best_r, ga.points, h_elit, h_no_elit, gens_conv)

if __name__ == "__main__":
    main_etapa2()