import numpy as np
import matplotlib.pyplot as plt

# --- 目標函數 ---
def target_function(x):
    return -15 * (np.sin(2 * x))**2 - (x - 2)**2 + 160

# --- 初始族群 ---
def initialize_population(pop_size):
    return np.random.uniform(-10, 10, size=pop_size)

# --- 交配操作 ---
def real_crossover_with_order(x1, x2):
    if x1 > x2:
        x1, x2 = x2, x1
    delta = np.random.uniform(-1, 1)
    diff = x1 - x2
    child1 = x1 + delta * diff
    child2 = x2 - delta * diff
    return np.clip(child1, -10, 10), np.clip(child2, -10, 10)

# --- 突變操作 ---
def scaled_uniform_mutation(x, mutation_rate, scale, generation, max_gen):
    if np.random.rand() < mutation_rate:
        noise = np.random.uniform(-1, 1)
        adaptive_scale = 2
        x += adaptive_scale * noise
        x = np.clip(x, -10, 10)
    return x

# --- GA 主程式 ---
def run_ga_with_correct_mating_rate(mode='direct', pop_size=10, mutation_rate=0.01,
                                    crossover_rate=0.8, max_gen=1000):
    pop = initialize_population(pop_size)
    history_best, history_avg = [], []

    x_grid = np.linspace(-10, 10, 10000)
    y_grid = target_function(x_grid)
    f_opt = y_grid.max()
    avg_above_152_flag = False

    for generation in range(max_gen):
        fx_values = target_function(pop)
        best_fx = fx_values.max()
        best_x = pop[np.argmax(fx_values)]
        avg_fx = fx_values.mean()

        history_best.append(best_fx)
        history_avg.append(avg_fx)

        if avg_fx > 152:
            avg_above_152_flag = True
        if avg_above_152_flag and abs(best_fx - f_opt) <= 1e-5:
            print(f"Converged at generation {generation} (mode={mode}), best_fx={best_fx:.5f}, best_x={best_x:.5f}")
            break

        if mode == 'direct':
            fitness = fx_values
        elif mode == 'square':
            fx_clipped = np.clip(fx_values, 1e-8, None)
            fitness = np.exp(4 * np.log(fx_clipped))
        elif mode == 'linear':
            fitness = 3 * fx_values + 50
        else:
            raise ValueError("Invalid mode")

        fitness -= fitness.min()
        fitness += 1e-6
        total_fit = fitness.sum()
        if total_fit < 1e-12 or np.isnan(total_fit) or not np.isfinite(total_fit):
            probs = np.ones_like(fitness) / len(fitness)
        else:
            probs = fitness / total_fit

        num_survivor = int((1 - crossover_rate) * pop_size)
        survivor_indices = np.random.choice(pop_size, size=num_survivor, replace=False, p=probs)
        survivors = pop[survivor_indices]

        num_children_needed = pop_size - num_survivor
        children = []
        while len(children) < num_children_needed:
            idx = np.random.choice(pop_size, size=2, replace=False, p=probs)
            x1, x2 = pop[idx[0]], pop[idx[1]]
            c1, c2 = real_crossover_with_order(x1, x2)
            c1 = scaled_uniform_mutation(c1, mutation_rate, 0.5, generation, max_gen)
            c2 = scaled_uniform_mutation(c2, mutation_rate, 0.5, generation, max_gen)
            children.extend([c1, c2])

        pop = np.concatenate([survivors, children[:num_children_needed]])

    return history_best, history_avg

# --- 畫出 f(x) 全貌圖 ---
x_plot = np.arange(-10, 10.001, 0.001)
y_plot = target_function(x_plot)
f_opt = y_plot.max()
x_opt = x_plot[np.argmax(y_plot)]

plt.figure(figsize=(10, 5))
plt.plot(x_plot, y_plot, label='f(x)')
plt.scatter(x_opt, f_opt, color='red', zorder=5, label=f'Max f(x) = {f_opt:.5f} at x = {x_opt:.4f}')
plt.title('Target Function f(x) with Global Maximum (dx = 0.001)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- 執行 GA 三種適應度模式 ---
results_correct_rate = {}
for mode in ['direct', 'square', 'linear']:
    best, avg = run_ga_with_correct_mating_rate(mode=mode, pop_size=10, mutation_rate=0.01,
                                                 crossover_rate=0.8, max_gen=1000)
    results_correct_rate[mode] = (best, avg)

# --- 收斂圖繪製 ---
for mode in results_correct_rate:
    best, avg = results_correct_rate[mode]
    plt.figure(figsize=(8, 4))
    plt.plot(best, label='Best')
    plt.plot(avg, label='Average')
    plt.xlabel('Generation')
    plt.ylabel('f(x)')
    plt.title(f'GA with 80% Mating Rate & 20% Survivor (mode={mode})')
    plt.legend()
    plt.grid(True)
    plt.show()