import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    return -15 * (np.sin(2 * x))**2 - (x - 2)**2 + 160

def initialize_population(pop_size):
    return np.random.uniform(-10, 10, size=pop_size)

def plot_target_function():
    x_grid = np.arange(-10, 10.001, 0.001)
    y_grid = target_function(x_grid)
    f_opt = y_grid.max()
    x_opt = x_grid[np.argmax(y_grid)]

    plt.figure(figsize=(10, 5))
    plt.plot(x_grid, y_grid, label='f(x)')
    plt.scatter(x_opt, f_opt, color='red', zorder=5, label=f'Max f(x) = {f_opt:.5f} at x = {x_opt:.4f}')
    plt.title('Target Function f(x) with Global Maximum (dx = 0.001)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run_ea_final(pop_size=10, crossover_rate=0.8, mutation_rate=0.01,
                 max_gen=1000, mode='direct'):
    pop = initialize_population(pop_size)
    history_best, history_avg = [], []

    x_grid = np.arange(-10, 10.001, 0.001)
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
        if avg_above_152_flag and abs(best_fx - f_opt) <= 0.0004:
            print(f"[EA-{mode}] Converged at generation {generation}, best_fx={best_fx:.5f}, best_x={best_x:.5f}")
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
        probs = fitness / fitness.sum()

        num_offspring = int(pop_size * crossover_rate)
        children = []
        while len(children) < num_offspring:
            i, j = np.random.choice(pop_size, 2, replace=False, p=probs)
            xj, xk = pop[i], pop[j]
            r = np.random.uniform(0, 1)
            child = r * xj + (1 - r) * xk
            children.append(np.clip(child, -10, 10))

        num_survivors = pop_size - num_offspring
        survivor_indices = np.random.choice(pop_size, size=num_survivors, replace=False, p=probs)
        survivors = pop[survivor_indices]

        for i in range(len(children)):
            if np.random.rand() < mutation_rate:
                r = np.random.uniform(0, 1)
                d = np.random.uniform(-1, 1)
                children[i] += r * d
                children[i] = np.clip(children[i], -10, 10)

        pop = np.concatenate([survivors, np.array(children)])

    return history_best, history_avg

# Step 1: 畫出 f(x) 圖
plot_target_function()

# Step 2: 執行 EA 並畫結果
results_fused = {}
for mode in ['direct', 'square', 'linear']:
    best, avg = run_ea_final(mode=mode)
    results_fused[mode] = (best, avg)

for mode in results_fused:
    best, avg = results_fused[mode]
    plt.figure(figsize=(8, 4))
    plt.plot(best, label='Best')
    plt.plot(avg, label='Average')
    plt.xlabel('Generation')
    plt.ylabel('f(x)')
    plt.title(f'Convex+Mutation (mode={mode})')
    plt.legend()
    plt.grid(True)
    plt.show()