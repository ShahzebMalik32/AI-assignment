import random

# =====================================================
# Genetic Algorithm for f(x) = -x^2 + 14x + 5
# x in {0,...,15} using 4-bit binary encoding
# =====================================================


# -------------------------------
# Core Functions (Part a)
# -------------------------------

def decode(chromosome):
    return int("".join(map(str, chromosome)), 2)

def fitness(chromosome):
    x = decode(chromosome)
    return -x**2 + 14*x + 5

def roulette_select(population):
    total_fitness = sum(fitness(ind) for ind in population)

    if total_fitness == 0:
        return random.choice(population)

    r = random.random() * total_fitness

    cumulative = 0
    for ind in population:
        cumulative += fitness(ind)
        if cumulative >= r:
            return ind

    return population[-1]

def single_point_crossover(parent1, parent2, point):
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome, pm):
    return [1 - bit if random.random() < pm else bit for bit in chromosome]


# -------------------------------
# GA Implementation (Part b)
# -------------------------------

def random_chromosome():
    return [random.randint(0, 1) for _ in range(4)]

def run_ga(pop_size, num_generations, pm, elitism, verbose=True):
    population = [random_chromosome() for _ in range(pop_size)]
    history = []

    for gen in range(num_generations):

        decoded = [decode(ind) for ind in population]
        fitness_vals = [fitness(ind) for ind in population]

        best_idx = fitness_vals.index(max(fitness_vals))
        best_ind = population[best_idx]
        best_fit = fitness_vals[best_idx]
        best_x = decoded[best_idx]

        history.append((gen, best_fit, best_x))

        if verbose:
            print(f"\nGeneration {gen}")
            for i in range(pop_size):
                print(f"{population[i]} -> x={decoded[i]}, f={fitness_vals[i]}")
            print(f"Best: {best_ind}, x={best_x}, f={best_fit}")

        new_population = []

        # Elitism: keep best individual unchanged
        if elitism:
            new_population.append(best_ind.copy())

        while len(new_population) < pop_size:
            p1 = roulette_select(population)
            p2 = roulette_select(population)

            point = random.randint(1, 3)

            c1, c2 = single_point_crossover(p1, p2, point)

            c1 = mutate(c1, pm)
            c2 = mutate(c2, pm)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population

    return history


# -------------------------------
# Testing Core Functions (Part a)
# -------------------------------

def test_core_functions():
    print("\n--- Testing Core Functions ---")

    test_chromosomes = [
        [0,1,1,0],
        [1,0,0,1],
        [1,1,0,0],
        [0,0,1,1]
    ]

    for c in test_chromosomes:
        x = decode(c)
        f = fitness(c)
        print(f"Chromosome: {c}, x = {x}, fitness = {f}")


# -------------------------------
# Experiments (Part c)
# -------------------------------

def experiment_elitism():
    print("\n--- Elitism vs No Elitism ---")

    for elitism in [False, True]:
        best_fitnesses = []
        found_optimum = 0
        gen_to_50 = []

        for _ in range(30):
            results = run_ga(4, 20, 0.1, elitism, verbose=False)

            best_fit = max(r[1] for r in results)
            best_fitnesses.append(best_fit)

            if any(r[2] == 7 for r in results):
                found_optimum += 1

            for r in results:
                if r[1] >= 50:
                    gen_to_50.append(r[0])
                    break

        avg_fit = sum(best_fitnesses) / 30
        avg_gen = sum(gen_to_50)/len(gen_to_50) if gen_to_50 else None

        print(f"\nElitism = {elitism}")
        print(f"Average Best Fitness: {avg_fit}")
        print(f"Runs reaching x=7: {found_optimum}/30")
        print(f"Avg generation to reach f>=50: {avg_gen}")


def experiment_mutation_rates():
    print("\n--- Mutation Rate Experiment ---")

    for pm in [0.01, 0.1, 0.3, 0.5]:
        fits = []

        for _ in range(30):
            results = run_ga(4, 20, pm, False, verbose=False)
            fits.append(max(r[1] for r in results))

        avg_fit = sum(fits) / 30
        print(f"pm = {pm}, Average Best Fitness = {avg_fit}")

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":

    # Part (a)
    test_core_functions()

    # Part (b)
    print("\n--- Running GA (Required Run) ---")
    run_ga(pop_size=4, num_generations=10, pm=0.1, elitism=False, verbose=True)

    # Part (c)
    experiment_elitism()
    experiment_mutation_rates()