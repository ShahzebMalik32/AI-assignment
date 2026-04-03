import random

# =====================================================
# DATA
# =====================================================

demands = [
    12,45,23,67,34,19,
    56,38,72,15,49,61,
    27,83,41,55,30,77,
    64,18,52,39,71,26,
    44,91,33,58,22,85,
    16,69,47,74,31,53
]

GRID_SIZE = 6
NUM_ZONES = 36
NUM_DRIVERS = 10
PENALTY = 5


# =====================================================
# (a) FOUNDATION
# =====================================================

def state_fitness(state):
    return sum(demands[i] for i in state) - PENALTY * NUM_DRIVERS


def random_state():
    return set(random.sample(range(NUM_ZONES), NUM_DRIVERS))


def get_neighbours(state):
    neighbours = []

    for s in state:
        for z in range(NUM_ZONES):
            if z not in state:
                new_state = set(state)
                new_state.remove(s)
                new_state.add(z)
                neighbours.append(new_state)

    return neighbours


def test_part_a():
    print("\n--- Part (a) ---")
    for _ in range(3):
        s = random_state()
        neigh = get_neighbours(s)
        print("State:", s)
        print("Fitness:", state_fitness(s))
        print("Neighbours:", len(neigh))
        print()


# =====================================================
# (b) HILL CLIMBING + RRHC
# =====================================================

def hc_driver(state, variant="first"):
    current = state
    current_fit = state_fitness(current)
    steps = 0

    while True:
        neighbours = get_neighbours(current)

        if variant == "first":
            improved = False
            for n in neighbours:
                f = state_fitness(n)
                if f > current_fit:
                    current = n
                    current_fit = f
                    improved = True
                    break
            if not improved:
                break

        elif variant == "stochastic":
            better = [n for n in neighbours if state_fitness(n) > current_fit]
            if not better:
                break
            current = random.choice(better)
            current_fit = state_fitness(current)

        steps += 1

    return current, current_fit, steps


def rrhc_driver(num_restarts, variant="stochastic"):
    best_state = None
    best_fit = float("-inf")
    history = []

    for _ in range(num_restarts):
        s = random_state()
        final_state, final_fit, _ = hc_driver(s, variant)

        history.append(final_fit)

        if final_fit > best_fit:
            best_fit = final_fit
            best_state = final_state

    return best_state, best_fit, history


def print_state_coords(state):
    coords = [(i // GRID_SIZE, i % GRID_SIZE) for i in state]
    print(coords)


# =====================================================
# (c) GENETIC ALGORITHM
# =====================================================

def ga_fitness(chrom):
    return sum(demands[i] for i in chrom) - PENALTY * NUM_DRIVERS


def random_chromosome():
    return sorted(random.sample(range(NUM_ZONES), NUM_DRIVERS))


def ordered_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))

    child = [None]*size

    # copy slice
    child[a:b] = p1[a:b]

    # fill remaining from p2
    p2_vals = [x for x in p2 if x not in child]

    j = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_vals[j]
            j += 1

    return sorted(child)


def ga_mutate(chrom, pm):
    if random.random() < pm:
        chrom = chrom.copy()
        remove_idx = random.randint(0, NUM_DRIVERS-1)

        available = [i for i in range(NUM_ZONES) if i not in chrom]
        new_val = random.choice(available)

        chrom[remove_idx] = new_val
        chrom = sorted(list(set(chrom)))

        # ensure size 10
        while len(chrom) < NUM_DRIVERS:
            extra = random.choice(available)
            if extra not in chrom:
                chrom.append(extra)

        return sorted(chrom)

    return chrom


def tournament_select(pop):
    k = 3
    sample = random.sample(pop, k)
    return max(sample, key=ga_fitness)


def run_driver_ga(pop_size, generations, pm):
    population = [random_chromosome() for _ in range(pop_size)]

    for gen in range(generations):
        new_pop = []

        for _ in range(pop_size):
            p1 = tournament_select(population)
            p2 = tournament_select(population)

            child = ordered_crossover(p1, p2)
            child = ga_mutate(child, pm)

            new_pop.append(child)

        population = new_pop

    best = max(population, key=ga_fitness)

    print("\n--- GA RESULT ---")
    print("Best Chromosome:", best)
    print("Fitness:", ga_fitness(best))
    print("Coordinates:")
    print_state_coords(best)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    # (a)
    test_part_a()

    # (b)
    print("\n--- RRHC ---")
    best_state, best_fit, history = rrhc_driver(30, variant="stochastic")

    print("Best State:", best_state)
    print("Fitness:", best_fit)
    print("Coordinates:")
    print_state_coords(best_state)
    print("Per-restart fitness:", history)

    # (c)
    run_driver_ga(pop_size=30, generations=100, pm=0.1)