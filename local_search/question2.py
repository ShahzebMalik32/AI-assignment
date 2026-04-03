import random
import math

# ─────────────────────────────────────────────
# Landscape definition (1-indexed positions)
# ─────────────────────────────────────────────
LANDSCAPE = [5, 8, 6, 12, 9, 7, 17, 14, 10, 6, 19, 15, 11, 8]
# Index:       0  1  2   3  4  5   6   7   8  9  10  11  12 13
# State:       1  2  3   4  5  6   7   8   9 10  11  12  13 14

GLOBAL_MAX_STATE = 11  # 1-indexed (f=19)
GLOBAL_MAX_IDX   = 10  # 0-indexed

# ─────────────────────────────────────────────
# Q1 helper: First-Choice Hill Climbing
# ─────────────────────────────────────────────
def first_choice_hc(landscape, start):
    """
    Moves to the FIRST neighbour that strictly improves the current value.
    Returns (terminal_index, path_of_indices).
    """
    current = start
    path = [current]
    n = len(landscape)
    while True:
        improved = False
        neighbours = []
        if current > 0:
            neighbours.append(current - 1)
        if current < n - 1:
            neighbours.append(current + 1)
        for nb in neighbours:
            if landscape[nb] > landscape[current]:
                current = nb
                path.append(current)
                improved = True
                break
        if not improved:
            break
    return current, path


# ─────────────────────────────────────────────
# Q1 helper: Stochastic Hill Climbing
# ─────────────────────────────────────────────
def stochastic_hc(landscape, start):
    """
    Collects all improving neighbours; chooses one uniformly at random.
    Returns (terminal_index, path_of_indices).
    """
    current = start
    path = [current]
    n = len(landscape)
    while True:
        improving = []
        if current > 0 and landscape[current - 1] > landscape[current]:
            improving.append(current - 1)
        if current < n - 1 and landscape[current + 1] > landscape[current]:
            improving.append(current + 1)
        if not improving:
            break
        current = random.choice(improving)
        path.append(current)
    return current, path


# ─────────────────────────────────────────────
# Helper: find all local maxima
# ─────────────────────────────────────────────
def find_local_maxima(landscape):
    """
    Returns list of 0-based indices whose f-value strictly exceeds both neighbours.
    Edge positions are considered to have one neighbour only.
    """
    maxima = []
    n = len(landscape)
    for i in range(n):
        left_ok  = (i == 0)     or landscape[i] > landscape[i - 1]
        right_ok = (i == n - 1) or landscape[i] > landscape[i + 1]
        if left_ok and right_ok:
            maxima.append(i)
    return maxima


# ─────────────────────────────────────────────
# Q2(a): Random Restart Hill Climbing
# ─────────────────────────────────────────────
def random_restart_hc(landscape, num_restarts, variant='first_choice'):
    """
    Runs hill climbing from num_restarts random starting positions.

    Parameters
    ----------
    landscape    : list of int/float utility values
    num_restarts : int
    variant      : 'first_choice' | 'stochastic'

    Returns
    -------
    best_state   : 0-based index of the best terminal state found
    best_value   : f(best_state)
    all_results  : list of (start, terminal, path) tuples (all 0-based indices)
    """
    hc = first_choice_hc if variant == 'first_choice' else stochastic_hc

    best_state = None
    best_value = float('-inf')
    all_results = []

    for _ in range(num_restarts):
        start = random.randint(0, len(landscape) - 1)
        terminal, path = hc(landscape, start)
        val = landscape[terminal]
        all_results.append((start, terminal, path))
        if val > best_value:
            best_value = val
            best_state = terminal

    return best_state, best_value, all_results


# ─────────────────────────────────────────────
# Q2(b): Empirical probability experiment
# ─────────────────────────────────────────────
def empirical_probability(landscape, restart_counts, trials=100,
                          variant='first_choice', global_max_idx=GLOBAL_MAX_IDX):
    """
    For each n in restart_counts, runs 'trials' independent RRHC runs and
    returns the fraction that find the global maximum.
    """
    results = {}
    for n in restart_counts:
        successes = 0
        for _ in range(trials):
            _, _, all_res = random_restart_hc(landscape, n, variant)
            if any(terminal == global_max_idx for _, terminal, _ in all_res):
                successes += 1
        results[n] = successes / trials
    return results


def theoretical_probability(p, restart_counts):
    """P(find global max in n restarts) = 1 - (1-p)^n"""
    return {n: 1 - (1 - p) ** n for n in restart_counts}


def derive_p_first_choice(landscape, global_max_idx=GLOBAL_MAX_IDX):
    """
    Run first-choice HC from every starting state and count how many
    lead to the global maximum.  Returns p = count / len(landscape).
    """
    n = len(landscape)
    count = 0
    print("\n  First-Choice HC basin analysis (all starting states):")
    print(f"  {'Start (state)':>14} {'Terminal (state)':>16} {'Reaches Global Max?':>20}")
    print("  " + "-" * 54)
    for i in range(n):
        terminal, _ = first_choice_hc(landscape, i)
        reaches = terminal == global_max_idx
        if reaches:
            count += 1
        print(f"  {i+1:>14} {terminal+1:>16} {'Yes' if reaches else 'No':>20}")
    p = count / n
    print(f"\n  States reaching global max: {count}/{n}  =>  p = {p:.4f}")
    return p


# ─────────────────────────────────────────────
# Q2(c): Plateau investigation
# ─────────────────────────────────────────────
def rrhc_with_plateau_counter(landscape, num_restarts, variant='first_choice',
                               plateau_states=None, global_max_idx=GLOBAL_MAX_IDX):
    """
    Extended RRHC that additionally counts restarts terminating on a plateau state.

    Returns
    -------
    best_state, best_value, all_results, plateau_count, global_count
    """
    if plateau_states is None:
        plateau_states = []

    hc = first_choice_hc if variant == 'first_choice' else stochastic_hc

    best_state = None
    best_value = float('-inf')
    all_results = []
    plateau_count = 0
    global_count  = 0

    for _ in range(num_restarts):
        start = random.randint(0, len(landscape) - 1)
        terminal, path = hc(landscape, start)
        val = landscape[terminal]
        all_results.append((start, terminal, path))

        if terminal in plateau_states:
            plateau_count += 1
        if terminal == global_max_idx:
            global_count += 1

        if val > best_value:
            best_value = val
            best_state = terminal

    return best_state, best_value, all_results, plateau_count, global_count


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    random.seed(42)   # reproducibility

    # ── Print landscape info ──────────────────────────────────────────────
    print("=" * 65)
    print("  Q2 — RANDOM RESTART HILL CLIMBING")
    print("=" * 65)
    print("\nLandscape:")
    print(f"  {'State':>6}  {'f(s)':>6}")
    print("  " + "-" * 16)
    for i, v in enumerate(LANDSCAPE):
        print(f"  {i+1:>6}  {v:>6}")

    # ── Local maxima ──────────────────────────────────────────────────────
    lm = find_local_maxima(LANDSCAPE)
    print(f"\nLocal maxima (0-based indices): {lm}")
    print(f"Local maxima (states, 1-based): {[i+1 for i in lm]}")
    print(f"  Values: {[LANDSCAPE[i] for i in lm]}")
    print(f"  Global maximum: State {GLOBAL_MAX_IDX+1}, f = {LANDSCAPE[GLOBAL_MAX_IDX]}")

    # ─────────────────────────────────────────────────────────────────────
    # PART (a): Run RRHC with 20 restarts under both variants
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PART (a): RRHC with 20 restarts")
    print("=" * 65)

    for variant in ('first_choice', 'stochastic'):
        random.seed(42)
        best_state, best_value, all_results = random_restart_hc(
            LANDSCAPE, num_restarts=20, variant=variant)

        print(f"\n  Variant: {variant.upper().replace('_', '-')}")
        print(f"  {'Restart':>8}  {'Start':>6}  {'Terminal':>9}  {'f-value':>8}  {'Global Max?':>12}")
        print("  " + "-" * 50)
        for r_num, (start, terminal, _) in enumerate(all_results, 1):
            gm = "Yes" if terminal == GLOBAL_MAX_IDX else "No"
            print(f"  {r_num:>8}  {start+1:>6}  {terminal+1:>9}  "
                  f"{LANDSCAPE[terminal]:>8}  {gm:>12}")
        print(f"\n  => Best found: State {best_state+1}, f = {best_value}")

    # ─────────────────────────────────────────────────────────────────────
    # PART (b): Empirical vs theoretical probability
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PART (b): Empirical vs Theoretical probability")
    print("=" * 65)

    # Derive p from first-choice HC
    p = derive_p_first_choice(LANDSCAPE)

    restart_counts = [1, 3, 5, 10, 20]

    # Empirical
    random.seed(0)
    emp = empirical_probability(LANDSCAPE, restart_counts, trials=100,
                                variant='first_choice')

    # Theoretical
    theo = theoretical_probability(p, restart_counts)

    print("\n  Theoretical calculations (P = 1 - (1-p)^n):")
    for n in restart_counts:
        print(f"    n={n:>2}: P = 1 - (1 - {p:.4f})^{n} = 1 - {(1-p)**n:.6f} = {theo[n]:.6f}")

    print(f"\n  {'n':>4}  {'Empirical P':>12}  {'Theoretical P':>14}  {'Difference':>11}")
    print("  " + "-" * 46)
    for n in restart_counts:
        diff = emp[n] - theo[n]
        print(f"  {n:>4}  {emp[n]:>12.4f}  {theo[n]:>14.4f}  {diff:>+11.4f}")

    # ─────────────────────────────────────────────────────────────────────
    # PART (c): Plateau investigation
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  PART (c): Plateau investigation (states 7 & 8 both f=17)")
    print("=" * 65)

    LANDSCAPE_PLATEAU = LANDSCAPE.copy()
    LANDSCAPE_PLATEAU[7] = 17   # state 8 raised to 17 (was 14)
    # state 7 (index 6) already = 17

    PLATEAU_INDICES = [6, 7]    # 0-based indices for states 7 & 8

    print("\nModified landscape:")
    print(f"  {'State':>6}  {'Original f(s)':>14}  {'Modified f(s)':>14}  {'Changed?':>9}")
    print("  " + "-" * 48)
    for i, (orig, mod) in enumerate(zip(LANDSCAPE, LANDSCAPE_PLATEAU)):
        changed = "*" if orig != mod else ""
        print(f"  {i+1:>6}  {orig:>14}  {mod:>14}  {changed:>9}")

    NUM_RESTARTS = 20
    TRIALS       = 100

    # ── original landscape ───────────────────────────────────────────────
    random.seed(7)
    orig_global_rates = []
    orig_plateau_rates = []
    for _ in range(TRIALS):
        _, _, all_res, p_cnt, g_cnt = rrhc_with_plateau_counter(
            LANDSCAPE, NUM_RESTARTS, variant='first_choice',
            plateau_states=PLATEAU_INDICES, global_max_idx=GLOBAL_MAX_IDX)
        orig_global_rates.append(g_cnt / NUM_RESTARTS)
        orig_plateau_rates.append(p_cnt / NUM_RESTARTS)
    avg_orig_global  = sum(orig_global_rates)  / TRIALS
    avg_orig_plateau = sum(orig_plateau_rates) / TRIALS

    # ── modified (plateau) landscape ─────────────────────────────────────
    random.seed(7)
    mod_global_rates  = []
    mod_plateau_rates = []
    for _ in range(TRIALS):
        _, _, all_res, p_cnt, g_cnt = rrhc_with_plateau_counter(
            LANDSCAPE_PLATEAU, NUM_RESTARTS, variant='first_choice',
            plateau_states=PLATEAU_INDICES, global_max_idx=GLOBAL_MAX_IDX)
        mod_global_rates.append(g_cnt / NUM_RESTARTS)
        mod_plateau_rates.append(p_cnt / NUM_RESTARTS)
    avg_mod_global  = sum(mod_global_rates)  / TRIALS
    avg_mod_plateau = sum(mod_plateau_rates) / TRIALS

    # ── single representative run (for per-restart table) ────────────────
    random.seed(42)
    _, _, all_res_orig, pc_orig, gc_orig = rrhc_with_plateau_counter(
        LANDSCAPE, NUM_RESTARTS, 'first_choice', PLATEAU_INDICES, GLOBAL_MAX_IDX)
    _, _, all_res_mod,  pc_mod,  gc_mod  = rrhc_with_plateau_counter(
        LANDSCAPE_PLATEAU, NUM_RESTARTS, 'first_choice', PLATEAU_INDICES, GLOBAL_MAX_IDX)

    # Per-restart detail tables
    for label, results, p_cnt, g_cnt, land in [
        ("ORIGINAL landscape", all_res_orig, pc_orig, gc_orig, LANDSCAPE),
        ("MODIFIED landscape (states 7&8 both f=17)", all_res_mod, pc_mod, gc_mod, LANDSCAPE_PLATEAU),
    ]:
        print(f"\n  --- {label} ---")
        print(f"  {'Restart':>8}  {'Start':>6}  {'Terminal':>9}  {'f-value':>8}  "
              f"{'Global Max?':>12}  {'On Plateau?':>12}")
        print("  " + "-" * 66)
        for r_num, (start, terminal, _) in enumerate(results, 1):
            gm  = "Yes" if terminal == GLOBAL_MAX_IDX else "No"
            plt = "Yes" if terminal in PLATEAU_INDICES else "No"
            print(f"  {r_num:>8}  {start+1:>6}  {terminal+1:>9}  "
                  f"{land[terminal]:>8}  {gm:>12}  {plt:>12}")
        print(f"\n  Plateau terminations : {p_cnt}/{NUM_RESTARTS}")
        print(f"  Global max finds     : {g_cnt}/{NUM_RESTARTS}")

    # Summary comparison table
    print("\n  --- Summary over 100 independent RRHC(20) runs ---")
    print(f"  {'Landscape':>30}  {'Avg Global-Max Rate':>20}  {'Avg Plateau Rate':>17}")
    print("  " + "-" * 72)
    print(f"  {'Original':>30}  {avg_orig_global:>20.4f}  {avg_orig_plateau:>17.4f}")
    print(f"  {'Modified (plateau at 7&8)':>30}  {avg_mod_global:>20.4f}  {avg_mod_plateau:>17.4f}")


if __name__ == "__main__":
    main()