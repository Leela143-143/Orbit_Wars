import time
import cProfile
import pstats
from htmrl_agent import agent_fn
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
import math
import random

# Generate mock data
player = 0
planets = []
fleets = []

# Generate 40 planets owned by player
for i in range(40):
    p = [
        i,             # id
        player,        # owner
        100 + i,       # x
        100 + i,       # y
        10.0,          # radius
        random.randint(50, 200), # ships
        1              # production
    ]
    planets.append(p)

# Generate some enemies and neutrals
for i in range(40, 60):
    owner = 1 if i % 2 == 0 else -1
    p = [
        i,
        owner,
        300 + i,
        300 - i,
        10.0,
        100,
        1
    ]
    planets.append(p)

# Generate massive fleet spam (Stress test for Late Game)
for i in range(500):
    owner = player if i % 2 == 0 else 1
    f = [
        1000 + i,      # id
        owner,         # owner
        200 + (i/10),  # x
        200 - (i/10),  # y
        math.pi / 4,   # angle
        0,             # from planet id
        50             # ships
    ]
    fleets.append(f)

# Mock observation dictionary as kaggle-environments passes
obs = {
    "step": 1,
    "player": player,
    "planets": planets,
    "fleets": fleets
}

def run_benchmark():
    start = time.time()
    agent_fn(obs)
    end = time.time()
    print(f"Time taken for 40 planets: {end - start:.4f} seconds")

if __name__ == "__main__":
    print("Running initial test (cold start, will initialize HTMRL)")
    run_benchmark()

    print("\nRunning warm test (HTMRL loaded in cache)")
    run_benchmark()

    print("\nProfiling warm test...")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(5):
        obs["step"] += 1
        agent_fn(obs)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(30)
