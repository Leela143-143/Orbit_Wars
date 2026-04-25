import time
from kaggle_environments import make
from htmrl_agent import agent_fn

env = make("orbit_wars", debug=True, configuration={"actTimeout": 60.0, "episodeSteps": 500})

obs = env.reset()
t0 = time.time()
print("Running full game...")
while not env.done:
    actions = [agent_fn(obs[0].observation), agent_fn(obs[1].observation)]
    obs = env.step(actions)

print("Done in", time.time() - t0)
