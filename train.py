import os
import sys
import pickle
import random
import json
import multiprocessing
from tqdm import tqdm

# Suppress annoying kaggle_environments initialization warnings
_old_stdout = sys.stdout
_old_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')
try:
    from kaggle_environments import make
    from htmrl_agent import HTMRLAgent
finally:
    sys.stdout = _old_stdout
    sys.stderr = _old_stderr

NUM_BOTS = 20
MATCHES = 2000

class BotManager:
    def __init__(self, bot_id):
        self.bot_id = bot_id
        self.filepath = f"bots/bot_{bot_id}.pkl"
        self.elo = 1000

    def init_disk(self):
        # Create default empty agent if not exists
        if not os.path.exists(self.filepath):
            agent = HTMRLAgent()
            with open(self.filepath, "wb") as f:
                pickle.dump({"sp": agent.sp, "tm": agent.tm}, f)

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, score_a, k=32):
    expected_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - expected_a))
    return new_a, new_b

def play_match_worker(args):
    bot1_id, bot1_filepath, bot2_id, bot2_filepath = args
    
    agent1 = HTMRLAgent(bot1_filepath)
    agent2 = HTMRLAgent(bot2_filepath)
    
    env = make("orbit_wars", debug=False)
    state = env.reset(num_agents=2)
    done = False
    
    while not done:
        obs1 = state[0].observation
        obs2 = state[1].observation
        
        act1 = agent1.get_moves(obs1, learn=True, reward=0)
        act2 = agent2.get_moves(obs2, learn=True, reward=0)
        
        state = env.step([act1, act2])
        
        if state[0].status == 'DONE' or state[1].status == 'DONE':
            done = True
            
    reward1 = state[0].reward
    reward2 = state[1].reward
    
    # Final reinforcement
    agent1.get_moves(state[0].observation, learn=True, reward=reward1 if reward1 else -1)
    agent2.get_moves(state[1].observation, learn=True, reward=reward2 if reward2 else -1)
    
    # Save updated brains back to disk
    with open(bot1_filepath, "wb") as f:
        pickle.dump({"sp": agent1.sp, "tm": agent1.tm}, f)
    with open(bot2_filepath, "wb") as f:
        pickle.dump({"sp": agent2.sp, "tm": agent2.tm}, f)
        
    return bot1_id, bot2_id, reward1, reward2

def train():
    os.makedirs("bots", exist_ok=True)
    
    bots = [BotManager(i) for i in range(NUM_BOTS)]
    
    # Ensure all bots exist on disk before multiprocessing
    print("Initializing bots on disk...")
    for b in bots:
        b.init_disk()
        
    if os.path.exists("elo.json"):
        with open("elo.json", "r") as f:
            elos = json.load(f)
            for i, elo in enumerate(elos):
                bots[i].elo = elo

    cores = multiprocessing.cpu_count()
    print(f"\nStarting parallel training on {cores} CPU cores...")
    
    pool = multiprocessing.Pool(processes=cores)
    
    matches_run = 0
    pbar = tqdm(total=MATCHES, desc="Training")
    try:
        while matches_run < MATCHES:
            tasks = []
            busy = set()  # Track bots already assigned in this batch
            # Generate a batch of matches equal to number of cores
            bots.sort(key=lambda b: b.elo)
            for _ in range(cores):
                if matches_run + len(tasks) >= MATCHES:
                    break
                
                # Find two bots that are not busy and not the same
                available = [i for i in range(NUM_BOTS) if bots[i].bot_id not in busy]
                if len(available) < 2:
                    break  # Not enough free bots for another match
                    
                idx1 = random.choice(available)
                # Pick a nearby opponent (by Elo rank) that isn't the same bot
                candidates = [i for i in available if i != idx1]
                # Prefer close Elo neighbors
                nearby = [i for i in candidates if abs(i - idx1) <= 5]
                if nearby:
                    idx2 = random.choice(nearby)
                else:
                    idx2 = random.choice(candidates)
                
                busy.add(bots[idx1].bot_id)
                busy.add(bots[idx2].bot_id)
                tasks.append((bots[idx1].bot_id, bots[idx1].filepath, bots[idx2].bot_id, bots[idx2].filepath))
                
            # Execute matches in parallel
            results = pool.map(play_match_worker, tasks)
            
            for r in results:
                b1_id, b2_id, rew1, rew2 = r
                b1 = next(b for b in bots if b.bot_id == b1_id)
                b2 = next(b for b in bots if b.bot_id == b2_id)
                
                if rew1 > rew2: score1 = 1
                elif rew2 > rew1: score1 = 0
                else: score1 = 0.5
                    
                b1.elo, b2.elo = update_elo(b1.elo, b2.elo, score1)
                matches_run += 1
                pbar.update(1)
                
            # Update progress bar with top 5 bots
            top_bots = sorted(bots, key=lambda b: b.elo, reverse=True)[:5]
            top_str = " | ".join([f"B{b.bot_id}:{b.elo:.0f}" for b in top_bots])
            pbar.set_postfix_str(f"Top 5: {top_str}")
                
            with open("elo.json", "w") as f:
                json.dump([b.elo for b in sorted(bots, key=lambda b: b.bot_id)], f)
                
        pbar.close()
        pool.close()
        pool.join()
                
        best_bot = max(bots, key=lambda b: b.elo)
        print(f"\nTraining complete. Best bot is Bot {best_bot.bot_id} with Elo {best_bot.elo:.0f}")
        
        import shutil
        shutil.copyfile(best_bot.filepath, "best_bot.pkl")
        print("Saved best bot to best_bot.pkl")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Shutting down completely...")
        pbar.close()
        pool.terminate()
        pool.join()
        
        # Optionally, save the best bot we have so far
        best_bot = max(bots, key=lambda b: b.elo)
        print(f"Interrupted. Current best bot is Bot {best_bot.bot_id} with Elo {best_bot.elo:.0f}")
        import shutil
        shutil.copyfile(best_bot.filepath, "best_bot.pkl")
        print("Saved current best bot to best_bot.pkl")
        
        import sys
        sys.exit(0)

if __name__ == "__main__":
    train()
