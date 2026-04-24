import math
import numpy as np
import pickle
import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
import HTMRL.spatial_pooler as spatial_pooler
import HTMRL.temporal_memory as temporal_memory

# 25 actions: 0=Do nothing, 1-12=50% ships in 12 directions, 13-24=100% ships in 12 directions
NUM_ACTIONS = 25
INPUT_SIZE = 40  # 12 enemy, 12 neutral, 12 friendly sectors + 4 ship bins

class OrbitWarsEncoder:
    def __init__(self, size=INPUT_SIZE):
        self.size = size

    def encode(self, my_planet, planets, player):
        # State array:
        # 0-11: Enemy in sector (12 directional slices)
        # 12-23: Neutral in sector
        # 24-35: Friendly in sector
        # 36-39: My ships (discretized to 4 bins)
        
        state = np.zeros(self.size, dtype=bool)
        
        # Discretize our ships into 4 bins (e.g. 0-25, 26-50, 51-75, 75+)
        ships_idx = min(3, int(my_planet.ships / 25))
        state[36 + ships_idx] = True
        
        for p in planets:
            if p.id == my_planet.id:
                continue
                
            # Calculate angle from our planet to the other planet
            angle = math.atan2(p.y - my_planet.y, p.x - my_planet.x)
            if angle < 0:
                angle += 2 * math.pi
                
            # Map angle to one of the 12 sectors
            sector = int((angle / (2 * math.pi)) * 12) % 12
            
            if p.owner == -1:
                state[12 + sector] = True
            elif p.owner == player:
                state[24 + sector] = True
            else:
                state[sector] = True
                
        return state

def encoding_to_action(encoding, actions, sp_size):
    buckets = np.floor(encoding / (float(sp_size) / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets, minlength=actions)
    return counts.argmax()

class HTMRLAgent:
    def __init__(self, load_path=None):
        self.encoder = OrbitWarsEncoder(INPUT_SIZE)
        if load_path and os.path.exists(load_path):
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict) and "sp" in data:
                    self.sp = data["sp"]
                    self.tm = data.get("tm", temporal_memory.TemporalMemory())
                else:
                    self.sp = data
                    self.tm = temporal_memory.TemporalMemory()
        else:
            self.sp = spatial_pooler.SpatialPooler(
                input_size=(INPUT_SIZE,), 
                acts_n=NUM_ACTIONS,
                cell_count=2048,
                active_count=40
            )
            self.tm = temporal_memory.TemporalMemory()
        self.tm_size = 2048 * 32
        self.tm_states = {}
        self.last_actions = {}
        self.last_states = {}

    def get_moves(self, obs, learn=False, reward=0):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        planets = [Planet(*p) for p in raw_planets]
        
        # Clear local timeline memory when a new match starts
        if step == 0:
            self.tm_states = {}
            
        my_planets = [p for p in planets if p.owner == player]
        enemies = [p for p in planets if p.owner != player and p.owner != -1]
        neutrals = [p for p in planets if p.owner == -1]
        
        # If learning, we can reinforce based on the total reward change
        # Wait, the SP reinforce takes action and reward. But we have multiple actions (one per planet).
        # We can just reinforce all actions taken in the last step with the same global reward.
        # But SpatialPooler in HTMRL only stores a single `_reinf_buf`.
        # To handle multiple planets, we should ideally have one SP or reinforce multiple times.
        # For simplicity, if learn=True, we reinforce the FIRST action taken.
        # Or better, we only use HTMRL for a SINGLE main planet's decision?
        # Let's reinforce the SP with the last state and action of one planet.
        
        if learn and self.last_states:
            # We just reinforce based on the first recorded state
            first_p_id = list(self.last_states.keys())[0]
            action = self.last_actions[first_p_id]
            state = self.last_states[first_p_id]
            
            # Manually set reinf_buf and reinforce
            activated_cols = self.sp._get_activated_cols(state)
            self.sp._reinf_buf = (state, activated_cols)
            self.sp.reinforce(action, reward)
            
        self.last_actions = {}
        self.last_states = {}

        for mine in my_planets:
            state = self.encoder.encode(mine, planets, player)
            sp_active_cols = self.sp.step(state, learn=False) # We'll do manual learning above
            
            if mine.id not in self.tm_states:
                from scipy.sparse import csr_matrix
                self.tm_states[mine.id] = {
                    "actives": csr_matrix((1, 2048 * 32), dtype=bool),
                    "winners": csr_matrix((1, 2048 * 32), dtype=bool),
                    "active_segs": csr_matrix((2048, 32 * 4), dtype=bool),
                    "matching_segs": csr_matrix((2048, 32 * 4), dtype=bool),
                    "matches_per_col": np.zeros((2048,)),
                    "actives_per_col": np.zeros((2048,)),
                    "active_pot_counts": [0] * (2048 * 32 * 4),
                    "actives_old_t": csr_matrix((1, 2048 * 32), dtype=bool).transpose().tocsr(),
                    "actives_old_perms": [0.0] * (2048 * 32),
                    "permanence_updates_buffer": [[], [], []],
                    "active_updates_buffer": [[], []],
                    "winner_updates_buffer": [[], []]
                }
                
            # INJECT STATE
            for k, v in self.tm_states[mine.id].items():
                setattr(self.tm, k, v)
                
            # STEP TM
            tm_actives = self.tm.step(sp_active_cols)
            
            # EXTRACT STATE
            for k in self.tm_states[mine.id].keys():
                self.tm_states[mine.id][k] = getattr(self.tm, k)
            
            if tm_actives.nnz > 0:
                action = encoding_to_action(tm_actives.indices, NUM_ACTIONS, self.tm_size)
            else:
                action = encoding_to_action(sp_active_cols, NUM_ACTIONS, self.sp.size)
            
            self.last_states[mine.id] = state
            self.last_actions[mine.id] = action
            
            # Execute action based on the 25 directional sectors
            if action == 0:
                continue
                
            # Actions 1 to 12 map to 50% ships, 13 to 24 map to 100% ships
            if action <= 12:
                ships = max(1, int(mine.ships * 0.5))
                sector = action - 1
            else:
                ships = mine.ships
                sector = action - 13
                
            # Convert sector back to angle in radians
            angle = (sector * (2 * math.pi / 12))
            
            moves.append([mine.id, angle, ships])
                
        return moves

_cached_agents = {}

def agent_fn(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    if player not in _cached_agents:
        _cached_agents[player] = HTMRLAgent(load_path="best_bot.pkl")
    return _cached_agents[player].get_moves(obs, learn=False)
