import math
import numpy as np
import pickle
import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
import HTMRL.spatial_pooler as spatial_pooler
import HTMRL.temporal_memory as temporal_memory

from HTMRL.encoders import ScalarEncoder, CyclicEncoder, TileGeospatialEncoder
from HTMRL.decoders import action_decode

# 2.5% Sparsity rule
# My Ships: Size 1000, Active 25
# 3 Planet Channels: 2000 size (50 active) = 6,000
# 2 Fleet Channels: 4000 size (100 active) = 8,000
# Total INPUT_SIZE = 15000
INPUT_SIZE = 15000

class OrbitWarsEncoder:
    def __init__(self):
        self.size = INPUT_SIZE
        self.ships_encoder = ScalarEncoder(1000, 25, 0, 500)
        self.geo_planet_encoder = TileGeospatialEncoder(2000, 50, is_fleet=False)
        self.geo_fleet_encoder = TileGeospatialEncoder(4000, 100, is_fleet=True)

    def encode(self, my_planet, planets, fleets, player):
        state = np.zeros(self.size, dtype=bool)
        
        # 1. My ships
        state[0:1000] = self.ships_encoder.encode(my_planet.ships)

        # 2. Separate entities
        enemy_planets = [p for p in planets if p.owner != player and p.owner != -1 and p.id != my_planet.id]
        neutral_planets = [p for p in planets if p.owner == -1 and p.id != my_planet.id]
        friendly_planets = [p for p in planets if p.owner == player and p.id != my_planet.id]

        enemy_fleets = [f for f in fleets if f.owner != player]
        friendly_fleets = [f for f in fleets if f.owner == player]

        # 3. Encode Top-K Unions
        offset = 1000
        state[offset:offset+2000] = self.geo_planet_encoder.encode_union_topk(my_planet.x, my_planet.y, enemy_planets)
        offset += 2000
        state[offset:offset+2000] = self.geo_planet_encoder.encode_union_topk(my_planet.x, my_planet.y, neutral_planets)
        offset += 2000
        state[offset:offset+2000] = self.geo_planet_encoder.encode_union_topk(my_planet.x, my_planet.y, friendly_planets)
        offset += 2000
        state[offset:offset+4000] = self.geo_fleet_encoder.encode_union_topk(my_planet.x, my_planet.y, enemy_fleets)
        offset += 4000
        state[offset:offset+4000] = self.geo_fleet_encoder.encode_union_topk(my_planet.x, my_planet.y, friendly_fleets)
        
        return state
def encoding_to_action(encoding, actions, sp_size):
    buckets = np.floor(encoding / (float(sp_size) / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets, minlength=actions)
    return counts.argmax()

class HTMRLAgent:
    def __init__(self, load_path=None):
        self.encoder = OrbitWarsEncoder()
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
                acts_n=1,
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
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
        planets = [Planet(*p) for p in raw_planets]
        from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
        fleets = [Fleet(*f) for f in raw_fleets]
        
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
            state = self.encoder.encode(mine, planets, fleets, player)
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
                angle, ship_pct = action_decode(tm_actives.indices, self.sp.size, num_cells=32)
            else:
                angle, ship_pct = action_decode(sp_active_cols, self.sp.size, num_cells=None)
            
            self.last_states[mine.id] = state
            # For learning we only pass action 0 since we have acts_n=1
            self.last_actions[mine.id] = 0
            
            # ship_pct is continuous between 0 and 1
            # If ship_pct < 0.1 or we have no ships, do nothing
            if ship_pct < 0.1 or mine.ships == 0:
                continue
                
            ships = int(mine.ships * ship_pct)
            if ships == 0:
                continue
            
            moves.append([mine.id, angle, ships])


                
        return moves

_cached_agents = {}

def agent_fn(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    if player not in _cached_agents:
        _cached_agents[player] = HTMRLAgent(load_path="best_bot.pkl")
    return _cached_agents[player].get_moves(obs, learn=False)
