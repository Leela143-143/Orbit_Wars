import math
import numpy as np
import pickle
import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
import HTMRL.spatial_pooler as spatial_pooler
import HTMRL.temporal_memory as temporal_memory

from HTMRL.encoders import ScalarEncoder, CyclicEncoder, TileGeospatialEncoder
from HTMRL.decoders import action_decode
def encoding_to_action(encoding, actions, sp_size):
    buckets = np.floor(encoding / (float(sp_size) / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets, minlength=actions)
    return counts.argmax()


# 2.5% Sparsity rule
# My Ships: Size 1000, Active 25
# 3 Planet Channels: 2000 size (50 active) = 6,000
# 2 Fleet Channels: 4000 size (100 active) = 8,000
# Total INPUT_SIZE = 15000
INPUT_SIZE = 15000

class GlobalEmpireEncoder:
    def __init__(self):
        # We don't have "my_planet.ships" anymore since we encode globally
        # So we repurpose the first 1000 bits for "total empire ships"
        self.size = INPUT_SIZE
        self.empire_ships_encoder = ScalarEncoder(1000, 25, 0, 5000)
        self.geo_planet_encoder = TileGeospatialEncoder(2000, 50, is_fleet=False)
        self.geo_fleet_encoder = TileGeospatialEncoder(4000, 100, is_fleet=True)

    def encode(self, planets, fleets, player, center_x=500, center_y=500):
        state = np.zeros(self.size, dtype=bool)
        
        my_planets = [p for p in planets if p.owner == player]
        enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
        neutral_planets = [p for p in planets if p.owner == -1]

        enemy_fleets = [f for f in fleets if f.owner != player]
        friendly_fleets = [f for f in fleets if f.owner == player]

        total_ships = sum(p.ships for p in my_planets) + sum(f.ships for f in friendly_fleets)

        # 1. Total empire ships
        state[0:1000] = self.empire_ships_encoder.encode(total_ships)

        # 2. Encode Top-K Unions relative to map center
        offset = 1000
        state[offset:offset+2000] = self.geo_planet_encoder.encode_union_topk(center_x, center_y, enemy_planets)
        offset += 2000
        state[offset:offset+2000] = self.geo_planet_encoder.encode_union_topk(center_x, center_y, neutral_planets)
        offset += 2000
        state[offset:offset+2000] = self.geo_planet_encoder.encode_union_topk(center_x, center_y, my_planets)
        offset += 2000
        state[offset:offset+4000] = self.geo_fleet_encoder.encode_union_topk(center_x, center_y, enemy_fleets)
        offset += 4000
        state[offset:offset+4000] = self.geo_fleet_encoder.encode_union_topk(center_x, center_y, friendly_fleets)
        
        return state
def encoding_to_action(encoding, actions, sp_size):
    buckets = np.floor(encoding / (float(sp_size) / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets, minlength=actions)
    return counts.argmax()

class HTMRLAgent:
    def __init__(self, load_path=None):
        self.encoder = GlobalEmpireEncoder()
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
                active_count=41
            )
            self.tm = temporal_memory.TemporalMemory()
        self.tm_size = 2048 * 32
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
        
        if step == 0:
            self.tm.reset()
            
        my_planets = [p for p in planets if p.owner == player]
        if not my_planets:
            return moves

        if learn and self.last_states:
            # Simplistic reinforcement on the first state recorded
            first_p_id = list(self.last_states.keys())[0]
            action = self.last_actions[first_p_id]
            state = self.last_states[first_p_id]
            activated_cols = self.sp._get_activated_cols(state)
            self.sp._reinf_buf = (state, activated_cols)
            self.sp.reinforce(action, reward)
            
        self.last_actions = {}
        self.last_states = {}

        # The agent queries the map up to 10 times to form a strategy.
        # It updates its internal mock state of my_planets (ships left)
        # to ensure it doesn't double-spend ships.
        acted_planets = set()

        # Max moves bounded by smaller of 10 or number of owned planets
        max_queries = min(10, len(my_planets))

        # Create a mutable proxy for tracking ships since kaggle environment namedtuples are immutable
        mutable_ships = {p.id: p.ships for p in my_planets}

        for query_idx in range(max_queries):
            # 1. Encode global state (this uses the un-mutated base state, which is fine for the single turn)
            state = self.encoder.encode(planets, fleets, player)
            
            # 2. Process through Spatial Pooler and Temporal Memory
            sp_active_cols = self.sp.step(state, learn=False)
            tm_actives = self.tm.step(sp_active_cols)
            
            # 3. Decode action
            if tm_actives.nnz > 0:
                target_x, target_y, angle, ship_pct = action_decode(tm_actives.indices, self.sp.size, num_cells=32)
            else:
                target_x, target_y, angle, ship_pct = action_decode(sp_active_cols, self.sp.size, num_cells=None)
            
            if ship_pct < 0.1:
                # Agent decided not to act anymore
                break

            # 4. Find the closest un-acted planet to the target coordinates
            closest_planet = None
            closest_dist = float('inf')
            for p in my_planets:
                if p.id in acted_planets or mutable_ships[p.id] == 0:
                    continue
                dist = math.hypot(p.x - target_x, p.y - target_y)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_planet = p

            if not closest_planet:
                break

            acted_planets.add(closest_planet.id)
            
            # 5. Form the move
            ships_to_send = int(mutable_ships[closest_planet.id] * ship_pct)
            if ships_to_send == 0:
                continue
                
            moves.append([closest_planet.id, angle, ships_to_send])

            # 6. Update internal mock state for next query in loop
            mutable_ships[closest_planet.id] -= ships_to_send

            # Track state for potential learning
            if query_idx == 0:
                self.last_states[closest_planet.id] = state
                self.last_actions[closest_planet.id] = 0

        return moves

_cached_agents = {}

def agent_fn(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    if player not in _cached_agents:
        _cached_agents[player] = HTMRLAgent(load_path="best_bot.pkl")
    return _cached_agents[player].get_moves(obs, learn=False)
