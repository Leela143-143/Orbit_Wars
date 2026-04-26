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
# Total INPUT_SIZE = 20000 (after adding velocity tracking)
INPUT_SIZE = 20000

class GlobalEmpireEncoder:
    def __init__(self):
        # We don't have "my_planet.ships" anymore since we encode globally
        # So we repurpose the first 1000 bits for "total empire ships"
        self.size = INPUT_SIZE
        self.empire_ships_encoder = ScalarEncoder(1000, 25, 0, 5000)
        self.geo_planet_encoder = TileGeospatialEncoder(3000, 75, is_fleet=False)
        self.geo_fleet_encoder = TileGeospatialEncoder(5000, 125, is_fleet=True)

    def encode(self, planets, fleets, player, velocity_map=None, center_x=50, center_y=50):
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
        state[offset:offset+3000] = self.geo_planet_encoder.encode_union_topk(center_x, center_y, enemy_planets, velocity_map)
        offset += 3000
        state[offset:offset+3000] = self.geo_planet_encoder.encode_union_topk(center_x, center_y, neutral_planets, velocity_map)
        offset += 3000
        state[offset:offset+3000] = self.geo_planet_encoder.encode_union_topk(center_x, center_y, my_planets, velocity_map)
        offset += 3000
        state[offset:offset+5000] = self.geo_fleet_encoder.encode_union_topk(center_x, center_y, enemy_fleets, velocity_map)
        offset += 5000
        state[offset:offset+5000] = self.geo_fleet_encoder.encode_union_topk(center_x, center_y, friendly_fleets, velocity_map)
        
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
        self.planet_history = {} # Tracks previous planet positions to compute velocity

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
            self.planet_history = {}
            self.last_empire_value = 0.0
            self.last_total_ships = 0
            self.last_num_planets = 0
            
        # Compute velocities for planets based on history
        velocity_map = {}
        for p in planets:
            if p.id in self.planet_history:
                old_x, old_y = self.planet_history[p.id]
                velocity_map[p.id] = (p.x - old_x, p.y - old_y)
            else:
                velocity_map[p.id] = (0.0, 0.0)
            self.planet_history[p.id] = (p.x, p.y)

        my_planets = [p for p in planets if p.owner == player]

        # --- Biological Reward Calculation ---
        my_ships = sum(p.ships for p in my_planets) + sum(f.ships for f in fleets if f.owner == player)
        my_prod = sum(p.production for p in my_planets)

        strongest_enemy_ships = 0
        strongest_enemy_prod = 0
        for e in range(1, 4):
            if e == player: continue
            e_ships = sum(p.ships for p in planets if p.owner == e) + sum(f.ships for f in fleets if f.owner == e)
            e_prod = sum(p.production for p in planets if p.owner == e)
            if e_ships > strongest_enemy_ships:
                strongest_enemy_ships = e_ships
                strongest_enemy_prod = e_prod

        # Base Empire Value (Survival signal)
        empire_value = (my_ships - strongest_enemy_ships) + 10 * (my_prod - strongest_enemy_prod)
        continuous_reward = empire_value - getattr(self, 'last_empire_value', 0.0)
        self.last_empire_value = empire_value

        # Dopamine Hit (Captured a planet)
        num_planets = len(my_planets)
        if num_planets > getattr(self, 'last_num_planets', 0):
            continuous_reward += 100.0 # Huge positive spike for conquering
        self.last_num_planets = num_planets

        # Pain Signal (Ships destroyed unexpectedly - void or sun)
        # Production naturally adds `my_prod` ships per turn. If we lost more than we produced,
        # and it wasn't due to combat (which affects empire_value), it's bad.
        # Empire value mostly handles combat, but explicit pain for pure loss speeds up avoidance.
        expected_ships = getattr(self, 'last_total_ships', 0) + my_prod
        if step > 1 and my_ships < expected_ships:
            # We lost ships. Check if it was a combat trade (enemy ships also dropped)
            # If not combat, we hit the sun or void. Apply sharp penalty.
            loss = expected_ships - my_ships
            continuous_reward -= loss * 0.5

        self.last_total_ships = my_ships
        # --- End Reward Calculation ---

        if not my_planets:
            return moves
            
        # The SP runs continuously in unsupervised mode to build stable representations
        if learn:
            pass # The SP will learn its representation naturally during step()

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
            state = self.encoder.encode(planets, fleets, player, velocity_map)
            
            # 2. Process through Spatial Pooler and Temporal Memory
            sp_active_cols = self.sp.step(state, learn=learn)
            tm_actives = self.tm.step(sp_active_cols, reward=continuous_reward if learn else 0.0)
            
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

            # Metabolic cost: penalize sending ships very slightly to prevent pointless spam
            if learn:
                continuous_reward -= ships_to_send * 0.05

            # 6. Update internal mock state for next query in loop
            mutable_ships[closest_planet.id] -= ships_to_send

        return moves

_cached_agents = {}

def agent_fn(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    if player not in _cached_agents:
        _cached_agents[player] = HTMRLAgent(load_path="best_bot.pkl")
    return _cached_agents[player].get_moves(obs, learn=False)
