import math
import numpy as np
import statistics as stats
from collections import deque, namedtuple
import pickle
import os

# Dummy log to replace HTMRL.log dependency for standalone execution
class log:
    @staticmethod
    def has_debug(): return False
    @staticmethod
    def has_trace(): return False
    @staticmethod
    def debug(*args): pass
    @staticmethod
    def trace(*args): pass

# === SPATIAL POOLER CORE ===
class SpatialPooler:
    def __init__(self, input_size, acts_n, boost_strength=1.0, reward_scaled_reinf=True, boost_scaled_reinf=False,
                 only_reinforce_selected=True, normalize_rewards=True, cell_count=2048, active_count=41, boost_until=0,
                 reward_window=1000):
        self.input_size = input_size
        self.input_size_flat = np.prod(input_size)
        self.i = 0
        self.size = max(2, math.floor(cell_count / acts_n)) * acts_n
        self.stimulus_thresh = 0
        self.active_columns_count = active_count
        self.init_synapse_count = min(5 * self.active_columns_count, int(self.input_size_flat * 0.5))
        self.init_synapse_count = max(1, self.init_synapse_count)
        self.connected_perm_thresh = 0.5
        self.perm_inc_step = 0.05
        self.perm_dec_step = 0.0
        self.perm_min = 0.01
        self.perm_max = 1.01
        self.acts_n = acts_n
        assert (self.size % self.acts_n == 0)
        self.cells_per_act = int(self.size / self.acts_n)
        self.active_duty_cycles = np.zeros(self.size)
        self.boost_strength = boost_strength
        self.boost_factors = np.ones(self.size, dtype=np.float32)
        self.boost_anneal_until = boost_until
        self.boost_strength_init = boost_strength
        self.permanences = self._get_initialized_permanences()
        self.synapse_reinf_coeffs = np.zeros((self.input_size_flat, self.size), dtype=float)
        self.discount = 0.0
        self._tie_break_scale = 0.00001
        self._tie_breaker = np.random.rand(self.size) * self._tie_break_scale
        self.reward_scaled_reinf = reward_scaled_reinf
        self.boost_scaled_reinf = boost_scaled_reinf
        self.only_reinforce_selected = only_reinforce_selected
        self.normalize_rewards = normalize_rewards
        self._rewards = deque(maxlen=reward_window)
        self._reinf_buf = None

    def _get_initialized_permanences(self):
        permanences = np.empty((self.input_size_flat, self.size), dtype=float)
        permanences[:, :] = np.nan
        for col in range(self.size):
            rand_selection = np.random.choice(self.input_size_flat, self.init_synapse_count, replace=False)
            permanences[rand_selection, col] = self._get_initialized_segment()
        return permanences

    def _get_initialized_segment(self):
        vals = np.zeros((self.init_synapse_count,), dtype=float)
        is_actives = [randval > 0.5 for randval in np.random.random(self.init_synapse_count)]
        for i in range(self.init_synapse_count):
            if is_actives[i]:
                vals[i] = self.connected_perm_thresh + (self.perm_max - self.connected_perm_thresh) * np.random.random()
            else:
                vals[i] = self.connected_perm_thresh * np.random.random()
        return vals

    def _perms_to_activateds(self, inputs, perms):
        connecteds = np.array((perms - self.connected_perm_thresh).clip(min=0), dtype=bool) * (
            ~ np.isnan(self.permanences))
        conn_counts = np.dot(np.expand_dims(inputs, 0), np.array(connecteds, dtype=int))
        conn_counts = np.squeeze(conn_counts)
        conn_counts = np.add(conn_counts, self._tie_breaker, casting='unsafe')
        activated = np.argpartition(- conn_counts, self.active_columns_count)[:self.active_columns_count, ]
        return activated

    def _get_activated_cols(self, inputs):
        if self.boost_strength:
            boost_perms = self.permanences * self.boost_factors
        else:
            boost_perms = self.permanences
        activated = self._perms_to_activateds(inputs, boost_perms)
        return activated

    def _reinforce(self, inputs, activated, action, reward):
        action_range = (self.cells_per_act * action, self.cells_per_act * (action + 1))
        inputs_pos = inputs * self.perm_inc_step
        inputs_neg = (inputs - 1) * self.perm_dec_step
        inputs_shift = inputs_pos + inputs_neg
        inputs_shift = np.expand_dims(inputs_shift, 1)
        if self.reward_scaled_reinf:
            inputs_shift *= reward
        activated = [a for a in activated if action_range[0] <= a < action_range[1]]
        inactivated = [a for a in activated if not action_range[0] <= a < action_range[1]]

        if self.discount > 0.0:
            self.synapse_reinf_coeffs *= self.discount
            self.synapse_reinf_coeffs[np.ix_(np.nonzero(inputs)[0].tolist(), activated)] += 1.0
            self.synapse_reinf_coeffs = self.synapse_reinf_coeffs.clip(max=2.0)
            self.permanences = self.permanences + self.synapse_reinf_coeffs * inputs_shift
        else:
            self.permanences[:, activated] = self.permanences[:, activated] + inputs_shift

        if not self.only_reinforce_selected:
            self.permanences[:, inactivated] = self.permanences[:, inactivated] - inputs_shift

        self.permanences = self.permanences.clip(min=self.perm_min, max=self.perm_max)

        if self.discount > 0.0 and reward == 1.0:
            self.synapse_reinf_coeffs = np.zeros((self.input_size_flat, self.size))

    def _get_normalized_boost(self):
        mean = stats.mean(self.boost_factors)
        stdev = stats.stdev(self.boost_factors)
        stdev = 1 if stdev == 0 else stdev
        return (self.boost_factors - mean) / stdev

    def reinforce(self, action, reward):
        self._rewards.append(reward)
        mean = stats.mean(self._rewards)
        stdev = stats.stdev(self._rewards) if len(self._rewards) > 1 else 1
        stdev = 1 if stdev == 0 else stdev
        if self.normalize_rewards:
            reward = (reward - mean) / stdev
        (inputs, activated_cols) = self._reinf_buf
        self._reinf_buf = None
        self._reinforce(inputs, activated_cols, action, reward)

    def _updateDutyCycle(self, activated_cols):
        if self.boost_anneal_until > 0:
            self.boost_strength = max(0, self.boost_strength_init * (
                        self.boost_anneal_until - self.i) / self.boost_anneal_until)
        cols_dense = np.zeros(self.size, dtype=np.float32)
        cols_dense[activated_cols] = 1.0
        period = 1000 if self.i >= 1000 else self.i + 1
        self.active_duty_cycles = ((period - 1.) * self.active_duty_cycles + cols_dense) / float(period)

        self.boost_factors = np.exp(
            (self.active_columns_count / float(self.size) - self.active_duty_cycles) * self.boost_strength)

    def _init_next_step(self):
        self.i += 1
        self._tie_breaker = np.random.rand(self.size) * self._tie_break_scale

    def step(self, inputs, learn=True):
        activated_cols = self._get_activated_cols(inputs)
        if learn:
            self._reinf_buf = (inputs, activated_cols)
        self._updateDutyCycle(activated_cols)
        self._init_next_step()
        return activated_cols

from collections import namedtuple
import math
Planet = namedtuple('Planet', ['id', 'owner', 'x', 'y', 'radius', 'ships', 'production'])

# --- ENCODERS ---



class ScalarEncoder:
    """
    Encodes a continuous numerical value into an overlapping SDR.
    Similar values will share active bits.
    """
    def __init__(self, size, active_bits, min_val, max_val):
        self.size = size
        self.active_bits = active_bits
        self.min_val = min_val
        self.max_val = max_val
        self.range = max(1e-6, max_val - min_val) # Prevent div by 0
        self.buckets = size - active_bits + 1

    def encode(self, value):
        state = np.zeros(self.size, dtype=bool)
        v = min(max(value, self.min_val), self.max_val)
        bucket = int((v - self.min_val) / self.range * (self.buckets - 1))
        bucket = min(max(bucket, 0), self.buckets - 1)
        state[bucket : bucket + self.active_bits] = True
        return state

class CyclicEncoder:
    """
    Encodes an angle/cyclic value into an overlapping SDR.
    """
    def __init__(self, size, active_bits, min_val=0.0, max_val=2*math.pi):
        self.size = size
        self.active_bits = active_bits
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val

    def encode(self, value):
        state = np.zeros(self.size, dtype=bool)
        norm_val = ((value - self.min_val) % self.range) / self.range
        if norm_val < 0:
            norm_val += 1.0
        center_idx = int(norm_val * self.size)
        half_active = self.active_bits // 2
        for i in range(self.active_bits):
            idx = (center_idx - half_active + i) % self.size
            state[idx] = True
        return state

class TileGeospatialEncoder:
    def __init__(self, size=2000, active_bits=50, is_fleet=False):
        self.size = size
        self.active_bits = active_bits
        self.is_fleet = is_fleet

        if self.is_fleet:
            # Fleet: 4000 bits total, 100 active
            # 1. Positional Angle (1000 bits)
            # 2. Distance (1000 bits)
            # 3. Positional Angle Phase-Shifted (1000 bits)
            # 4. Flight Heading / Trajectory (1000 bits)
            self.angle_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.dist_encoder = ScalarEncoder(1000, 25, 0, 150)
            self.angle_encoder_phase = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.heading_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
        else:
            # Planet: 2000 bits total, 50 active
            self.angle_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.dist_encoder = ScalarEncoder(1000, 25, 0, 150)

    def encode(self, dx, dy, heading=None):
        dist = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi

        angle_sdr = self.angle_encoder.encode(angle)
        dist_sdr = self.dist_encoder.encode(dist)

        if self.is_fleet:
            angle_shifted = (angle + math.pi) % (2*math.pi)
            angle_phase_sdr = self.angle_encoder_phase.encode(angle_shifted)

            # Use provided heading, or default to 0 if none provided
            h = heading if heading is not None else 0.0
            heading_sdr = self.heading_encoder.encode(h)

            return np.concatenate((angle_sdr, dist_sdr, angle_phase_sdr, heading_sdr))
        else:
            return np.concatenate((angle_sdr, dist_sdr))

    def encode_union_topk(self, origin_x, origin_y, targets):
        float_state = np.zeros(self.size, dtype=np.float32)

        for target in targets:
            dx = target.x - origin_x
            dy = target.y - origin_y

            # Check if target has an angle (flight heading)
            heading = getattr(target, 'angle', None)

            dist = math.sqrt(dx**2 + dy**2)
            weight = 1.0 / (1.0 + dist)

            sdr = self.encode(dx, dy, heading)
            float_state[sdr] += weight

        final_state = np.zeros(self.size, dtype=bool)
        if len(targets) > 0:
            non_zero_count = np.count_nonzero(float_state)
            k = min(self.active_bits, non_zero_count)
            if k > 0:
                top_k_indices = np.argpartition(float_state, -k)[-k:]
                final_state[top_k_indices] = True

        return final_state


# --- DECODERS ---



class ReverseScalarDecoder:
    """
    Decodes an SDR back into a continuous scalar value.
    Assumes the SDR was generated by a contiguous chunk or finds the center of mass.
    """
    def __init__(self, size, min_val, max_val):
        self.size = size
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val

    def decode(self, indices):
        if len(indices) == 0:
            return self.min_val # Default if no bits active

        # Center of mass of the active indices
        mean_idx = np.mean(indices)

        # Map back to value space
        val = self.min_val + (mean_idx / (self.size - 1)) * self.range
        return min(max(val, self.min_val), self.max_val)

class ReverseCyclicDecoder:
    """
    Decodes an SDR back into a cyclic value (like an angle).
    Uses circular mean to handle wrap-around correctly.
    """
    def __init__(self, size, min_val=0.0, max_val=2*math.pi):
        self.size = size
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val

    def decode(self, indices):
        if len(indices) == 0:
            return self.min_val

        # Convert indices to angles on a unit circle
        angles = (indices / self.size) * 2 * math.pi

        # Compute circular mean
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))

        mean_angle = math.atan2(sin_sum, cos_sum)
        if mean_angle < 0:
            mean_angle += 2 * math.pi

        # Map back to desired range
        val = self.min_val + (mean_angle / (2 * math.pi)) * self.range
        return val

def encoding_to_action(encoding, actions, sp_size):
    buckets = np.floor(encoding / (float(sp_size) / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets, minlength=actions)
    return counts.argmax()

def indices_to_columns(tm_indices, cells_per_col):
    cols = np.array(tm_indices) // cells_per_col
    return np.unique(cols)


# --- GAME LOGIC ---
INPUT_SIZE = 15000

class OrbitWarsEncoder:
    def __init__(self):
        self.size = INPUT_SIZE
        self.ships_encoder = ScalarEncoder(1000, 25, 0, 500)
        self.geo_planet_encoder = TileGeospatialEncoder(2000, 50, is_fleet=False)
        self.geo_fleet_encoder = TileGeospatialEncoder(4000, 100, is_fleet=True)

    def encode(self, my_planet, planets, fleets, player):
        state = np.zeros(self.size, dtype=bool)
        state[0:1000] = self.ships_encoder.encode(my_planet.ships)

        enemy_planets = [p for p in planets if p.owner != player and p.owner != -1 and p.id != my_planet.id]
        neutral_planets = [p for p in planets if p.owner == -1 and p.id != my_planet.id]
        friendly_planets = [p for p in planets if p.owner == player and p.id != my_planet.id]

        enemy_fleets = [f for f in fleets if f.owner != player]
        friendly_fleets = [f for f in fleets if f.owner == player]

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

class HTMRLAgent:
    def __init__(self, load_path=None):
        self.encoder = OrbitWarsEncoder()
        if load_path and os.path.exists(load_path):
            with open(load_path, "rb") as f:
                self.sp = pickle.load(f)
        else:
            self.sp = SpatialPooler(
                input_size=(INPUT_SIZE,), 
                acts_n=1,
                cell_count=2048,
                active_count=41
            )

    def get_moves(self, obs, learn=False, reward=0):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])

        planets = [Planet(*p) for p in raw_planets]
        from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
        fleets = [Fleet(*f) for f in raw_fleets]
        
        my_planets = [p for p in planets if p.owner == player]
        
        for mine in my_planets:
            state = self.encoder.encode(mine, planets, fleets, player)
            encoding = self.sp.step(state, learn=False)

            # Reverting back to original 25 categorical actions correctly mapped from SP column chunks
            action = encoding_to_action(encoding, 25, self.sp.size)

            if action == 0:
                continue
                
            if action <= 12:
                ships = max(1, int(mine.ships * 0.5))
                sector = action - 1
            else:
                ships = mine.ships
                sector = action - 13
                
            angle = (sector * (2 * math.pi / 12))
            moves.append([mine.id, angle, ships])
                
        return moves

_cached_agents = {}

def agent_fn(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    if player not in _cached_agents:
        path = "best_bot.pkl"
        if not os.path.exists(path):
            try:
                path = os.path.join(os.path.dirname(__file__), "best_bot.pkl")
            except NameError:
                pass
                
        _cached_agents[player] = HTMRLAgent(load_path=path)
        
    return _cached_agents[player].get_moves(obs, learn=False)
