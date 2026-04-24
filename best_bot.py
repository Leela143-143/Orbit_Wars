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
                 only_reinforce_selected=True, normalize_rewards=True, cell_count=2048, active_count=40, boost_until=0,
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


# === AGENT CODE ===
Planet = namedtuple('Planet', ['id', 'owner', 'x', 'y', 'radius', 'ships', 'production'])

NUM_ACTIONS = 25
INPUT_SIZE = 40

class OrbitWarsEncoder:
    def __init__(self, size=INPUT_SIZE):
        self.size = size

    def encode(self, my_planet, planets, player):
        state = np.zeros(self.size, dtype=bool)
        ships_idx = min(3, int(my_planet.ships / 25))
        state[36 + ships_idx] = True
        
        for p in planets:
            if p.id == my_planet.id:
                continue
                
            angle = math.atan2(p.y - my_planet.y, p.x - my_planet.x)
            if angle < 0:
                angle += 2 * math.pi
                
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
                self.sp = pickle.load(f)
        else:
            self.sp = SpatialPooler(
                input_size=(INPUT_SIZE,), 
                acts_n=NUM_ACTIONS,
                cell_count=2048,
                active_count=40
            )

    def get_moves(self, obs, learn=False, reward=0):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        planets = [Planet(*p) for p in raw_planets]
        
        my_planets = [p for p in planets if p.owner == player]
        
        for mine in my_planets:
            state = self.encoder.encode(mine, planets, player)
            encoding = self.sp.step(state, learn=False)
            action = encoding_to_action(encoding, NUM_ACTIONS, self.sp.size)
            
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
        # Load best_bot.pkl from local directory (useful for Kaggle kernels)
        path = "best_bot.pkl"
        if not os.path.exists(path):
            try:
                # Try relative to the script location
                path = os.path.join(os.path.dirname(__file__), "best_bot.pkl")
            except NameError:
                pass
                
        _cached_agents[player] = HTMRLAgent(load_path=path)
        
    return _cached_agents[player].get_moves(obs, learn=False)
