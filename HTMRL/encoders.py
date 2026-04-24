import numpy as np
import math

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
    """
    Splits the encoding space into distinct blocks (e.g. Angle and Distance).
    """
    def __init__(self, size=2000, active_bits=50, is_fleet=False):
        self.size = size
        self.active_bits = active_bits
        self.is_fleet = is_fleet

        if self.is_fleet:
            # Fleet: 4000 bits total, 100 active
            # 1. Angle (1000 bits)
            # 2. Distance (3000 bits)
            self.angle_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.dist_encoder = ScalarEncoder(3000, 75, 0, 150)
        else:
            # Planet: 2000 bits total, 50 active
            self.angle_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.dist_encoder = ScalarEncoder(1000, 25, 0, 150)

    def encode(self, dx, dy):
        dist = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi

        angle_sdr = self.angle_encoder.encode(angle)
        dist_sdr = self.dist_encoder.encode(dist)

        return np.concatenate((angle_sdr, dist_sdr))

    def encode_union_topk(self, origin_x, origin_y, targets):
        """
        targets is a list of objects with x and y attributes.
        Returns a single unioned SDR with exactly `self.active_bits` active
        using Proximity-Weighted Density Summation.
        """
        float_state = np.zeros(self.size, dtype=np.float32)

        for target in targets:
            dx = target.x - origin_x
            dy = target.y - origin_y

            # Distance weight: closer entities have higher priority
            dist = math.sqrt(dx**2 + dy**2)
            weight = 1.0 / (1.0 + dist)

            sdr = self.encode(dx, dy)
            float_state[sdr] += weight

        # Top-K Sparsified Union
        final_state = np.zeros(self.size, dtype=bool)
        if len(targets) > 0:
            non_zero_count = np.count_nonzero(float_state)
            k = min(self.active_bits, non_zero_count)
            if k > 0:
                top_k_indices = np.argpartition(float_state, -k)[-k:]
                final_state[top_k_indices] = True

        return final_state
