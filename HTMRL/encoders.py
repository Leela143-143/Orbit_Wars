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
        # Clip value to min/max
        v = min(max(value, self.min_val), self.max_val)

        # Calculate bucket
        bucket = int((v - self.min_val) / self.range * (self.buckets - 1))
        bucket = min(max(bucket, 0), self.buckets - 1)

        # Activate consecutive bits
        state[bucket : bucket + self.active_bits] = True
        return state

class CyclicEncoder:
    """
    Encodes an angle/cyclic value into an overlapping SDR where values at the
    edges (e.g. 0 and 2*pi) wrap around and overlap.
    """
    def __init__(self, size, active_bits, min_val=0.0, max_val=2*math.pi):
        self.size = size
        self.active_bits = active_bits
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val

    def encode(self, value):
        state = np.zeros(self.size, dtype=bool)

        # Normalize to 0-1
        norm_val = ((value - self.min_val) % self.range) / self.range
        if norm_val < 0:
            norm_val += 1.0

        center_idx = int(norm_val * self.size)
        half_active = self.active_bits // 2

        for i in range(self.active_bits):
            idx = (center_idx - half_active + i) % self.size
            state[idx] = True

        return state

class GeospatialEncoder:
    """
    Encodes relative coordinates into an SDR using distance semantics.
    For variable entities, we can bitwise OR the outputs.
    Instead of fixed hashing per bit, we can project the coordinates into
    distance and angle, and concatenate their scalar/cyclic encodings.
    """
    def __init__(self, size_per_entity=200):
        # We split the 200 bits into: 100 for cyclic angle, 100 for distance
        self.size = size_per_entity
        self.angle_encoder = CyclicEncoder(100, 21, 0, 2*math.pi)
        self.dist_encoder = ScalarEncoder(100, 21, 0, 150) # Max dist on 100x100 board is ~141

    def encode(self, origin_x, origin_y, target_x, target_y):
        dx = target_x - origin_x
        dy = target_y - origin_y

        dist = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi

        angle_sdr = self.angle_encoder.encode(angle)
        dist_sdr = self.dist_encoder.encode(dist)

        return np.concatenate((angle_sdr, dist_sdr))

    def encode_union(self, origin_x, origin_y, targets):
        """
        targets is a list of objects with x and y attributes.
        Returns a single unioned SDR.
        """
        state = np.zeros(self.size, dtype=bool)
        for target in targets:
            state |= self.encode(origin_x, origin_y, target.x, target.y)
        return state
