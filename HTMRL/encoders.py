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

class HashGeospatialEncoder:
    """
    Encodes relative coordinates into an SDR using deterministic hashing.
    Generates a unique but continuous spatial representation.
    """
    def __init__(self, size=2000, active_bits=50, grid_resolution=1.0):
        self.size = size
        self.active_bits = active_bits
        self.grid_resolution = grid_resolution

    def encode(self, dx, dy):
        # Snap to grid to create consistent overlapping regions
        # Shift coordinate slightly based on bits to create overlap
        state = np.zeros(self.size, dtype=bool)

        # We need `active_bits` number of hash outputs.
        # To ensure spatial overlap, we sample points within a radius.

        # A simpler way to get spatial overlap with deterministic hashing is to
        # use the coordinates directly to seed a random number generator, but
        # since we want overlap for NEARBY locations, we encode the location
        # as a combination of cyclic grid phases or overlapping tiles.

        # The easiest approach for true continuous overlap without collision
        # is translating dx, dy to angle and distance, and concatenating them.
        # But for 2.5% sparsity in 2000 bits (50 active), we can use two 1000-bit encoders.
        pass

# Let's implement a clean 2D tile encoder.
class TileGeospatialEncoder:
    def __init__(self, size=2000, active_bits=50):
        self.size = size
        self.active_bits = active_bits

        # We divide size into 2 parts: Distance and Angle
        # Distance: 1000 bits, 25 active (2.5%)
        # Angle: 1000 bits, 25 active (2.5%)
        # Concatenated size = 2000 bits, 50 active. Sparsity = 50/2000 = 2.5%
        self.angle_encoder = CyclicEncoder(size // 2, active_bits // 2, 0, 2*math.pi)
        self.dist_encoder = ScalarEncoder(size // 2, active_bits // 2, 0, 150) # Board diagonal is ~141

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
            # Get indices of top K values
            # If there are fewer than K non-zero values, just take all non-zero
            non_zero_count = np.count_nonzero(float_state)
            k = min(self.active_bits, non_zero_count)
            if k > 0:
                top_k_indices = np.argpartition(float_state, -k)[-k:]
                final_state[top_k_indices] = True

        # If no targets exist, final_state is all 0s (0% sparsity).
        # We can leave it as 0s.
        return final_state
