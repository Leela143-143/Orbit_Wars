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

    def encode_batch_indices(self, values):
        """Returns the flat indices that would be true for an array of values"""
        values = np.asarray(values)
        v = np.clip(values, self.min_val, self.max_val)
        buckets = ((v - self.min_val) / self.range * (self.buckets - 1)).astype(np.int32)
        buckets = np.clip(buckets, 0, self.buckets - 1)

        # Create an array of offsets
        offsets = np.arange(self.active_bits)
        # Broadcasting to create a 2D array of indices: [num_values, active_bits]
        indices = buckets[:, None] + offsets
        return indices

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

    def encode_batch_indices(self, values):
        """Returns the flat indices that would be true for an array of values"""
        values = np.asarray(values)
        norm_vals = ((values - self.min_val) % self.range) / self.range
        norm_vals[norm_vals < 0] += 1.0

        center_idxs = (norm_vals * self.size).astype(np.int32)
        half_active = self.active_bits // 2

        # Create an array of offsets
        offsets = np.arange(self.active_bits) - half_active
        # Broadcasting to create a 2D array of indices: [num_values, active_bits]
        indices = (center_idxs[:, None] + offsets) % self.size
        return indices

class TileGeospatialEncoder:
    def __init__(self, size=3000, active_bits=75, is_fleet=False):
        self.size = size
        self.active_bits = active_bits
        self.is_fleet = is_fleet

        if self.is_fleet:
            # Fleet: 5000 bits total, 125 active
            # 1. Positional Angle (1000 bits)
            # 2. Distance (1000 bits)
            # 3. Positional Angle Phase-Shifted (1000 bits)
            # 4. Flight Heading / Trajectory Angle (1000 bits)
            # 5. Velocity Magnitude / Speed (1000 bits)
            self.angle_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.dist_encoder = ScalarEncoder(1000, 25, 0, 150)
            self.angle_encoder_phase = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.heading_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.speed_encoder = ScalarEncoder(1000, 25, 0, 10.0)
        else:
            # Planet: 3000 bits total, 75 active
            # 1. Positional Angle
            # 2. Distance
            # 3. Velocity / Orbit Heading (1000 bits) - To see if it's moving
            self.angle_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.dist_encoder = ScalarEncoder(1000, 25, 0, 150)
            self.heading_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)

    def encode(self, dx, dy, heading=None, speed=0.0):
        dist = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi

        angle_sdr = self.angle_encoder.encode(angle)
        dist_sdr = self.dist_encoder.encode(dist)

        h = heading if heading is not None else 0.0
        heading_sdr = self.heading_encoder.encode(h)

        if self.is_fleet:
            angle_shifted = (angle + math.pi) % (2*math.pi)
            angle_phase_sdr = self.angle_encoder_phase.encode(angle_shifted)
            speed_sdr = self.speed_encoder.encode(speed)

            return np.concatenate((angle_sdr, dist_sdr, angle_phase_sdr, heading_sdr, speed_sdr))
        else:
            return np.concatenate((angle_sdr, dist_sdr, heading_sdr))

    def encode_union_topk(self, origin_x, origin_y, targets, velocity_map=None):
        float_state = np.zeros(self.size, dtype=np.float32)

        if not targets:
            return np.zeros(self.size, dtype=bool)

        if velocity_map is None:
            velocity_map = {}

        # Vectorize target properties
        target_ids = [t.id for t in targets]
        t_x = np.array([t.x for t in targets])
        t_y = np.array([t.y for t in targets])

        dx = t_x - origin_x
        dy = t_y - origin_y

        dists = np.hypot(dx, dy)
        angles = np.arctan2(dy, dx)
        angles[angles < 0] += 2 * math.pi

        weights = 1.0 / (1.0 + dists)

        if self.is_fleet:
            headings = np.array([t.angle for t in targets])
            speeds = np.array([t.ships for t in targets]) # Proxy for speed

            # Batch encode
            angle_idx = self.angle_encoder.encode_batch_indices(angles)
            dist_idx = self.dist_encoder.encode_batch_indices(dists) + 1000

            angle_shifted = (angles + math.pi) % (2*math.pi)
            angle_phase_idx = self.angle_encoder_phase.encode_batch_indices(angle_shifted) + 2000
            heading_idx = self.heading_encoder.encode_batch_indices(headings) + 3000
            speed_idx = self.speed_encoder.encode_batch_indices(speeds) + 4000

            all_indices = np.concatenate((angle_idx, dist_idx, angle_phase_idx, heading_idx, speed_idx), axis=1)

        else:
            headings = np.zeros(len(targets))
            for i, tid in enumerate(target_ids):
                if tid in velocity_map:
                    vx, vy = velocity_map[tid]
                    if vx != 0 or vy != 0:
                        h = math.atan2(vy, vx)
                        headings[i] = h if h >= 0 else h + 2 * math.pi

            # Batch encode
            angle_idx = self.angle_encoder.encode_batch_indices(angles)
            dist_idx = self.dist_encoder.encode_batch_indices(dists) + 1000
            heading_idx = self.heading_encoder.encode_batch_indices(headings) + 2000

            all_indices = np.concatenate((angle_idx, dist_idx, heading_idx), axis=1)

        # Accumulate weights into float_state
        # all_indices is shape [num_targets, num_active_bits]
        # weights is shape [num_targets]
        for i in range(len(targets)):
            float_state[all_indices[i]] += weights[i]

        final_state = np.zeros(self.size, dtype=bool)
        non_zero_count = np.count_nonzero(float_state)
        k = min(self.active_bits, non_zero_count)
        if k > 0:
            top_k_indices = np.argpartition(float_state, -k)[-k:]
            final_state[top_k_indices] = True

        return final_state
