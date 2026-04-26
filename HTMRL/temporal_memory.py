from scipy.sparse import csr_matrix, dok_matrix, coo_matrix, find
import numpy as np
import random
from collections import defaultdict
import HTMRL.log as log
import time
cells_per_col = 32

activation_thresh = 10
initial_perm = 0.38
connected_perm = 0.4  # 0.5
learning_thresh = 8
learning_enabled = True

perm_inc_step = 0.05
perm_dec_step = 0.001
perm_dec_predict_step = 0.000#5
# max_seg set very low right now: serious impact on performance; should autoscale internally
max_segments_per_cell = 32
max_synapses_per_segment = 32

sp_size = (2048,)
sp_size_flat = np.prod(sp_size)

tm_size = (sp_size_flat, cells_per_col)
tm_size_flat = np.prod(tm_size)
max_segs_total = tm_size_flat * max_segments_per_cell

timer = 0
timera = 0
timerb = 0

ctr = 0
def to_flat_tm(col, cell):
    """
    Col and cell index to flattened representation, assuming no segments
    """
    return col * cells_per_col + cell


def to_flat_segments(col, cell, seg=0):
    """
    Col, cell and segment index to flattened representation
    """
    result = col * cells_per_col * max_segments_per_cell + (cell * max_segments_per_cell) + seg
    return result


def unflatten_segments(flat):
    """
    Given a flattened index with segments, convert to col, cell and segment indices
    :param flat:
    :return:
    """
    col = flat // (cells_per_col * max_segments_per_cell)
    remain = flat % (cells_per_col * max_segments_per_cell)
    cell = remain // max_segments_per_cell
    seg = remain % max_segments_per_cell
    return (col, cell, seg)

def csr_double(a):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    #a.data = np.hstack((a.data,b.data))
    #a.indices = np.hstack((a.indices,b.indices))
    extra_rows = a.shape[0] if a.shape[0] else 1
    a.indptr = np.append(a.indptr, extra_rows * [a.nnz])#np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+extra_rows,a.shape[1])
    return a

def arr_double(a):
    extra_rows = a.shape[0] if a.shape[0] else 1
    return np.append(a, extra_rows * [0])


class TemporalMemory(object):
    def __init__(self):
        self.actives = csr_matrix((1, tm_size_flat), dtype=bool)
        self.winners = csr_matrix((1, tm_size_flat), dtype=bool)
        self.active_segs = csr_matrix((sp_size_flat, cells_per_col * max_segments_per_cell), dtype=bool)
        self.matching_segs = csr_matrix((sp_size_flat, cells_per_col * max_segments_per_cell), dtype=bool)

        self.matches_per_col = np.zeros((sp_size_flat,))  # csc_matrix((1,sp_size_flat))
        self.actives_per_col = np.zeros((sp_size_flat,))

        # These buffers contain a list of values followed by a list of row [and col] indices
        # At the end of each step, each buffer must be added to the appropriate matrix/array
        self.permanence_updates_buffer = [[], [], []]
        self.active_updates_buffer = [[], []]
        self.winner_updates_buffer = [[], []]

        #self.active_pot_counts = [0] * l#csc_matrix((1, tm_size_flat[0] * max_segments_per_cell), dtype=np.int)
        self.active_pot_counts = [0] * max_segs_total #np.array(max_segs_total, dtype=np.int)#csc_matrix((1, tm_size_flat[0] * max_segments_per_cell), dtype=np.int)

        self.seg_matrix = csr_matrix((0,tm_size_flat))#csc_matrix((tm_size_flat[0], tm_size_flat[0] * max_segments_per_cell))
        self.seg_linkings = dict()
        self.seg_linkings_reverse = np.empty((0,), dtype=int)
        self.seg_counts = defaultdict(int)

    def add_segment(self, col_id, cell_id):
        """
        Create a new segment on a specific cell.
        """
        # The synapse matrix is large enough to accomodate for any segment,
        # just remember how many are created on the current cell.
        index = to_flat_tm(col_id, cell_id)
        index_seg = to_flat_segments(col_id, cell_id, self.seg_counts[index])
        if self.seg_matrix.shape[0] == len(self.seg_linkings):
            self.seg_matrix = csr_double(self.seg_matrix)
        self.seg_counts[index] += 1
        if self.seg_linkings_reverse.shape[0] == len(self.seg_linkings):
            self.seg_linkings_reverse = arr_double(self.seg_linkings_reverse)
        self.seg_linkings_reverse[len(self.seg_linkings)] =index_seg
        self.seg_linkings[index_seg] = len(self.seg_linkings)

        return self.seg_counts[index] - 1

    def get_least_used_cell(self, col):
        """
        Of all cells for the given col, pick one so that no other cell has fewer segments.
        """
        minimum = max_segments_per_cell
        mins = []

        base_idx = col * cells_per_col
        # Get all cells for which no cell with fewer segments exists for this col
        for i in range(cells_per_col):
            this_len = self.seg_counts.get(base_idx + i, 0)
            if this_len == minimum:
                mins.append(i)
            elif this_len < minimum:
                minimum = this_len
                mins = [i]

        # From those, pick one at random
        return random.choice(mins) if len(mins) else None

    def get_best_matching_seg(self, col):
        """
        Of all segments for any cell in a specific column, get the one with most matching synapses.
        In case of ties, get the first one (??)
        """
        best_score = -1
        best_cell = None
        best_seg = None

        col_offset = col * cells_per_col * max_segments_per_cell
        indices = self.matching_segs.indices[self.matching_segs.indptr[col]:self.matching_segs.indptr[col + 1]]

        for idx in indices:
            # Inline unflatten to avoid function call overhead
            flat = idx + col_offset
            remain = flat % (cells_per_col * max_segments_per_cell)
            cell = remain // max_segments_per_cell
            seg_idx = remain % max_segments_per_cell

            # Inline to_flat_segments
            full_idx = col_offset + (cell * max_segments_per_cell) + seg_idx

            seg_linked = self.seg_linkings.get(full_idx)
            if seg_linked is not None:
                this_score = self.active_pot_counts[seg_linked]
                if this_score > best_score:
                    best_cell = cell
                    best_seg = seg_idx
                    best_score = this_score

        return (best_cell, best_seg)

    def grow_synapses(self, col, cell, seg_idx, count):
        """
        For a given segment, grow up to a number of synapses to winner cells of the previous step.
        """
        if count <= 0:
            return

        idx = to_flat_segments(col, cell, seg_idx)
        idx_toseg = self.seg_linkings[idx]

        # Array difference avoids slow sets and random.sample
        arr_a = self.winners.indices
        arr_b = self.seg_matrix.indices[self.seg_matrix.indptr[idx_toseg]:self.seg_matrix.indptr[idx_toseg + 1]]
        unconnected = np.setdiff1d(arr_a, arr_b, assume_unique=True)

        if unconnected.size == 0:
            return

        count = min(unconnected.size, count)
        inds = np.random.choice(unconnected, count, replace=False)

        # Append in bulk
        self.permanence_updates_buffer[0].extend([initial_perm] * count)
        self.permanence_updates_buffer[1].extend([idx_toseg] * count)
        self.permanence_updates_buffer[2].extend(inds.tolist())

    def burst(self, col):
        """
        Burst all cells in an unpredicted column.
        """
        # Activate all cells in the col
        _from = to_flat_tm(col, 0)
        _to = to_flat_tm(col + 1, 0)
        # Actually apply these later on, for efficiency
        self.active_updates_buffer[0].extend(cells_per_col * [True])
        self.active_updates_buffer[1].extend(list(range(_from, _to)))
        is_new_seg = False

        # Micro-optimization: cache the call result
        if self.matches_per_col[col]:
            # Winner cell is the one with the best matching segment...
            (winner_cell, learning_seg) = self.get_best_matching_seg(col)
        else:
            # ...or if there are not with the least segments
            winner_cell = self.get_least_used_cell(col)
            if learning_enabled and winner_cell is not None:
                # Grow a new segment because none matched this sequence
                learning_seg = self.add_segment(col, winner_cell)
                is_new_seg = True
            elif learning_enabled and winner_cell is None:
                pass
                #print("Ran out of segment space!")


        # Actually set the winner cell later on
        if winner_cell is not None: #hotfix for low seg count
            self.winner_updates_buffer[0].append(True)
            self.winner_updates_buffer[1].append(to_flat_tm(col, winner_cell))

        if learning_enabled and winner_cell is not None:

            seg_idx = to_flat_segments(col, winner_cell, learning_seg)
            seg_linked = self.seg_linkings[seg_idx]
            seg = self.seg_matrix.indices[self.seg_matrix.indptr[seg_linked]:self.seg_matrix.indptr[seg_linked + 1]]

            if len(seg) > 0:
                # Find which synapses are connected to previously active cells
                # Use array-based boolean checking against active_old_t which is dense/accessible
                # Since we want to check if the synapse column was active in the previous step
                # Actually, self.actives is the currently active ones (from previous step? yes, we haven't overwritten it yet)
                active_idxs = set(self.actives.indices)

                # Vectorized permanence update appends
                for syn_col in seg:
                    if syn_col in active_idxs:
                        self.permanence_updates_buffer[0].append(perm_inc_step)
                    else:
                        self.permanence_updates_buffer[0].append(-perm_dec_step)
                    self.permanence_updates_buffer[1].append(seg_linked)
                    self.permanence_updates_buffer[2].append(syn_col)

            # Aim for specific number of potential synapses for winner segment
            new_syn_count = max_synapses_per_segment - (self.active_pot_counts[seg_linked] if not is_new_seg else 0)
            if new_syn_count:
                self.grow_synapses(col, winner_cell, learning_seg, new_syn_count)

    def activate_predicted_col(self, col):
        """
        At least one activate segment in this activated column, activate all cells with active segment
        """
        if log.has_trace():
            log.trace("Active col has {} active segs".format(self.get_activated_segs_for_col_count(col)))
        for idx in self.get_activated_segs_for_col(col):
            #idx = idx + (col * cells_per_col * max_segments_per_cell)
            cell = idx // max_segments_per_cell
            seg_idx = idx % max_segments_per_cell
            cell_idx = to_flat_tm(col, cell)
            # Actually apply later
            self.active_updates_buffer[0].append(True)
            self.active_updates_buffer[1].append(cell_idx)
            self.winner_updates_buffer[0].append(True)
            self.winner_updates_buffer[1].append(cell_idx)

            if learning_enabled:
                idx = to_flat_segments(col, cell, seg_idx)
                seg_linked = self.seg_linkings[idx]
                existing_synapses = self.seg_matrix.indices[
                                    self.seg_matrix.indptr[seg_linked]:self.seg_matrix.indptr[seg_linked + 1]]

                if len(existing_synapses) > 0:
                    # Actives_old_perms contains permanence increase value for previously active cells
                    # and the decrease value for previously inactive
                    # Use numpy indexing instead of list comprehension
                    data = self.actives_old_perms[existing_synapses].tolist()
                    self.permanence_updates_buffer[0].extend(data)
                    self.permanence_updates_buffer[1].extend([seg_linked] * len(existing_synapses))
                    self.permanence_updates_buffer[2].extend(existing_synapses.tolist())

                new_syn_count = max_synapses_per_segment - self.active_pot_counts[seg_linked]
                if new_syn_count:
                    self.grow_synapses(col, cell, seg_idx, new_syn_count)

    def get_activated_segs_for_col(self, col):
        """
        Gets the previous step's activated segment indices (flattened) for one column
        """
        return self.active_segs.indices[self.active_segs.indptr[col]:self.active_segs.indptr[col + 1]]
        #return self.active_segs_old[:, to_flat_segments(col, 0):to_flat_segments(col + 1, 0)]

    def get_activated_segs_for_col_count(self, col):
        """
        Counts the number of activated segments in the previous step for one column
        """
        val = self.actives_per_col[col]
        return val

    def get_matching_segs_for_col(self, col):
        """
        Gets the previous step's matching segment indices (flattened) for one column

        """
        return self.matching_segs.indices[self.matching_segs.indptr[col]:self.matching_segs.indptr[col + 1]]
        #return self.matching_segs_old[:, to_flat_segments(col, 0):to_flat_segments(col + 1, 0)]

    def get_matching_segs_for_col_count(self, col):
        """
        Counts the number of matching segments in the previous step for one column
        """
        return self.matches_per_col[col]

    def punish_predicted(self, col):
        """
        Column was predicted to become active but didn't. Punish all synapses contributing to this prediction
        """
        if learning_enabled and perm_dec_predict_step > 0:
            for match_idx in self.get_matching_segs_for_col(col):

                (c, cell, seg) = unflatten_segments(match_idx + (col * cells_per_col * max_segments_per_cell))
                #assert c == col
                seg_linked = self.seg_linkings[to_flat_segments(col, cell, seg)]

                indices = self.seg_matrix.indices[self.seg_matrix.indptr[seg_linked]: self.seg_matrix.indptr[seg_linked+1]]
                for idx in indices:
                    # Actives_old_t is CSC matrix of previously active cells (all in 1 row).
                    # If this and the next cell have the same indptr, there are no nonzero values in that column
                    # so it wasn't active. There are probably cleaner ways of doing this as efficiently.
                    if self.actives_old_t.indptr[idx] != self.actives_old_t.indptr[idx + 1]:
                        self.permanence_updates_buffer[0].append(-perm_dec_predict_step)
                        self.permanence_updates_buffer[1].append(seg_linked)
                        self.permanence_updates_buffer[2].append(idx)

    def activate(self):
        """
        Calculate the matching and active segments, for use in the next step
        """
        # Truncate seg matrix to actual size instead of doing it after multiply
        # This drastically speeds up the multiply
        actual_seg_matrix = self.seg_matrix[:len(self.seg_linkings)]

        # Broadcasting pointwise multiplication, contains permanences of synapses to active cells
        active_synapses = actual_seg_matrix.multiply(self.actives)

        connected_synapses = active_synapses.copy()  # Copy because we still need this version for potentials
        # Only considered connected if permanence is high enough
        connected_synapses.data[connected_synapses.data < connected_perm] = 0.0
        connected_synapses.eliminate_zeros()

        connected_synapses.has_canonical_format = True # Avoids sum_duplicates; not necessary here and slow
        connected_synapses = connected_synapses.astype(bool)
        conn_syns_counts = connected_synapses.sum(axis=1)
        # Segment is only active if there are enough connected synapses
        conn_syns_counts[conn_syns_counts < activation_thresh] = 0
        #self.active_segs = csc_matrix(conn_syns_counts, dtype=bool)
        conn_syns_counts_full = coo_matrix((np.squeeze(np.asarray(conn_syns_counts)), (self.seg_linkings_reverse[:len(self.seg_linkings)], [0] * len(self.seg_linkings))),
                                           dtype=bool, shape=(sp_size_flat*cells_per_col*max_segments_per_cell,1))
        self.active_segs = conn_syns_counts_full.reshape((sp_size_flat, cells_per_col * max_segments_per_cell),
                                                                    order='C').tocsr()

        self.active_segs.eliminate_zeros()
        #print("Active seg count:", self.active_segs.data.shape)
        # finds = find(self.active_segs)
        # for i in range(finds[0].shape[0]):
        #     (row, col, val) = (finds[0][i], finds[1][i], finds[2][i])
        #     cell = col // max_segments_per_cell
        #     seg_idx = col % max_segments_per_cell
        #     this_idx = to_flat_segments(row, cell, seg_idx)
        #     assert this_idx in self.seg_linkings

        # Any permanence is enough to be potential
        active_synapses.has_canonical_format = True # Avoids sum_duplicates; not necessary here and slow
        pot_syns_counts = active_synapses.astype(bool).sum(axis=1)
        self.active_pot_counts = np.squeeze(np.asarray(pot_syns_counts)).tolist()#dok_matrix(pot_syns_counts)

        # Segment is only matching if there are enough potential connections
        pot_syns_counts[pot_syns_counts < learning_thresh] = 0
        pot_syns_counts_full = coo_matrix((np.squeeze(np.asarray(pot_syns_counts)), (self.seg_linkings_reverse[:len(self.seg_linkings)], [0] * len(self.seg_linkings))),
                                          dtype=bool, shape=(sp_size_flat*cells_per_col*max_segments_per_cell,1))

        #self.matching_segs = csr_matrix(pot_syns_counts_full, dtype=bool) #TODO CONFIRM ORDER

        self.matching_segs = pot_syns_counts_full.reshape((sp_size_flat, cells_per_col * max_segments_per_cell), order='C').tocsr()
        self.matching_segs.eliminate_zeros()
        #TEMP DEBUG CHECK
        #print("Matching seg count:", self.matching_segs.data.shape)
        # finds = find(self.matching_segs)
        # for i in range(finds[0].shape[0]):
        #     (row, col, val) = (finds[0][i], finds[1][i], finds[2][i])
        #     cell = col // max_segments_per_cell
        #     seg_idx = col % max_segments_per_cell
        #     this_idx = to_flat_segments(row, cell, seg_idx)
        #     assert this_idx in self.seg_linkings

        # Reshape matrix to have 1 col per TM column instead of per cell, for easy counting
        self.matches_per_col = np.asarray(self.matching_segs
            .sum(
                axis=1)).ravel()
        #print("Should contain ints:", self.matches_per_col.dtype)
        self.actives_per_col = np.asarray(self.active_segs.sum(axis=1)).ravel()
        #print("cols with active:", np.count_nonzero(self.actives_per_col))
        #print("cols with matching:", np.count_nonzero(self.matches_per_col))
        # assert self.actives_per_col.sum() == self.active_segs.data.shape[0]
        # assert self.matches_per_col.sum() == self.matching_segs.data.shape[0]

    def update_synapses(self):
        """
        Performs the actual changes to the permanence matrix for one step.
        Grouped into one addition for efficiency
        """
        if len(self.permanence_updates_buffer[0]):
            # Use actual sum_duplicates on COO instead of naive addition
            # This is significantly faster for overlapping modifications
            modder = coo_matrix((self.permanence_updates_buffer[0],
                                 (self.permanence_updates_buffer[1], self.permanence_updates_buffer[2])),
                                shape=self.seg_matrix.shape)
            modder.sum_duplicates()
            self.seg_matrix = self.seg_matrix + modder.tocsr()


    def update_actives_and_winners(self):
        """
        Performs the actual changes to the winner and active matrices for one step.
        Grouped into one addition each for efficiency
        """
        if len(self.active_updates_buffer[0]):
            # Directly construct CSR since row is always 0
            n_items = len(self.active_updates_buffer[0])
            # For a 1-row CSR matrix:
            # indptr is [0, n_items], indices are the column indices, data is the values
            # However, we must ensure columns are sorted and unique for standard behavior.
            # Using COO -> CSR handles this perfectly in C.
            # Micro-opt: use np.zeros for the row indices.
            rows = np.zeros(n_items, dtype=np.int32)
            self.actives = coo_matrix((self.active_updates_buffer[0],
                                       (rows, self.active_updates_buffer[1])),
                                      shape=(1,tm_size_flat), dtype=bool).tocsr()
        if len(self.winner_updates_buffer[0]):
            n_items = len(self.winner_updates_buffer[0])
            rows = np.zeros(n_items, dtype=np.int32)
            self.winners = coo_matrix((self.winner_updates_buffer[0],
                                       (rows, self.winner_updates_buffer[1])),
                                      shape=(1,tm_size_flat), dtype=bool).tocsr()


    def step_end(self):
        """
        End-of-step bookkeeping. Move current matrices to their _old versions and reset them
        :return:
        """
        self.actives_old_t = self.actives.transpose().tocsr()

        dense_acts = self.actives.toarray()[0]

        self.actives_old_perms = np.where(dense_acts, perm_inc_step, -perm_dec_step)

        self.permanence_updates_buffer = [[], [], []]
        self.winner_updates_buffer = [[], []]
        self.active_updates_buffer = [[], []]

    def reset(self):
        self.actives = csr_matrix((1, tm_size_flat), dtype=bool)
        self.actives_old_t = self.actives.transpose().tocsr()
        self.winners = csr_matrix((1, tm_size_flat), dtype=bool)
        self.active_segs = csr_matrix((sp_size_flat, cells_per_col * max_segments_per_cell), dtype=bool)
        self.matching_segs = csr_matrix((sp_size_flat, cells_per_col * max_segments_per_cell), dtype=bool)
        self.active_pot_counts = [0] * max_segs_total #dok_matrix((1, tm_size_flat[0] * max_segments_per_cell))


    def step(self, activated_cols, reward=0.0):
        """
        The full step: from SP active columns to TM active cells
        """
        # If learning with a continuous reward, scale the permanence updates
        if reward != 0.0 and learning_enabled:
            # Simple scaling function
            active_scale = max(-1.0, min(1.0, reward))
            dense_acts = self.actives.toarray()[0]
            scaled_inc = perm_inc_step * active_scale
            scaled_dec = perm_dec_step * active_scale

            self.actives_old_perms = np.where(dense_acts, scaled_inc, -scaled_dec)
            self._cached_scale = active_scale
        else:
            self._cached_scale = 1.0

        activated_set = set(activated_cols)
        for col in activated_cols:
            # For each active SP column, either...
            if self.get_activated_segs_for_col_count(col) > 0:
                # ...activate the cell(s) predicted for that column
                self.activate_predicted_col(col)
            else:
                # ...or burst all cells
                self.burst(col)

        if perm_dec_predict_step > 0:
            matching_cols = np.nonzero(self.matches_per_col)[0]
            for col in matching_cols:
                if col not in activated_set:
                    self.punish_predicted(col)

        # The previous steps generated lists of updates to perform to the synapses and active/winner cells,
        # but they still have to be actually applied to the sparse matrices
        self.update_synapses()
        self.update_actives_and_winners()

        # Given the current state, generate the active/matching segments for the next step
        self.activate()

        # Bookkeeping towards next step
        self.step_end()

        # self.actives is already reset to return the "old" actives
        return self.actives
