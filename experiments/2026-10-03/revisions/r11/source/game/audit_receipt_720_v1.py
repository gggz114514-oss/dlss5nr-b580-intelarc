"""Bounded JSON-only frame receipts; sequence is independent of ring length."""
from collections import deque
from collections.abc import Sequence
from copy import deepcopy


class ReceiptRing(Sequence):
    def __init__(self, capacity=64, event_capacity=16):
        if type(capacity) is not int or capacity < 1 or type(event_capacity) is not int or event_capacity < 1:
            raise ValueError('receipt capacities must be positive integers')
        self.capacity, self.event_capacity = capacity, event_capacity
        self.rows = deque(maxlen=capacity)
        self.events = deque(maxlen=event_capacity)
        self.total = 0
        self.event_total = 0

    def append(self, row):
        sequence = row.get('frame', self.total)
        if sequence != self.total:
            raise RuntimeError('Receipt sequence must advance monotonically')
        self.rows.append(row)
        if self.total == 0 or row.get('new_entries', 0) or not row.get('passed', row.get('selection_gate', True)):
            self.events.append(row)
            self.event_total += 1
        self.total += 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self.rows)[index]
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def __deepcopy__(self, memo):
        return deepcopy(list(self.rows), memo)

    def snapshot(self):
        return dict(total=self.total, retained=len(self.rows), dropped=self.total - len(self.rows),
                    sequence_range=None if not self.rows else [self.total - len(self.rows), self.total - 1],
                    capacity=self.capacity, rows=deepcopy(list(self.rows)),
                    event_total=self.event_total, event_retained=len(self.events),
                    event_dropped=self.event_total - len(self.events),
                    event_capacity=self.event_capacity, events=deepcopy(list(self.events)))


def frame_sequence(counter):
    rows = counter.frame_routes
    return rows.total if isinstance(rows, ReceiptRing) else len(rows)


def latest_history_frame(counter, before_sequence):
    if frame_sequence(counter) != before_sequence + 1:
        raise RuntimeError("Actual history child did not record this frame's route")
    row = counter.frame_routes[-1]
    if row['frame'] != before_sequence:
        raise RuntimeError('Retained history sequence differs from the actual frame')
    # Compact copied metadata: no tensor/owner pointers, options or source
    # manifests duplicated in every branch receipt.
    return {key: row.get(key) for key in
            ('frame', 'history_route', 'frontend_route', 'new_entries', 'replay_delta',
             'seed_before', 'seed_after', 'history_calls', 'selection_gate')}


def retention(counter):
    rows = counter.frame_routes
    return rows.snapshot() if isinstance(rows, ReceiptRing) else dict(total=len(rows), retained=len(rows), dropped=0)
