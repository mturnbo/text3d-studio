"""Undo history: a bounded stack of states to go back to, and the states undone since. No Qt here."""

import time

HISTORY_LIMIT = 100
MERGE_WINDOW_S = 0.6  # changes to the same field closer together than this merge into one step


class History:
    """Holds opaque states. `record` takes the state from just before a change; `undo` and `redo` trade the
    current state for the one to restore.

    A change recorded with a `key` merges into the previous step when that step had the same key and came less
    than MERGE_WINDOW_S ago, so a burst (wheel scaling, typing) undoes in one go. Any new step clears the redo
    stack, and past `limit` steps the oldest is dropped.
    """

    def __init__(self, limit: int = HISTORY_LIMIT, clock=time.monotonic):
        self.limit = limit
        self.clock = clock
        self._undo: list = []
        self._redo: list = []
        self._last_key = None
        self._last_time = 0.0

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def record(self, state, key=None) -> None:
        now = self.clock()
        merges = key is not None and key == self._last_key and now - self._last_time < MERGE_WINDOW_S
        self._last_key, self._last_time = key, now
        self._redo.clear()
        if merges:
            return  # the step already holds the state from before the burst
        self._undo.append(state)
        if len(self._undo) > self.limit:
            del self._undo[0]

    def undo(self, current):
        """The state to restore, or None when there is nothing to undo. `current` becomes redoable."""
        if not self._undo:
            return None
        self._last_key = None
        self._redo.append(current)
        return self._undo.pop()

    def redo(self, current):
        """The state to restore, or None when there is nothing to redo. `current` becomes undoable."""
        if not self._redo:
            return None
        self._last_key = None
        self._undo.append(current)
        return self._redo.pop()

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._last_key = None
