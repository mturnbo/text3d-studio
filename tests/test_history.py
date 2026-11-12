from text3d_studio.history import HISTORY_LIMIT, MERGE_WINDOW_S, History


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_new_history_has_nothing_to_undo_or_redo():
    history = History()
    assert not history.can_undo
    assert not history.can_redo


def test_undo_and_redo_on_empty_history_return_none_and_change_nothing():
    history = History()
    assert history.undo("current") is None
    assert history.redo("current") is None
    assert not history.can_undo
    assert not history.can_redo


def test_undo_returns_states_newest_first_and_redo_replays_them():
    history = History()
    history.record("a")
    history.record("b")
    assert history.undo("c") == "b"
    assert history.undo("b") == "a"
    assert not history.can_undo
    assert history.redo("a") == "b"
    assert history.redo("b") == "c"
    assert not history.can_redo
    assert history.undo("c") == "b"


def test_limit_drops_the_oldest_state():
    history = History(limit=3)
    for state in "abcd":
        history.record(state)
    assert [history.undo(None) for _ in range(4)] == ["d", "c", "b", None]


def test_default_limit_is_one_hundred():
    history = History()
    for i in range(HISTORY_LIMIT + 5):
        history.record(i)
    undone = []
    while history.can_undo:
        undone.append(history.undo(None))
    assert len(undone) == HISTORY_LIMIT == 100
    assert undone[-1] == 5


def test_a_new_change_after_undo_clears_redo():
    history = History()
    history.record("a")
    history.record("b")
    history.undo("c")
    assert history.can_redo
    history.record("b2")
    assert not history.can_redo
    assert history.redo("x") is None


def test_clear_empties_both_stacks():
    history = History()
    history.record("a")
    history.record("b")
    history.undo("c")
    history.clear()
    assert not history.can_undo
    assert not history.can_redo


def test_same_key_within_the_window_merges_into_one_step():
    clock = FakeClock()
    history = History(clock=clock)
    history.record("before", key="size")
    clock.now += MERGE_WINDOW_S * 0.9
    history.record("mid", key="size")
    clock.now += MERGE_WINDOW_S * 0.9  # the window runs from the latest change, so a long burst stays one step
    history.record("late", key="size")
    assert history.undo("now") == "before"
    assert not history.can_undo


def test_a_gap_longer_than_the_window_starts_a_new_step():
    clock = FakeClock()
    history = History(clock=clock)
    history.record("a", key="size")
    clock.now += MERGE_WINDOW_S + 0.01
    history.record("b", key="size")
    assert history.undo("c") == "b"
    assert history.undo("b") == "a"


def test_different_keys_or_no_key_never_merge():
    history = History(clock=FakeClock())
    history.record("a", key="size")
    history.record("b", key="x")
    history.record("c")
    history.record("d")
    history.record("e", key="x")
    assert [history.undo(None) for _ in range(6)] == ["e", "d", "c", "b", "a", None]


def test_undo_ends_a_burst_so_the_next_change_is_its_own_step():
    history = History(clock=FakeClock())
    history.record("a", key="size")
    history.undo("b")
    history.record("a2", key="size")
    assert history.undo("c") == "a2"
    assert not history.can_undo

