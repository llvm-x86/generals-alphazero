import time
import threading

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from socketio import SimpleClient

from generals.core import game
from generals.core.env import GeneralsEnv
from generals.remote.generalsio_state import GeneralsIOstate
from generals.remote.local_server import LocalServer
from generals.remote.generalsio_client import GeneralsIOClient

SEED, SIZE = 3, 8


def _recv(client, name, timeout=10):
    """Next event called `name` (queue_update heartbeats and others are skipped)."""
    end = time.time() + timeout
    while True:
        ev = client.receive(timeout=max(0.1, end - time.time()))
        if ev[0] == name:
            return ev[1:]


@pytest.fixture
def match():
    srv = LocalServer(seed=SEED, size=SIZE, start_ticker=False)  # ticks driven by the test
    httpd = srv.serve("127.0.0.1", 0, background=True)
    url = f"http://127.0.0.1:{httpd.server_port}"
    a, b = SimpleClient(), SimpleClient()
    a.connect(url)
    b.connect(url)
    yield srv, a, b
    a.disconnect()
    b.disconnect()
    srv.close()


def _start(srv, a, b):
    a.emit("join_private", ("room", "alice", "k"))
    _recv(a, "queue_update")
    b.emit("join_private", ("room", "bob"))
    _recv(b, "queue_update")
    a.emit("set_force_start", ("room", True))
    b.emit("set_force_start", ("room", True))
    sa = _recv(a, "game_start")[0]
    sb = _recv(b, "game_start")[0]
    return sa, sb


def _expected_state():
    return GeneralsEnv(grid_dims=(SIZE, SIZE), pool_size=1).init_state(jrandom.PRNGKey(SEED))


def test_requires_two_distinct_clients_and_force_start(match):
    srv, a, b = match
    a.emit("join_private", ("room", "alice"))
    a.emit("set_force_start", ("room", True))
    _recv(a, "queue_update")
    time.sleep(0.3)
    assert not srv.lobbies["room"].playing  # one client alone never starts
    b.emit("join_private", ("room", "alice"))  # same user id: rejected
    _recv(b, "error_join")
    assert len(srv.lobbies["room"].sids) == 1


def test_game_start_update_attack_and_win(match):
    srv, a, b = match
    sa, sb = _start(srv, a, b)
    assert (sa["playerIndex"], sb["playerIndex"]) == (0, 1)
    assert sa["usernames"] == ["alice", "bob"] and sa["replay_id"]

    ua, = _recv(a, "game_update")
    ub, = _recv(b, "game_update")
    ref = _expected_state()
    pa = GeneralsIOstate(sa)
    pa.update(ua)
    W = SIZE
    assert pa.map[:2] == [W, W] and pa.turn == 1
    assert pa.map == LocalServer.payload(ref, 0)["map_diff"][2:]
    assert pa.generals[1] == -1 and pa.generals[0] != -1  # opponent hidden by fog
    gr, gc = (int(x) for x in np.asarray(ref.general_positions[0]))
    src = gr * W + gc

    # pick a passable neighbour for a legal move; one more with the wrong owner is ignored
    moves = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]
    dr, dc, d = next(m for m in moves if 0 <= gr + m[0] < W and 0 <= gc + m[1] < W and bool(ref.passable[gr + m[0], gc + m[1]]))
    dst = (gr + dr) * W + gc + dc
    a.emit("attack", (src, dst, 0))
    b.emit("attack", (0, 0, 0))  # bob does not own tile 0 (or has 1 army): skipped
    b.emit("attack", (10_000, 3, 0))  # out of bounds: rejected
    time.sleep(0.3)  # let the server thread enqueue before ticking
    srv.tick()

    ref, _ = game.step(ref, jnp.array([[0, gr, gc, d, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32))
    ua, = _recv(a, "game_update")
    ub, = _recv(b, "game_update")
    assert ua == LocalServer.payload(ref, 0) and ub == LocalServer.payload(ref, 1)
    pa.update(ua)
    assert pa.map[2 + (gr + dr) * W + gc + dc] == 0  # alice now owns the destination
    assert pa.turn == 2

    # force a capture: alice owns a tile next to bob's general holding a big army
    lob = srv.lobbies["room"]
    st = lob.state
    br, bc = (int(x) for x in np.asarray(st.general_positions[1]))
    ar, ac, d = (br, bc - 1, 3) if bc > 0 else (br, bc + 1, 2)
    st = st._replace(
        ownership=st.ownership.at[:, ar, ac].set(False).at[0, ar, ac].set(True),
        ownership_neutral=st.ownership_neutral.at[ar, ac].set(False),
        armies=st.armies.at[ar, ac].set(500),
        passable=st.passable.at[ar, ac].set(True),
        mountains=st.mountains.at[ar, ac].set(False),
        castles=st.castles.at[ar, ac].set(False),
    )
    lob.state = st
    a.emit("attack", (ar * W + ac, br * W + bc, 0))
    time.sleep(0.3)
    srv.tick()
    ref2, info = game.step(st, jnp.array([[0, ar, ac, d, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32))
    assert int(info.winner) == 0
    ua, = _recv(a, "game_update")
    assert ua == LocalServer.payload(ref2, 0)
    _recv(a, "game_won")
    assert _recv(b, "game_lost") == [{"killer": 0}]
    _recv(a, "game_over")
    assert not lob.playing


def test_leave_game_awards_win(match):
    srv, a, b = match
    _start(srv, a, b)
    b.emit("leave_game")
    _recv(a, "game_won")

def test_simultaneous_general_capture_trades_instead_of_ending(match):
    srv, a, b = match
    _start(srv, a, b)
    _recv(a, "game_update")
    _recv(b, "game_update")
    grid = jnp.zeros((SIZE, SIZE), dtype=jnp.int32).at[0, 0].set(1).at[0, SIZE - 1].set(2)
    st = game.create_initial_state(grid)._replace(time=jnp.int32(10))
    st = st._replace(
        ownership=st.ownership.at[0, 0, SIZE - 2].set(True).at[1, 1, 0].set(True),
        ownership_neutral=st.ownership_neutral.at[0, SIZE - 2].set(False).at[1, 0].set(False),
        armies=st.armies.at[0, 0].set(5).at[0, SIZE - 1].set(7).at[0, SIZE - 2].set(40).at[1, 0].set(30),
    )
    srv.lobbies["room"].state = st
    a.emit("attack", (SIZE - 2, SIZE - 1, 0))
    b.emit("attack", (SIZE, 0, 0))
    time.sleep(0.3)
    srv.tick()
    updated, = _recv(a, "game_update")
    assert updated["turn"] == 12
    assert srv.lobbies["room"].playing
    assert np.asarray(srv.lobbies["room"].state.general_positions).tolist() == [[0, SIZE - 1], [0, 0]]


def test_same_api_client_works_by_switching_endpoint():
    from generals.agents import ExpanderAgent

    srv = LocalServer(seed=SEED, size=SIZE, start_ticker=False)
    httpd = srv.serve("127.0.0.1", 0, background=True)
    url = f"http://127.0.0.1:{httpd.server_port}"
    peer = SimpleClient()
    try:
        peer.connect(url)
        with GeneralsIOClient(ExpanderAgent(), "alice", endpoint=url) as client:
            client.join_private_lobby("switch")
            thread = threading.Thread(target=client.join_game, daemon=True)
            thread.start()
            peer.emit("join_private", ("switch", "bob"))
            peer.emit("set_force_start", ("switch", True))
            _recv(peer, "game_start")
            _recv(peer, "game_update")
            peer.emit("leave_game")
            thread.join(timeout=10)
            assert not thread.is_alive()
            assert client.game_state.map[:2] == [SIZE, SIZE]
            assert client._score_wins == 1
    finally:
        peer.disconnect()
        srv.close()
