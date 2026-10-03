"""Local private-lobby 1v1 server speaking generals.io's Socket.IO event schema.

Lets `GeneralsIOClient`-style clients switch URL between botws.generals.io and
localhost for private 1v1 tests. The official API does not permit bots on the
public human server; WebBridge can operate its user-facing browser UI.
Only private 1v1 is implemented: no ranked queue, team games, or browser client.

    python -m generals.remote.local_server --port 8080 --seed 0 --size 10 --tick 0.5

Protocol assumptions (NOT verified against a live generals.io trace):
  * client -> server: join_private(lobby, user[, key]), set_force_start(lobby, bool),
    attack(source, dest[, is50]), leave_game. Tile index = row * width + col.
  * server -> client: queue_update (reply to join_private + heartbeat while waiting),
    game_start({playerIndex, usernames, replay_id, ...}),
    game_update(data) one dict arg, game_lost({killer}), game_won (no args), then game_over
    (per official 31.4.3 bundle, via parent).
  * map_diff/cities_diff are always the full replacement [0, len(new), *new].
  * Transport is HTTP long-polling only (stdlib WSGI host has no websocket upgrade);
    clients forcing transports=['websocket'] will not connect.
  * One queued attack is consumed per tick; invalid queued attacks are skipped.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import socketio  # type: ignore

from generals.core import game
from generals.core.env import GeneralsEnv

# direction index order of game.DIRECTIONS: UP, DOWN, LEFT, RIGHT
_DELTAS = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
_DIRS = {v: k for k, v in _DELTAS.items()}


class _ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass


class Lobby:
    def __init__(self, name: str):
        self.name = name
        self.sids: list[str] = []
        self.users: dict[str, str] = {}  # sid -> user id
        self.force: set[str] = set()
        self.state = None  # GameState while playing
        self.queues: dict[int, deque] = {0: deque(), 1: deque()}
        self.replay_id = ""

    @property
    def playing(self) -> bool:
        return self.state is not None


class LocalServer:
    def __init__(self, seed: int = 0, size: int = 10, tick: float = 0.5, start_ticker: bool = True):
        self.seed, self.size, self.tick_seconds = seed, size, tick
        self.env = GeneralsEnv(grid_dims=(size, size), pool_size=1, general_trade=True)
        self.sio = socketio.Server(async_mode="threading", allow_upgrades=False)
        self.app = socketio.WSGIApp(self.sio)
        self.lobbies: dict[str, Lobby] = {}
        self.sid_lobby: dict[str, str] = {}
        self.games_started = 0
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._ticker = None
        self._httpd = None
        s = self.sio
        s.on("join_private", self._on_join_private)
        s.on("set_force_start", self._on_force_start)
        s.on("attack", self._on_attack)
        s.on("leave_game", self._on_leave)
        s.on("disconnect", self._on_disconnect)
        s.on("set_username", lambda sid, *a: None)
        if start_ticker:
            self._ticker = threading.Thread(target=self._run_ticker, daemon=True)
            self._ticker.start()

    # ---------------------------------------------------------------- events
    def _on_join_private(self, sid, lobby, user=None, key=None):
        with self.lock:
            lobby, user = str(lobby), str(user if user is not None else sid)
            if sid in self.sid_lobby:
                self._remove(sid)
            lob = self.lobbies.setdefault(lobby, Lobby(lobby))
            if lob.playing or len(lob.sids) >= 2 or user in lob.users.values():
                self.sio.emit("error_join", "lobby full, in game, or duplicate user", to=sid)
                return
            lob.sids.append(sid)
            lob.users[sid] = user
            self.sid_lobby[sid] = lobby
            self.sio.enter_room(sid, lobby)
            self._queue_update(lob)

    def _on_force_start(self, sid, lobby=None, flag=True):
        with self.lock:
            lob = self._lobby_of(sid)
            if lob is None or lob.playing:
                return
            (lob.force.add if flag else lob.force.discard)(sid)
            self._queue_update(lob)
            if len(lob.sids) == 2 and len(lob.force) == 2:
                self._start(lob)

    def _on_attack(self, sid, source=None, dest=None, is50=0, *_):
        with self.lock:
            lob = self._lobby_of(sid)
            if lob is None or not lob.playing:
                return
            try:
                src, dst, split = int(source), int(dest), int(bool(is50))
            except (TypeError, ValueError):
                return
            H, W = lob.state.armies.shape
            if not (0 <= src < H * W and 0 <= dst < H * W):
                return
            delta = (dst // W - src // W, dst % W - src % W)
            if delta not in _DELTAS:  # must be 4-adjacent (also rejects row wrap-around)
                return
            lob.queues[lob.sids.index(sid)].append((src // W, src % W, _DELTAS[delta], split))

    def _on_leave(self, sid, *_):
        with self.lock:
            self._remove(sid)

    def _on_disconnect(self, sid, *_):
        with self.lock:
            self._remove(sid)

    # ---------------------------------------------------------------- helpers
    def _lobby_of(self, sid):
        name = self.sid_lobby.get(sid)
        return self.lobbies.get(name) if name else None

    def _queue_update(self, lob: Lobby):
        self.sio.emit(
            "queue_update",
            {
                "lobbyId": lob.name,
                "isForcing": False,
                "numForce": len(lob.force),
                "forceStart": len(lob.force),
                "numPlayers": len(lob.sids),
                "playerIndices": list(range(len(lob.sids))),
                "usernames": [lob.users[s] for s in lob.sids],
            },
            room=lob.name,
        )

    def _remove(self, sid):
        lob = self._lobby_of(sid)
        if lob is None:
            return
        idx = lob.sids.index(sid)
        if lob.playing and int(lob.state.winner) < 0:
            other = lob.sids[1 - idx]
            self.sio.emit("game_won", to=other)
            self.sio.emit("game_over", to=other)
            lob.state = None
        self.sio.leave_room(sid, lob.name)
        lob.sids.remove(sid)
        lob.users.pop(sid, None)
        lob.force.discard(sid)
        del self.sid_lobby[sid]
        if lob.playing and not lob.sids:
            lob.state = None
        if not lob.sids:
            del self.lobbies[lob.name]
        elif not lob.playing:
            self._queue_update(lob)

    def _start(self, lob: Lobby):
        key = jrandom.PRNGKey(self.seed + self.games_started)
        self.games_started += 1
        lob.state = self.env.init_state(key)
        lob.queues = {0: deque(), 1: deque()}
        lob.force.clear()
        lob.replay_id = f"local-{lob.name}-{self.games_started}"
        usernames = [lob.users[s] for s in lob.sids]
        for i, sid in enumerate(lob.sids):
            self.sio.emit(
                "game_start",
                {
                    "playerIndex": i,
                    "usernames": usernames,
                    "replay_id": lob.replay_id,
                    "chat_room": f"game_{lob.replay_id}",
                    "team_chat_room": None,
                    "game_type": "private",
                    "swamps": [],
                    "teams": [0, 1],
                },
                to=sid,
            )
        self._send_updates(lob)

    @staticmethod
    def payload(state, player: int) -> dict:
        """game_update body for `player`, derived from the simulator's fog observation."""
        obs = game.get_observation(state, player)
        armies = np.asarray(obs.armies).astype(int)
        H, W = armies.shape
        terrain = np.full((H, W), -1, dtype=int)
        terrain[np.asarray(obs.fog_cells)] = -3
        terrain[np.asarray(obs.structures_in_fog)] = -4
        terrain[np.asarray(obs.mountains)] = -2
        terrain[np.asarray(obs.opponent_cells)] = 1 - player
        terrain[np.asarray(obs.owned_cells)] = player
        flat = [W, H] + armies.ravel().tolist() + terrain.ravel().tolist()
        castles = np.flatnonzero(np.asarray(obs.castles).ravel()).tolist()
        gens_vis = np.asarray(obs.generals)
        gpos = np.asarray(state.general_positions)
        generals = [int(r * W + c) if r >= 0 and c >= 0 and gens_vis[r, c] else -1 for r, c in gpos]
        info = game.get_info(state)
        eliminated = np.asarray(state.eliminated)
        scores = [
            {"i": p, "total": int(info.army[p]), "tiles": int(info.land[p]), "dead": bool(eliminated[p])}
            for p in range(2)
        ]
        return {
            "turn": int(state.time) + 1,
            "map_diff": [0, len(flat)] + flat,
            "cities_diff": [0, len(castles)] + castles,
            "generals": generals,
            "scores": scores,
        }

    def _send_updates(self, lob: Lobby):
        for i, sid in enumerate(lob.sids):
            self.sio.emit("game_update", self.payload(lob.state, i), to=sid)

    def _pick_action(self, lob: Lobby, p: int):
        """Pop queued attacks until one is legal on the current state (generals.io behaviour)."""
        st, q = lob.state, lob.queues[p]
        while q:
            r, c, d, split = q.popleft()
            if bool(st.ownership[p, r, c]) and int(st.armies[r, c]) > 1:
                dr, dc = _DIRS[d]
                nr, nc = r + dr, c + dc
                H, W = st.armies.shape
                if 0 <= nr < H and 0 <= nc < W and bool(st.passable[nr, nc]):
                    return [0, r, c, d, split]
        return [1, 0, 0, 0, 0]

    # ------------------------------------------------------------------ ticks
    def tick(self, lobby: str | None = None):
        """Advance every playing lobby (or just `lobby`) by exactly one turn."""
        with self.lock:
            for lob in list(self.lobbies.values()):
                if lobby is not None and lob.name != lobby:
                    continue
                if not lob.playing:
                    if lob.sids and lobby is None:
                        self._queue_update(lob)  # heartbeat so waiting clients keep receiving
                    continue
                actions = jnp.array([self._pick_action(lob, p) for p in range(2)], dtype=jnp.int32)
                lob.state, info = game.step(lob.state, actions, general_trade=self.env.general_trade)
                self._send_updates(lob)
                w = int(info.winner)
                if w >= 0:
                    for i, sid in enumerate(lob.sids):
                        if i == w:
                            self.sio.emit("game_won", to=sid)
                        else:
                            self.sio.emit("game_lost", {"killer": w}, to=sid)
                        self.sio.emit("game_over", to=sid)
                    lob.state = None
                    lob.force.clear()

    def _run_ticker(self):
        nxt = time.monotonic()
        while not self._stop.is_set():
            nxt += self.tick_seconds
            self.tick()
            self._stop.wait(max(0.0, nxt - time.monotonic()))

    # ------------------------------------------------------------------- host
    def serve(self, host: str = "127.0.0.1", port: int = 8080, background: bool = False):
        self._httpd = make_server(host, port, self.app, server_class=_ThreadingWSGIServer, handler_class=_QuietHandler)
        if background:
            threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        else:
            self._httpd.serve_forever()
        return self._httpd

    def close(self):
        self._stop.set()
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--tick", type=float, default=0.5)
    a = ap.parse_args()
    if a.size < 4 or a.tick <= 0 or not 0 <= a.port <= 65535:
        ap.error("--size must be at least 4, --tick positive, and --port in 0..65535")
    srv = LocalServer(a.seed, a.size, a.tick)
    print(f"Local generals server on http://{a.host}:{a.port}")
    try:
        srv.serve(a.host, a.port)
    finally:
        srv.close()


if __name__ == "__main__":
    main()
