from unittest.mock import patch

import jax.numpy as jnp

from generals.remote.generalsio_client import BOT_ENDPOINT, GeneralsIOClient, PUBLIC_ENDPOINT


class RightAgent:
    def reset(self):
        self.reset_called = True

    def act(self, observation, key):
        return jnp.array([0, 1, 2, 3, 0])


def test_bot_endpoint_and_rectangular_indexing():
    agent = RightAgent()
    with patch.object(GeneralsIOClient, "connect") as connect:
        client = GeneralsIOClient(agent, "test-user")
    connect.assert_called_once_with(BOT_ENDPOINT)
    client._initialize_game(({"usernames": ["us", "them"], "playerIndex": 0, "replay_id": "test"},))
    assert agent.reset_called
    client.game_state.map = [5, 4]
    assert client._generate_action(None) == (7, 8, 0)



def test_public_endpoint_is_explicit():
    with patch.object(GeneralsIOClient, "connect") as connect:
        GeneralsIOClient(RightAgent(), "test-user", endpoint=PUBLIC_ENDPOINT)
    connect.assert_called_once_with(PUBLIC_ENDPOINT)
def test_single_payload_update_does_not_fake_a_win():
    with patch.object(GeneralsIOClient, "connect"):
        client = GeneralsIOClient(RightAgent(), "test-user")
    client._initialize_game(({"usernames": ["us", "them"], "playerIndex": 0, "replay_id": "test"},))
    width, height = 5, 4
    armies = [0] * (width * height)
    armies[7] = 3
    terrain = [-1] * (width * height)
    terrain[7] = 0
    update = {
        "turn": 1,
        "map_diff": [0, 2 + 2 * len(armies), width, height, *armies, *terrain],
        "cities_diff": [0, 0],
        "generals": [7, -1],
        "scores": [{"i": 0, "tiles": 1, "total": 3}, {"i": 1, "tiles": 0, "total": 0}],
    }
    events = iter([("game_update", update), ("game_lost",)])
    client.receive = lambda: next(events)
    sent = []
    client.emit = lambda name, *args: sent.append((name, *args))
    client._play_game()
    assert ("attack", (7, 8, 0)) in sent
    assert ("leave_game",) in sent
    assert client._score_wins == 0 and client._score_losses == 1
