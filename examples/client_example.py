"""
Run a reference agent against local, bot, or explicitly selected public servers.

The official FAQ restricts bots to its bot server. Public matchmaking with
--ranked is opt-in; check current server rules before using your account.
"""

import argparse

from generals.remote import GeneralsIOClient, autopilot
from generals.agents import ExpanderAgent


def main():
    parser = argparse.ArgumentParser(description="Run agent on generals.io")
    parser.add_argument("--user_id", type=str, required=True, help="Your generals.io user ID")
    parser.add_argument("--lobby_id", type=str, default="bot_test", help="Lobby ID to join")
    parser.add_argument("--endpoint", default="https://botws.generals.io/", help="Socket.IO URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--ranked", action="store_true", help="Opt into a single public 1v1 queue game")
    args = parser.parse_args()

    agent = ExpanderAgent()
    if args.ranked:
        if args.endpoint.rstrip("/") != "https://ws.generals.io":
            parser.error("--ranked requires --endpoint https://ws.generals.io/")
        with GeneralsIOClient(agent, args.user_id, endpoint=args.endpoint) as client:
            client.join_1v1_queue()
    else:
        print(f"Starting {agent.id} agent in lobby '{args.lobby_id}'...")
        autopilot(agent, args.user_id, args.lobby_id, endpoint=args.endpoint)


if __name__ == "__main__":
    main()
