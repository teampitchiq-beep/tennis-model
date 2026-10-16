import os
from livetennisapi import LiveTennisAPI

api_key = os.environ.get("LIVETENNISAPI_KEY")

if not api_key:
    print("Please set your LIVETENNISAPI_KEY environment variable.")
    exit()

with LiveTennisAPI(api_key=api_key) as client:
    # 1. Get upcoming fixtures
    fixtures = client.list_fixtures(tour="atp")
    print(f"Found {len(fixtures)} upcoming ATP fixtures.\n")

    for match in fixtures[:3]:  # Show first 3
        print(f"{match.player1_name} vs {match.player2_name}")
        print(f"Tournament: {match.tournament} ({match.surface})")
        print(f"Round: {match.round}")
        print(f"Start: {match.start_time}")
        print("---")

        # 2. Fetch full player profiles using the IDs
        p1 = client.get_player(match.player1_id)
        p2 = client.get_player(match.player2_id)

        # Print available player fields to see what we can use
        print(f"{p1.name} (id={p1.id})")
        print(f"  Fields: {list(p1.__dict__.keys())}")
        print(f"{p2.name} (id={p2.id})")
        print(f"  Fields: {list(p2.__dict__.keys())}")
        print("==========\n")
