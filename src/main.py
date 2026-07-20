"""
Command line runner for the Music Recommender Simulation.

Runs the functional path end to end:
    load_songs -> recommend_songs (which uses score_song as the per-song judge)
"""

from src.recommender import load_songs, recommend_songs


def format_profile(profile: dict) -> str:
    """Render the taste profile as a compact, single-line summary."""
    return ", ".join(f"{key}={value}" for key, value in profile.items())


def print_recommendations(profile: dict, recommendations) -> None:
    """Print recommendations in a clean, readable terminal layout."""
    divider = "-" * 64
    print("\n🎵  Music Recommender — your top picks\n")
    print(f"Taste profile: {format_profile(profile)}")
    print(divider)
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{rank}. {song['title']} — {song['artist']}  [{song['genre']} · {song['mood']}]")
        print(f"   Score: {score:.2f}")
        print("   Why:")
        for reason in explanation.split("; "):
            print(f"     • {reason}")
        print(divider)


def main() -> None:
    songs = load_songs("data/songs.csv")

    # Taste profile for the functional path. Every key is optional — a missing
    # one simply contributes 0 to the score.
    taste_profile = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.40,
        "acousticness": 0.80,
        "valence": 0.55,
        "danceability": 0.40,
        "tempo": 78,  # BPM — a relaxed study-beat tempo
    }

    recommendations = recommend_songs(taste_profile, songs, k=5)
    print_recommendations(taste_profile, recommendations)


if __name__ == "__main__":
    main()
