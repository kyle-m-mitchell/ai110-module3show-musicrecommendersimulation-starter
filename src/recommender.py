import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# --- Scoring configuration -------------------------------------------------
# Weights control how much each feature can contribute to a song's score.
# Genre is the decisive signal (see README): at 3.0 it stays dominant while
# mood and the numeric features fine-tune the ranking within a genre.
WEIGHTS = {
    "genre": 3.0,
    "mood": 1.5,
    "energy": 1.0,
    "acousticness": 1.0,
    "valence": 1.0,
    "danceability": 0.5,
}

# Similarity families give partial credit for "cousin" genres/moods so the
# recommender degrades gracefully (a lofi fan sees ambient/jazz before metal).
# Each genre/mood belongs to exactly one family.
GENRE_FAMILIES = {
    "mellow":     {"lofi", "ambient", "jazz", "classical"},
    "pop_elec":   {"pop", "indie pop", "synthwave", "edm"},
    "rock_heavy": {"rock", "metal"},
    "roots":      {"country", "folk", "blues"},
    "groove":     {"hip hop", "r&b", "funk", "reggae"},
}

MOOD_FAMILIES = {
    "calm":     {"chill", "relaxed", "focused"},
    "upbeat":   {"happy", "uplifting", "euphoric", "playful", "hopeful"},
    "intense":  {"intense", "energetic", "aggressive"},
    "somber":   {"moody", "melancholy", "somber", "nostalgic"},
    "romantic": {"romantic"},
}

# Reverse lookups (member -> family) so the category helpers are O(1).
GENRE_TO_FAMILY = {g: fam for fam, members in GENRE_FAMILIES.items() for g in members}
MOOD_TO_FAMILY = {m: fam for fam, members in MOOD_FAMILIES.items() for m in members}

# Numeric features scored by closeness = 1 - |target - value|.
NUMERIC_FEATURES = ("energy", "acousticness", "valence", "danceability")


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_acousticness: float = 0.5
    target_valence: float = 0.5
    target_danceability: float = 0.5

def _profile_to_prefs(user: UserProfile) -> Dict:
    """Adapt a UserProfile object to the dict shape score_song expects."""
    return {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "acousticness": user.target_acousticness,
        "valence": user.target_valence,
        "danceability": user.target_danceability,
    }


class Recommender:
    """
    OOP wrapper around the same scoring core used by the functional path.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Store the catalog of Song objects to recommend from."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k Song objects for the user, highest score first."""
        # Reuse the functional core: score + rank as dicts, then map back to Songs.
        prefs = _profile_to_prefs(user)
        ranked = recommend_songs(prefs, [asdict(s) for s in self.songs], k)
        by_id = {s.id: s for s in self.songs}
        return [by_id[song["id"]] for song, _score, _why in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a one-line explanation of how the song scored for the user."""
        prefs = _profile_to_prefs(user)
        score, reasons = score_song(prefs, asdict(song))
        detail = "; ".join(reasons) if reasons else "no strong matches with your profile"
        return f"{song.title} scored {score:.2f}: {detail}"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file into a list of dictionaries.

    Numeric columns are converted so we can do math on them later:
    - id                                              -> int
    - energy, tempo_bpm, valence,
      danceability, acousticness                      -> float
    The remaining columns (title, artist, genre, mood) stay as strings.

    Required by src/main.py
    """
    int_fields = {"id"}
    float_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}

    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)
    return songs

def _category_score(pref_value: Optional[str], song_value: Optional[str],
                    mapping: Dict[str, str]) -> float:
    """Categorical match: 1.0 exact, 0.5 same family, 0.0 otherwise."""
    if not pref_value or not song_value:
        return 0.0
    if pref_value == song_value:
        return 1.0
    pref_family = mapping.get(pref_value)
    song_family = mapping.get(song_value)
    if pref_family is not None and pref_family == song_family:
        return 0.5
    return 0.0


def _genre_score(pref_genre: Optional[str], song_genre: Optional[str]) -> float:
    """Genre similarity: 1.0 exact, 0.5 same family, 0.0 otherwise."""
    return _category_score(pref_genre, song_genre, GENRE_TO_FAMILY)


def _mood_score(pref_mood: Optional[str], song_mood: Optional[str]) -> float:
    """Mood similarity: 1.0 exact, 0.5 same family, 0.0 otherwise."""
    return _category_score(pref_mood, song_mood, MOOD_TO_FAMILY)


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against the user's taste profile.

    Applies the algorithm recipe:
        score = 3.0*genre + 1.5*mood + 1.0*energy + 1.0*acousticness
              + 1.0*valence + 0.5*danceability
    where genre/mood use similarity families (1.0 exact / 0.5 cousin / 0 else)
    and each numeric feature scores its closeness = 1 - |target - value|.

    Every profile key is optional; a missing key simply contributes 0.

    Returns (score, reasons), where reasons lists each feature that added
    points so the user can see *why* a song was recommended.
    """
    score = 0.0
    reasons: List[str] = []

    # --- Categorical: genre, mood (exact 1.0 / cousin 0.5 / else 0) ---
    genre_sub = _genre_score(user_prefs.get("genre"), song.get("genre"))
    if genre_sub > 0:
        points = WEIGHTS["genre"] * genre_sub
        score += points
        if genre_sub == 1.0:
            reasons.append(f"genre match ({song['genre']}) +{points:.2f}")
        else:
            reasons.append(f"genre cousin of {user_prefs['genre']} ({song['genre']}) +{points:.2f}")

    mood_sub = _mood_score(user_prefs.get("mood"), song.get("mood"))
    if mood_sub > 0:
        points = WEIGHTS["mood"] * mood_sub
        score += points
        if mood_sub == 1.0:
            reasons.append(f"mood match ({song['mood']}) +{points:.2f}")
        else:
            reasons.append(f"mood cousin of {user_prefs['mood']} ({song['mood']}) +{points:.2f}")

    # --- Numeric: closeness = 1 - |target - value|, scaled by weight ---
    for feature in NUMERIC_FEATURES:
        target = user_prefs.get(feature)
        value = song.get(feature)
        if target is None or value is None:
            continue
        closeness = max(0.0, 1.0 - abs(target - value))
        points = WEIGHTS[feature] * closeness
        if points > 0:
            score += points
            reasons.append(f"{feature} fit (target {target}, song {value}) +{points:.2f}")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Ranks the whole catalog against the taste profile and returns the top k.

    score_song is the per-song "judge": every song in the catalog is scored
    first (local step). recommend_songs is the global step — once every song
    has a number, it sorts them all from highest to lowest score (ties broken
    by id so the order is stable) and keeps the best k.

    Returns a list of (song, score, explanation), where explanation is the
    song's reasons joined into one readable string.
    """
    # Judge every song. score_song returns (score, reasons); the * unpacks that
    # tuple so each item becomes (song, score, reasons).
    scored = [(song, *score_song(user_prefs, song)) for song in songs]

    # sorted() returns a NEW list without touching the caller's `songs`.
    # key = (-score, id): negative score => highest first; id => ascending tie-break.
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]["id"]))

    return [
        (song, score, "; ".join(reasons) if reasons else "no strong matches")
        for song, score, reasons in ranked[:k]
    ]
