import csv
import os
import re
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

HF_BASE = "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/resolve/main"

_MEM_INDEX = None
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def _mirrors(tour, filename):
    folder = "atp" if tour == "atp" else "wta"
    repo = "tennis_atp" if tour == "atp" else "tennis_wta"
    return [
        f"{HF_BASE}/{folder}/{filename}?download=true",
        f"https://gh-proxy.com/https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{filename}",
        f"https://ghfast.top/https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{filename}",
        f"https://xget.xi-xu.me/gh/JeffSackmann/{repo}/raw/refs/heads/master/{filename}",
        f"https://raw.githubusercontent.com/JeffSackmann/{repo}/master/{filename}",
    ]


def _download(tour):
    filename = "atp_players.csv" if tour == "atp" else "wta_players.csv"
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return path
    for url in _mirrors(tour, filename):
        try:
            r = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
            if r.status_code != 200 or len(r.content) < 500:
                continue
            first = r.content[:250].decode("utf-8", errors="replace").lower()
            if "player_id" not in first and "name_last" not in first:
                continue
            host = url.split("/")[2]
            print(f"[players] {filename}: OK via {host}")
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        except Exception:
            continue
    print(f"[players] {filename}: all mirrors failed")
    return None


def _load_players(tour):
    path = _download(tour)
    if not path:
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _name_key(first, last):
    if not last:
        return None
    ln = re.sub(r"[^a-z]", "", last.lower())
    if not ln:
        return None
    init = (first or "").strip()[:1].lower()
    return f"{ln}_{init}"


def _surname(last):
    return re.sub(r"[^a-z]", "", (last or "").lower())


def build_index():
    """Index of every known player: key and surname -> [(player_id, tour)].

    Unlike the original (which kept only the lowest player_id per key, i.e.
    typically an old/retired namesake, and mixed ATP with WTA), all candidates
    are kept so resolve() can pick the active one on the right tour.
    """
    global _MEM_INDEX
    if _MEM_INDEX is not None:
        return _MEM_INDEX

    by_key = {}
    by_surname = {}

    for tour in ("atp", "wta"):
        rows = _load_players(tour)
        print(f"[players] {tour}: {len(rows)} profiles")
        for row in rows:
            try:
                pid = int(row.get("player_id", 0))
            except (ValueError, TypeError):
                continue
            first = row.get("name_first", "") or ""
            last = row.get("name_last", "") or ""

            key = _name_key(first, last)
            if key:
                by_key.setdefault(key, []).append((pid, tour))

            sn = _surname(last)
            if sn:
                by_surname.setdefault(sn, []).append((pid, tour))

    print(f"[players] index: {len(by_key)} keys, {len(by_surname)} surnames")
    _MEM_INDEX = {"by_key": by_key, "by_surname": by_surname}
    return _MEM_INDEX


def _pick(cands, tour=None, ratings=None):
    """Choose one player id from [(pid, tour), ...]."""
    if not cands:
        return None
    if tour:
        same = [c for c in cands if c[1] == tour]
        if same:
            cands = same
    if len(cands) == 1:
        return cands[0][0]
    if ratings:
        def activity(c):
            e = ratings.get(c[0])
            if e is None:
                return ("", -1)
            return (e.get("last", ""), e.get("n", 0))
        best = max(cands, key=activity)
        if activity(best)[1] >= 0:
            return best[0]
    return min(c[0] for c in cands)


def resolve(display_name, index=None, tour=None, ratings=None):
    """
    Map a display name to a Sackmann player_id.

    tour:    'atp' / 'wta' if known (disambiguates namesakes across tours).
    ratings: Elo ratings dict {pid: {..., 'last', 'n'}}; among namesakes the
             most recently active one is chosen.

    Examples:
      'Zverev A.'   -> 100644
      'Zverev'      -> 100644
    """
    if not display_name:
        return None
    if index is None:
        index = build_index()

    by_key = index["by_key"]
    by_surname = index["by_surname"]

    tokens = [t.strip(".").lower() for t in display_name.split() if t]
    tokens = [re.sub(r"[^a-z]", "", t) for t in tokens]
    tokens = [t for t in tokens if t]

    if not tokens:
        return None

    if len(tokens) == 1:
        return _pick(by_surname.get(tokens[0], []), tour, ratings)

    initial = tokens[-1][:1] if tokens[-1] else ""
    surname_parts = tokens[:-1]

    for take in range(len(surname_parts), 0, -1):
        surname = "".join(surname_parts[-take:])
        key = f"{surname}_{initial}"
        if key in by_key:
            return _pick(by_key[key], tour, ratings)

    full_surname = "".join(surname_parts)
    return _pick(by_surname.get(full_surname, []), tour, ratings)
