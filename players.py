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
                if key not in by_key or pid < by_key[key]:
                    by_key[key] = pid

            sn = _surname(last)
            if sn:
                by_surname.setdefault(sn, []).append(pid)

    print(f"[players] index: {len(by_key)} keys, {len(by_surname)} surnames")
    _MEM_INDEX = {"by_key": by_key, "by_surname": by_surname}
    return _MEM_INDEX


def resolve(display_name, index=None):
    """
    Examples:
      'Zverev A.'   -> 20010422
      'Zverev'      -> 20010422
      'Rinderknech' -> 20000065
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
        pids = by_surname.get(tokens[0], [])
        return min(pids) if pids else None

    initial = tokens[-1][:1] if tokens[-1] else ""
    surname_parts = tokens[:-1]

    for take in range(len(surname_parts), 0, -1):
        surname = "".join(surname_parts[-take:])
        key = f"{surname}_{initial}"
        if key in by_key:
            return by_key[key]

    full_surname = "".join(surname_parts)
    pids = by_surname.get(full_surname, [])
    return min(pids) if pids else None
