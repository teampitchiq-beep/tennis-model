import csv
import os
import requests
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CURRENT_YEAR = datetime.now().year
YEARS_BACK = 5

HF_BASE = "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/resolve/main"

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


def _download(tour, filename, local_path):
    if os.path.exists(local_path):
        return local_path
    for url in _mirrors(tour, filename):
        try:
            r = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
            if r.status_code != 200 or len(r.content) < 500:
                continue
            first = r.content[:250].decode("utf-8", errors="replace").lower()
            if "tourney_id" not in first and "tourney_date" not in first and \
               "player_id" not in first and "name_last" not in first:
                continue
            host = url.split("/")[2]
            print(f"[dataset] {filename}: OK via {host}")
            with open(local_path, "wb") as f:
                f.write(r.content)
            return local_path
        except Exception:
            continue
    print(f"[dataset] {filename}: all mirrors failed")
    return None


def download_file(tour, filename, force=False):
    path = os.path.join(DATA_DIR, filename)
    if force and os.path.exists(path):
        os.remove(path)
    return _download(tour, filename, path)


def _parse_date(s):
    if not s or len(s) != 8 or not s.isdigit():
        return None
    try:
        return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def _load_csv(path, tour, rows, seen_ids):
    """Load a CSV file, dedupe by (tourney_id, match_num)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                d = _parse_date(r.get("tourney_date"))
                if not d:
                    continue
                key = (r.get("tourney_id"), r.get("match_num"))
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                r["_date"] = d
                r["_tour"] = tour
                rows.append(r)
    except Exception as e:
        print(f"[dataset] parse {os.path.basename(path)}: {e}")


def load_all_matches(force=False):
    """Load tour-level + Challenger + ITF matches from Sackmann archive."""
    rows = []
    seen_ids = set()
    years = list(range(CURRENT_YEAR - YEARS_BACK, CURRENT_YEAR + 1))

    for tour in ("atp", "wta"):
        prefix = "atp" if tour == "atp" else "wta"
        for y in years:
            # Tour-level: atp_matches_2024.csv / wta_matches_2024.csv
            for suffix in ["matches"]:
                filename = f"{prefix}_{suffix}_{y}.csv"
                path = download_file(tour, filename, force=force)
                if path:
                    _load_csv(path, tour, rows, seen_ids)

            # ATP-only: Challenger/qualifying + Futures/ITF
            if tour == "atp":
                for suffix in ["matches_qual_chall", "matches_futures"]:
                    filename = f"{prefix}_{suffix}_{y}.csv"
                    path = download_file(tour, filename, force=force)
                    if path:
                        _load_csv(path, tour, rows, seen_ids)

    if not rows:
        return []

    rows.sort(key=lambda r: r["_date"])
    print(f"[dataset] loaded {len(rows)} matches "
          f"({rows[0]['_date'].date()} → {rows[-1]['_date'].date()})")
    return rows
