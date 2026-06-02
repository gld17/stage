import os

CACHE_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0], "../../results/.cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)
