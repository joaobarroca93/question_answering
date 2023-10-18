import hashlib


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
