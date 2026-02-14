import os

def _resolve_path(p: str, base_dir: str) -> str:
    p = p.strip().strip('"').strip("'").strip()
    p = p.lstrip("\ufeff")
    p = p.replace("\\", "/")

    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(base_dir, p))
    else:
        p = os.path.abspath(p)

    p = os.path.normpath(p).replace("\\", "/")
    p = p.replace("/processed/processed/", "/processed/")

    return p


def _clean_path_str(p: str) -> str:
    p = p.strip().strip('"').strip("'").strip()
    p = p.lstrip("\ufeff")
    p = p.replace("\\", "/")
    return p
