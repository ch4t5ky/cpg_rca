"""
cpg._utils
~~~~~~~~~~
Internal helpers shared across the package.
"""

import html


def unescape(raw: object) -> str:
    """Decode HTML entities and strip surrounding quotes from a DOT attribute value."""
    s = html.unescape(str(raw)).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s
