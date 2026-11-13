"""Entity stimulus lists — REPLACE the placeholders with the actual 90 entities
used in the paper's experiments (from the experimental records).

Chimera examples and synthetic-unknown examples from the paper are pre-filled.
"""
KNOWN = [
    # TODO: 30 real, famous athletes (e.g., "Michael Jordan", "Muhammad Ali", ...)
]

CHIMERA = [
    "Kobe Williams", "Novak Zidane", "Stephen Ronaldo",
    # TODO: remaining 27 chimera names (authentic first + authentic surname of a different athlete)
]

SYNTHETIC_UNKNOWN = [
    "Kasimir Lindqvist", "Orrin Vandermeer",
    # TODO: remaining 28 fictitious athlete names
]

PROMPT_TEMPLATE = "Question: What is the nationality and main career achievement of {name}?\nAnswer:"


def all_prompts():
    for name in KNOWN:
        yield "known", name, PROMPT_TEMPLATE.format(name=name)
    for name in CHIMERA:
        yield "chimera", name, PROMPT_TEMPLATE.format(name=name)
    for name in SYNTHETIC_UNKNOWN:
        yield "unknown", name, PROMPT_TEMPLATE.format(name=name)
