#!/usr/bin/env python3
"""
Extract two-character dialogues from West Wing script files.

Usage:
    python extract_dialogues.py <script_file> [output_directory]

Examples:
    python extract_dialogues.py west_wing_S01E01.txt
    python extract_dialogues.py scripts/1x03_pilot.txt ./my_output
"""

import re
import os
import sys
import math
from collections import defaultdict


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_episode_id(filename):
    """Pull an episode code like S01E03 from the filename."""
    base = os.path.splitext(os.path.basename(filename))[0]

    # S01E01 or s1e3
    m = re.search(r'[Ss](\d+)[Ee](\d+)', base)
    if m:
        return f"S{m.group(1).zfill(2)}E{m.group(2).zfill(2)}"

    # 1x01
    m = re.search(r'(\d+)[xX](\d+)', base)
    if m:
        return f"S{m.group(1).zfill(2)}E{m.group(2).zfill(2)}"

    # Bare number(s) – take the first as episode
    m = re.search(r'(\d+)', base)
    if m:
        return f"EP{m.group(1)}"

    return base                       # fall back to whole filename


STAGE_DIR_STARTS = re.compile(
    r'^(CUT TO|INT[\.\s]|EXT[\.\s]|FADE|DISSOLVE|SMASH CUT|'
    r'CONTINUED|END OF|COLD OPEN|ACT\s|TEASER|PREVIOUSLY|'
    r'TITLE|CREDITS|THE END|BLACKOUT)', re.IGNORECASE
)


def is_character_name(line):
    """
    Return (True, cleaned_name) if *line* looks like a character heading,
    otherwise (False, None).
    """
    stripped = line.strip()
    if not stripped:
        return False, None

    # Remove parenthetical / bracket tags like [OS]  (V.O.)  (CONT'D)
    name = re.sub(r'[\[\(].*?[\]\)]', '', stripped).strip()
    if not name:
        return False, None

    if STAGE_DIR_STARTS.match(name):
        return False, None

    # Must be ALL-CAPS, start with a letter, reasonably short, no trailing ':'
    if (name == name.upper()
            and name[0].isalpha()
            and any(c.isalpha() for c in name)
            and len(name) < 30
            and not name.endswith(':')):
        return True, name

    return False, None


def is_stage_direction(line):
    """True for lines like CUT TO:, INT. OVAL OFFICE, etc."""
    return bool(STAGE_DIR_STARTS.match(line.strip()))


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_script(filepath):
    """Return a list of (character_name, dialogue_text) tuples."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    turns = []
    current_char = None
    current_lines = []

    for line in lines:
        stripped = line.strip()
        is_name, name = is_character_name(stripped)

        if is_name:
            # Save the turn we were building
            if current_char and current_lines:
                turns.append((current_char, '\n'.join(current_lines).strip()))
            current_char = name
            current_lines = []

        elif current_char:
            if is_stage_direction(stripped):
                # Stage direction breaks the current turn
                if current_lines:
                    turns.append((current_char, '\n'.join(current_lines).strip()))
                current_char = None
                current_lines = []
            elif stripped:
                current_lines.append(stripped)

    # Last turn
    if current_char and current_lines:
        turns.append((current_char, '\n'.join(current_lines).strip()))

    return turns


def merge_names(turns):
    """
    LEO McGARRY ➜ LEO.  If a short name is the first word of a longer name,
    map the longer name to the shorter one everywhere.
    """
    all_names = sorted(set(n for n, _ in turns), key=len)
    name_map = {}

    for i, short in enumerate(all_names):
        for long_name in all_names[i + 1:]:
            if long_name.startswith(short + ' ') and long_name not in name_map:
                name_map[long_name] = short

    return [(name_map.get(n, n), d) for n, d in turns]


# ── Dialogue extraction ─────────────────────────────────────────────────────

def find_two_person_dialogues(turns, min_bf=3):
    """
    Walk through the turn list and collect every maximal run where
    exactly two characters alternate.  Return only those with at least
    *min_bf* back-and-forths (= each character speaks at least min_bf times).
    """
    dialogues = []
    i = 0

    while i < len(turns) - 1:
        char_a = turns[i][0]

        # Find the second distinct character
        j = i + 1
        while j < len(turns) and turns[j][0] == char_a:
            j += 1
        if j >= len(turns):
            break

        char_b = turns[j][0]

        # Collect the full run of just these two characters
        run = []
        k = i
        while k < len(turns) and turns[k][0] in (char_a, char_b):
            run.append(turns[k])
            k += 1

        count_a = sum(1 for n, _ in run if n == char_a)
        count_b = sum(1 for n, _ in run if n == char_b)
        bf = min(count_a, count_b)

        char_a, char_b = sorted([char_a, char_b])

        if bf >= min_bf:
            dialogues.append({
                'char_a': char_a,
                'char_b': char_b,
                'turns': run,
                'back_and_forths': bf,
            })
            i = k          # skip past this whole dialogue
        else:
            i += 1         # slide forward and try again

    return dialogues


# ── Splitting ────────────────────────────────────────────────────────────────

def split_dialogue(dialogue, max_bf=6):
    """
    If the dialogue has more than *max_bf* back-and-forths, split it into
    roughly-equal pieces that each have at most *max_bf*.
    7 b/f  ➜  4 + 3      (2 chunks)
    15 b/f ➜  5 + 5 + 5  (3 chunks)
    """
    bf = dialogue['back_and_forths']
    if bf <= max_bf:
        return [dialogue]

    num_chunks = math.ceil(bf / max_bf)

    # Group turns into "exchanges" – each exchange starts when the speaker
    # changes, so typically [A-turn, B-turn].
    turns = dialogue['turns']
    exchanges = []
    idx = 0
    while idx < len(turns):
        ex = [turns[idx]]
        idx += 1
        # Grab the response from the other character (if present)
        if idx < len(turns) and turns[idx][0] != ex[0][0]:
            ex.append(turns[idx])
            idx += 1
        exchanges.append(ex)

    # Distribute exchanges as evenly as possible across chunks
    total = len(exchanges)
    base, extra = divmod(total, num_chunks)

    chunks = []
    start = 0
    for c in range(num_chunks):
        size = base + (1 if c < extra else 0)
        chunk_turns = [t for ex in exchanges[start:start + size] for t in ex]
        ca = sum(1 for n, _ in chunk_turns if n == dialogue['char_a'])
        cb = sum(1 for n, _ in chunk_turns if n == dialogue['char_b'])
        chunks.append({
            'char_a': dialogue['char_a'],
            'char_b': dialogue['char_b'],
            'turns': chunk_turns,
            'back_and_forths': min(ca, cb),
        })
        start += size

    return chunks


# ── Output ───────────────────────────────────────────────────────────────────

def write_excerpt(output_dir, episode_id, char_a, char_b,
                  excerpt_num, total_excerpts, chunk, pair_count):
    """
    Write one excerpt file.  Filename pattern:
        S01E01_SAM-BILLY_01_01.txt
        episode_charA-charB_pairInstance_excerptNumber
    """
    clean = lambda s: re.sub(r'[^A-Za-z0-9]', '', s)
    filename = (f"{episode_id}_{clean(char_a)}-{clean(char_b)}"
                f"_{pair_count:02d}_{excerpt_num:02d}.txt")
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("---METADATA---\n")
        f.write(f"episode: {episode_id}\n")
        f.write(f"character_a: {char_a}\n")
        f.write(f"character_b: {char_b}\n")
        f.write(f"pair_instance: {pair_count}\n")
        f.write(f"excerpt: {excerpt_num}\n")
        f.write(f"total_excerpts: {total_excerpts}\n")
        f.write(f"back_and_forths: {chunk['back_and_forths']}\n")
        f.write(f"total_turns: {len(chunk['turns'])}\n")
        f.write("---DIALOGUE---\n")
        for name, text in chunk['turns']:
            f.write(f"{name}: {text}\n")

    return filepath


# ── Main ─────────────────────────────────────────────────────────────────────

def main(file=None, output=None, min_bf=3, max_bf=6):
    if file:
        script_file = file
        output_dir  = output if output else './dialogues'
    else:
        if len(sys.argv) < 2:
            print("Usage: python extract_dialogues.py <script_file> [output_directory]")
            print()
            print("  script_file      Path to a West Wing episode script (.txt)")
            print("  output_directory  Where to save excerpts (default: ./dialogues)")
            sys.exit(1)

        script_file = sys.argv[1]
        output_dir  = sys.argv[2] if len(sys.argv) > 2 else './dialogues'

    os.makedirs(output_dir, exist_ok=True)

    episode_id = extract_episode_id(script_file)
    print(f"Episode:  {episode_id}")
    print(f"Parsing:  {script_file}")

    turns = parse_script(script_file)
    print(f"Found {len(turns)} dialogue turns")

    turns = merge_names(turns)

    dialogues = find_two_person_dialogues(turns, min_bf=min_bf)
    print(f"Found {len(dialogues)} two-person dialogues (≥{min_bf} back-and-forths)")

    pair_counter = defaultdict(int)
    files_written = []

    for d in dialogues:
        key = tuple(sorted([d['char_a'], d['char_b']]))
        pair_counter[key] += 1

        for i, chunk in enumerate(split_dialogue(d, max_bf=max_bf), start=1):
            path = write_excerpt(
                output_dir, episode_id,
                d['char_a'], d['char_b'],
                i, len(split_dialogue(d, max_bf=max_bf)),
                chunk, pair_counter[key],
            )
            files_written.append(path)

    print(f"\nWrote {len(files_written)} files to {output_dir}/")
    for f in files_written:
        print(f"  {os.path.basename(f)}")


if __name__ == '__main__':
    # Example usage
    main("scripts/S1E1.txt", "dialogues", min_bf=2, max_bf=10)