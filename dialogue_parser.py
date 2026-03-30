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
import shutil
from collections import defaultdict


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_episode_id(filename):
    """Pull an episode code like S01E03 from the filename."""
    base = os.path.splitext(os.path.basename(filename))[0]

    m = re.search(r'[Ss](\d+)[Ee](\d+)', base)
    if m:
        return f"S{m.group(1).zfill(2)}E{m.group(2).zfill(2)}"

    m = re.search(r'(\d+)[xX](\d+)', base)
    if m:
        return f"S{m.group(1).zfill(2)}E{m.group(2).zfill(2)}"

    m = re.search(r'Season[_\s-]*(\d+).*?Episode[_\s-]*(\d+)', base, re.IGNORECASE)
    if m:
        return f"S{m.group(1).zfill(2)}E{m.group(2).zfill(2)}"

    m = re.search(r'(\d+)', base)
    if m:
        return f"EP{m.group(1)}"

    return base


def is_character_name(line):
    """
    Return (True, cleaned_name) if the line looks like a character heading.
    Strips bracket/paren tags like [OS] (V.O.) (CONT'D) before checking.
    """
    stripped = line.strip()
    if not stripped:
        return False, None

    name = re.sub(r'[\[\(].*?[\]\)]', '', stripped).strip()
    if not name:
        return False, None

    if (name == name.upper()
            and name[0].isalpha()
            and any(c.isalpha() for c in name)
            and len(name) < 30
            and not name.endswith(':')):
        return True, name

    return False, None


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_blocks(raw_lines):
    """
    Split the file into blocks separated by empty lines.
    Each block is a dict with:
        start_line: first line index (in raw_lines)
        end_line:   last line index
        lines:      list of the raw line strings in this block
    """
    blocks = []
    current_start = None
    current_lines = []

    for i, line in enumerate(raw_lines):
        if line.strip() == '':
            if current_lines:
                blocks.append({
                    'start_line': current_start,
                    'end_line': i - 1,
                    'lines': current_lines,
                })
                current_lines = []
                current_start = None
        else:
            if current_start is None:
                current_start = i
            current_lines.append(line)

    # Last block if file doesn't end with blank line
    if current_lines:
        blocks.append({
            'start_line': current_start,
            'end_line': current_start + len(current_lines) - 1,
            'lines': current_lines,
        })

    return blocks


def classify_blocks(blocks):
    """
    For each block, check if the first line is a character name.
    Returns a list of dicts, each with:
        type:       'dialogue' or 'stage_direction'
        name:       character name (only for dialogue blocks)
        start_line: first line index in the original file
        end_line:   last line index in the original file
    """
    classified = []
    for block in blocks:
        first_line = block['lines'][0]
        is_name, name = is_character_name(first_line)

        if is_name:
            classified.append({
                'type': 'dialogue',
                'name': name,
                'start_line': block['start_line'],
                'end_line': block['end_line'],
            })
        else:
            classified.append({
                'type': 'stage_direction',
                'name': None,
                'start_line': block['start_line'],
                'end_line': block['end_line'],
            })

    return classified


def parse_script(filepath):
    """
    Return (raw_lines, turns) where turns are only the dialogue blocks,
    each with name, start_line, end_line.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw_lines = f.readlines()

    blocks = parse_blocks(raw_lines)
    classified = classify_blocks(blocks)
    turns = [b for b in classified if b['type'] == 'dialogue']

    return raw_lines, turns


def merge_names(turns):
    """
    LEO McGARRY -> LEO.  If a short name is the first word of a longer name,
    map the longer name to the shorter one.
    """
    all_names = sorted(set(t['name'] for t in turns), key=len)
    name_map = {}

    for i, short in enumerate(all_names):
        for long_name in all_names[i + 1:]:
            if long_name.startswith(short + ' ') and long_name not in name_map:
                name_map[long_name] = short

    for t in turns:
        t['name'] = name_map.get(t['name'], t['name'])

    return turns


# ── Dialogue extraction ─────────────────────────────────────────────────────

def find_two_person_dialogues(turns, min_bf=3):
    dialogues = []
    i = 0

    while i < len(turns) - 1:
        char_a = turns[i]['name']

        j = i + 1
        while j < len(turns) and turns[j]['name'] == char_a:
            j += 1
        if j >= len(turns):
            break

        char_b = turns[j]['name']

        run = []
        k = i
        while k < len(turns) and turns[k]['name'] in (char_a, char_b):
            run.append(turns[k])
            k += 1

        count_a = sum(1 for t in run if t['name'] == char_a)
        count_b = sum(1 for t in run if t['name'] == char_b)
        bf = min(count_a, count_b)

        if bf >= min_bf:
            dialogues.append({
                'char_a': min(char_a, char_b),
                'char_b': max(char_a, char_b),
                'turns': run,
                'back_and_forths': bf,
            })
            i = k
        else:
            i += 1

    return dialogues


# ── Splitting ────────────────────────────────────────────────────────────────

def split_dialogue(dialogue, max_bf=6):
    bf = dialogue['back_and_forths']
    if bf <= max_bf:
        return [dialogue]

    num_chunks = math.ceil(bf / max_bf)

    turns = dialogue['turns']
    exchanges = []
    idx = 0
    while idx < len(turns):
        ex = [turns[idx]]
        idx += 1
        if idx < len(turns) and turns[idx]['name'] != ex[0]['name']:
            ex.append(turns[idx])
            idx += 1
        exchanges.append(ex)

    total = len(exchanges)
    base, extra = divmod(total, num_chunks)

    chunks = []
    start = 0
    for c in range(num_chunks):
        size = base + (1 if c < extra else 0)
        chunk_turns = [t for ex in exchanges[start:start + size] for t in ex]
        ca = sum(1 for t in chunk_turns if t['name'] == dialogue['char_a'])
        cb = sum(1 for t in chunk_turns if t['name'] == dialogue['char_b'])
        chunks.append({
            'char_a': dialogue['char_a'],
            'char_b': dialogue['char_b'],
            'turns': chunk_turns,
            'back_and_forths': min(ca, cb),
        })
        start += size

    return chunks


# ── Output ───────────────────────────────────────────────────────────────────

def get_raw_block(raw_lines, turns_in_chunk):
    """
    Grab everything from the first turn's start_line to the last turn's
    end_line — verbatim from the original file.
    """
    first = turns_in_chunk[0]['start_line']
    last = turns_in_chunk[-1]['end_line']
    return ''.join(raw_lines[first:last + 1])


def write_excerpt(output_dir, episode_id, char_a, char_b,
                  excerpt_num, total_excerpts, chunk, pair_count,
                  raw_lines):
    clean = lambda s: re.sub(r'[^A-Za-z0-9]', '', s)
    filename = (f"{episode_id}_{clean(char_a)}-{clean(char_b)}"
                f"_{pair_count:02d}_{excerpt_num:02d}.txt")
    filepath = os.path.join(output_dir, filename)

    raw_block = get_raw_block(raw_lines, chunk['turns'])

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
        f.write(raw_block)

    return filepath


# ── Main ─────────────────────────────────────────────────────────────────────

def main(file=None, output=None, min_bf=3, max_bf=6):
    if file:
        script_file = file
        output_dir = output if output else './dialogues'
    else:
        if len(sys.argv) < 2:
            print("Usage: python extract_dialogues.py <script_file> [output_directory]")
            print()
            print("  script_file      Path to a West Wing episode script (.txt)")
            print("  output_directory  Where to save excerpts (default: ./dialogues)")
            sys.exit(1)

        script_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else './dialogues'

    os.makedirs(output_dir, exist_ok=True)

    episode_id = extract_episode_id(script_file)
    print(f"Episode:  {episode_id}")
    print(f"Parsing:  {script_file}")

    raw_lines, turns = parse_script(script_file)
    print(f"Found {len(turns)} dialogue turns")

    turns = merge_names(turns)

    dialogues = find_two_person_dialogues(turns, min_bf=min_bf)
    print(f"Found {len(dialogues)} two-person dialogues (>={min_bf} back-and-forths)")

    pair_counter = defaultdict(int)
    files_written = []

    for d in dialogues:
        key = tuple(sorted([d['char_a'], d['char_b']]))
        pair_counter[key] += 1
        chunks = split_dialogue(d, max_bf=max_bf)

        for i, chunk in enumerate(chunks, start=1):
            path = write_excerpt(
                output_dir, episode_id,
                d['char_a'], d['char_b'],
                i, len(chunks),
                chunk, pair_counter[key],
                raw_lines,
            )
            files_written.append(path)

    print(f"\nWrote {len(files_written)} files to {output_dir}/")
    for f in files_written:
        print(f"  {os.path.basename(f)}")


def parse_all_dialogues(script_dir, output_dir, min_bf=3, max_bf=6):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for season_dir in os.listdir(script_dir):
        if not os.path.isdir(os.path.join(script_dir, season_dir)):
            continue
        for filename in os.listdir(os.path.join(script_dir, season_dir)):
            if filename.endswith('.txt'):
                script_path = os.path.join(script_dir, season_dir, filename)
                main(script_path, os.path.join(output_dir, season_dir), min_bf=min_bf, max_bf=max_bf)


if __name__ == '__main__':
    # Example usage
    # main("episode_scripts/season_1/Season_1_Episode_1_Pilot.txt", "dialogues/season_1", min_bf=3, max_bf=10)
    parse_all_dialogues("scripts", "dialogues", min_bf=4, max_bf=8)