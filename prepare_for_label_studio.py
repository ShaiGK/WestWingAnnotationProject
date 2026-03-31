"""
Converts West Wing dialogue .txt files into a JSON file
that Label Studio can import.

Usage:
    python prepare_for_label_studio.py

This script looks in the dialogues/ folder for all .txt files
across all season subfolders, parses each one, and outputs a
single file called label_studio_tasks.json that you will import
into Label Studio.

Run this from your project root folder (WestWingAnnotationProject/).
"""

import os
import json
import glob


def parse_document(text):
    """Parse a single document (metadata + dialogue) into a dict."""
    lines = text.strip().split('\n')

    metadata = {}
    dialogue_lines = []
    in_dialogue = False

    for line in lines:
        if line.strip() == '---DIALOGUE---':
            in_dialogue = True
            continue
        if line.strip() == '---METADATA---':
            in_dialogue = False
            continue

        if not in_dialogue:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        else:
            dialogue_lines.append(line)

    dialogue_text = '\n'.join(dialogue_lines).strip()
    return metadata, dialogue_text


def main():
    # Find all .txt files in dialogues/ subfolders
    dialogue_files = sorted(glob.glob('dialogues/**/*.txt', recursive=True))

    if not dialogue_files:
        print("ERROR: No .txt files found in dialogues/ folder.")
        print("Make sure you run this script from your project root folder")
        print("(the folder that contains the dialogues/ directory).")
        return

    tasks = []
    file_count = 0
    doc_count = 0

    for filepath in dialogue_files:
        file_count += 1
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split file into individual documents (each starts with ---METADATA---)
        raw_docs = content.split('---METADATA---')
        raw_docs = [d for d in raw_docs if d.strip()]

        for raw_doc in raw_docs:
            metadata, dialogue = parse_document('---METADATA---\n' + raw_doc)

            if not dialogue or not metadata.get('character_a'):
                continue

            doc_count += 1

            # Build a unique ID for this document
            doc_id = (
                f"{metadata.get('episode', 'unknown')}_"
                f"{metadata.get('character_a', 'A')}-{metadata.get('character_b', 'B')}_"
                f"pair{metadata.get('pair_instance', '0')}_"
                f"exc{metadata.get('excerpt', '0')}"
            )

            # Format the metadata header that annotators will see
            meta_display = (
                f"Episode: {metadata.get('episode', '?')}  |  "
                f"Character A: {metadata.get('character_a', '?')}  |  "
                f"Character B: {metadata.get('character_b', '?')}  |  "
                f"Pair Instance: {metadata.get('pair_instance', '?')}  |  "
                f"Excerpt: {metadata.get('excerpt', '?')} of {metadata.get('total_excerpts', '?')}  |  "
                f"Back-and-forths: {metadata.get('back_and_forths', '?')}  |  "
                f"Total turns: {metadata.get('total_turns', '?')}"
            )

            # Create the Label Studio task
            task = {
                "data": {
                    "doc_id": doc_id,
                    "meta_display": meta_display,
                    "dialogue": dialogue,
                    "episode": metadata.get('episode', ''),
                    "character_a": metadata.get('character_a', ''),
                    "character_b": metadata.get('character_b', ''),
                    "pair_instance": metadata.get('pair_instance', ''),
                    "excerpt": metadata.get('excerpt', ''),
                    "total_excerpts": metadata.get('total_excerpts', ''),
                    "back_and_forths": metadata.get('back_and_forths', ''),
                    "total_turns": metadata.get('total_turns', ''),
                    "source_file": filepath,
                }
            }
            tasks.append(task)

    # Write output
    output_path = 'label_studio_tasks.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Done! Processed {file_count} files, found {doc_count} documents.")
    print(f"Output saved to: {output_path}")
    print(f"\nNext step: Import {output_path} into your Label Studio project.")


if __name__ == '__main__':
    main()