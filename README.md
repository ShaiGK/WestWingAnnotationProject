# West Wing Power Dynamics Annotation Project

A corpus annotation project studying conversational power dynamics between characters in *The West Wing*. We extract paired dialogue excerpts from episode scripts and annotate them for assertiveness, dominance, and conversational strategies.

For full annotation instructions, see `annotation_guidelines.md`. For a record of guideline revisions, see `guidelines_changelog.md`.

---

## Repository Structure

```
WestWingAnnotationProject/
├── dialogues/                    # Extracted dialogue excerpts (gitignored, generated locally)
├── scripts/                      # Raw episode scripts (gitignored, generated locally)
├── annotations/
│   └── all_annotations.json      # Shared annotation output (auto-updated)
├── annotate.py                   # Annotation workflow script
├── annotation_guidelines.md      # Full guidelines for annotators
├── dialogue_parser.py            # Splits scripts into paired dialogues
├── guidelines_changelog.md       # Record of guideline changes
├── label_studio_config.xml       # Label Studio interface configuration
├── prepare_for_label_studio.py   # Converts dialogues to Label Studio format
├── scrape_scripts.py             # Scrapes episode scripts from the web
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- pip

### 1. Clone the repo

```bash
git clone <repo-url>
cd WestWingAnnotationProject
```

### 2. Install dependencies

```bash
pip install label-studio playwright
playwright install
```

---

## Data Pipeline (every team member does this once)

The `scripts/` and `dialogues/` folders are gitignored since they're large and easily regenerated. After cloning, you need to run these two steps to produce the dialogue data locally.

### Scrape episode scripts

```bash
python scrape_scripts.py
```

You will be prompted to choose how many seasons to scrape. Scripts are saved to the `scripts/` folder. **This takes 10–15 minutes** since it downloads each episode individually.

### Parse scripts into dialogue excerpts

```bash
python dialogue_parser.py
```

This splits the scraped scripts into paired character dialogues and saves them to the `dialogues/` folder. Takes a few seconds. The minimum and maximum back-and-forths per excerpt are set as variables in the code — no need to change them unless you're a group member adjusting the extraction parameters.

---

## Annotation Setup (every team member does this once)

### 1. Generate the Label Studio import file

```bash
python prepare_for_label_studio.py
```

This reads all dialogue files in `dialogues/` and creates `label_studio_tasks.json` (gitignored since it's large and can always be regenerated).

### 2. Start Label Studio and create an account

```bash
label-studio start
```

This opens Label Studio in your browser at http://localhost:8080. Create an account with any email and password — this is a local account on your machine only.

### 3. Copy your Access Token

In Label Studio's web interface, click the person icon in the top-right corner, then click **Account & Settings**. Copy the **Access Token** shown on that page.

### 4. Close Label Studio

Go back to your terminal and press `Ctrl+C` to stop Label Studio.

### 5. Run the setup command

```bash
python annotate.py setup
```

Enter your first name when prompted (e.g., `nathan`) and paste your Access Token. This saves your config locally in `.annotate_config.json` (gitignored automatically).

---

## Annotating

### Start a session

```bash
python annotate.py start
```

This will:
1. Pull the latest annotations from git
2. Filter out documents already annotated by anyone on the team
3. Load a random batch of fresh documents into Label Studio
4. Open Label Studio in your browser

Click **Label All Tasks** and annotate as many documents as you want. Make sure to click **Submit** after each one.

### Finish a session

When you're done, go back to your terminal and run:

```bash
python annotate.py finish
```

This will:
1. Export your annotations from Label Studio
2. Merge them into `annotations/all_annotations.json`
3. Commit and push to git

### Check progress

```bash
python annotate.py status
```

Shows total documents in the corpus, how many have been annotated, and a per-annotator breakdown.

---

## Notes

- **Read `annotation_guidelines.md` before you start annotating.** It explains the rating scale, edge cases, and examples.
- The annotation interface includes: a 5-point power dynamic rating, a power shift flag, a power expression checklist, a document validity flag, and an optional notes field.
- If `git push` fails during `annotate.py finish`, run `git pull` and then `python annotate.py finish` again — a teammate likely pushed while you were annotating.
- If Label Studio doesn't start automatically, open a second terminal window, run `label-studio start` there, then run `python annotate.py start` in your original window.