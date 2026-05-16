# Terminal-First Forensic Protocol: The "Full Caboodle"

This protocol documents the most efficient method for an AI to interrogate the "Acoustic DNA" of an audio event detection file. It prioritizes local logic over live API grammar.

## The Superiority of Terminal-First Logic

While a live Dash server provides real-time interrogation, compiling the "Acoustic Brain" directly in the terminal is the most **AI-Friendly** forensic method.

- **Logic vs. Grammar:** Bypasses complex Dash/JSON path interrogation and "the dots" serialization traps.
- **Scale Clarity:** Automatically handles the "Scale Trap" by computing contributions and presenting them in a human-readable list.
- **Context Density:** Provides the Top 15 predictors (The "Acoustic DNA") in a single tool call rather than multiple chained `jq` requests.

## The Forensic Command

Use the following command to determine what "makes a sound what it is" (e.g., why the model thinks a window break is a "Splash" or a vocal is "Heavy Metal"):

```bash
python3 scripts/Shapash_visualization/launch_correlations_dashboard.py <PATH_TO_CSV> --target "<SOUND_CLASS>"
```

### Parameters:
- `<PATH_TO_CSV>`: Path to the `full_event_log.csv` artifact.
- `--target`: The specific sound class to explain (use the exact name from the vocabulary discovery).

## Interrogating the "Acoustic DNA"

A single label is a hypothesis; the Top 15 predictors are the **Acoustic DNA**.

### The Literal Signature
If the target class is explained by its expected physical correlates, accept the **Literal Interpretation**.
- **Example:** "Heavy Metal" is predicted by "Bass guitar" + "Cacophony" + "Bellow."

### The Forensic Pivot (Metaphor Discovery)
If the correlates are contradictory or unexpected, look for the **Mathematical Metaphor**.
- **Example:** "Splash, splatter" predicted by "Single-lens reflex camera" + "Wind" suggests a high-frequency impact (Falling Glass) rather than water.

## Archeological Baseline

Before running the "Full Caboodle," always read the **AI-Friendly Static Files** to "Read the Room":
- **Summary Events** (`summary_events.csv`): Identifies narrative phases (e.g., Intro, Solo, Confrontation).
- **Detailed Delta JSON** (`detailed_events_delta_ai_attention_friendly.json`): Reveals the pulse and momentum (Attack/Decay) of the signal.
