# Changelog

All notable changes to this project are documented in this file.

## Unreleased

### Added
- Live transcript streaming in the web UI: each segment now scrolls into the **Transcript** box as Whisper produces it, instead of the text only appearing once the whole run finishes.
- Timestamped run logs written to `logs/` for both the web UI and `transcribe.py`, recording model load time, per-segment progress, and total run time.
- `--quiet` flag for `transcribe.py` to suppress per-segment console output while still logging timing/status.

### Changed
- Removed the web UI's fixed-percentage progress bar during transcription (it froze at 20% for the entire run and didn't reflect real progress); the scrolling transcript text now serves as the progress indicator.
- `transcribe.py` now prints per-segment output by default, matching the behavior of Whisper's own CLI.
