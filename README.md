# Roots (Open) — 모두의 고향

*Everyone has a place they come from. This is a sound map for all of them.*

**▶ Try it live:** https://huggingface.co/spaces/ChiaraKia/Roots-Open

---

**Roots (Open)** is the open, participatory companion to a personal work,
[Roots — 내 고향](https://github.com/KiaChiara/Roots). Where the original piece is built
around four places that matter to me, this version opens the same idea to anyone: pick your
own country or city, and watch a piece of music become a drawing traced over its shape.

Upload a MIDI file — each note becomes a flowing curve, positioned by pitch and coloured
along a blue-to-green gradient, accumulating as the music plays and clipped to the silhouette
of the place you choose.

## Choosing a place

Type in the **Shape / map** field to search among:

- **countries of the world** (main territory and nearby islands)
- **major Korean cities** (shown with their Latin and Korean names, e.g. *Busan (부산)*)
- the four **maps from the original work**: Seoul and Rome (hand-drawn), Italy and South Korea

The canvas adapts its proportions to each shape, so every place is shown correctly.

## Two ways to see it

- **View in browser** — a live, interactive visualization you can play, scrub, screenshot,
  and record.
- **Generate MP4** — an offline, full-quality render exported as a video file.

You can use the working audio synthesized from the MIDI, or upload your own audio track.

## Running locally

Requires Python 3 and `ffmpeg` (for MP4 export).

```bash
pip install -r requirements.txt
python3 app.py
```

Then open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

## How it works

1. **MIDI → score.** The file is parsed into notes (time, pitch, amplitude, duration);
   drum tracks are excluded.
2. **Audio.** Synthesized from the score, or supplied by the user.
3. **Visualization.** Each note is drawn as a curved line with distant anchor points and
   Perlin-noise variation. Curves accumulate at low opacity, with a brief brighter flash on
   each new note. A mask clips everything to the chosen shape.
4. **Maps.** Country and city silhouettes are generated from open geographic data
   (Natural Earth). For each country, the main territory and its nearby islands are kept,
   while distant overseas territories are dropped so they don't distort the frame.

## The original work

This is an extension. The personal work it grows from:

- 🗺️ Live demo: https://huggingface.co/spaces/ChiaraKia/Roots
- 💻 Code: https://github.com/KiaChiara/Roots

## Author

Created by **Chiara Giustiniani**.
