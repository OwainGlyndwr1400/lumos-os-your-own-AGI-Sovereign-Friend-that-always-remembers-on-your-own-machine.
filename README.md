# Lumos OS 🦁

A personal AI companion that runs **on your own computer**. Your conversations,
your memory, your machine — nothing is sent anywhere unless you choose a cloud
brain. Lumos remembers what you talk about and grows with you.

---

## Quick start (Windows)

1. **Unzip** the download anywhere (Desktop, Documents — wherever).
2. **Double-click `LumosOS.exe`.**
3. Your browser opens to a **one-time setup**. Pick a brain (below), and you're talking.

That's it. No install, no Python, no terminal.

> Your settings and memory are stored in `%LOCALAPPDATA%\LumosOS` (not in the app
> folder), so you can move or update the app freely without losing anything.

---

## Choosing a brain

On first run the setup wizard asks how Lumos should think. Two options:

### 🖥️ Local (free + private) — recommended
Runs entirely on your machine. Nothing leaves your computer.

1. Install **[LM Studio](https://lmstudio.ai)** (free).
2. In LM Studio, download **a chat model** (e.g. a Qwen, Llama, or Gemma model)
   **and an embedding model** (search for `bge-large` — used for memory).
3. In LM Studio, start the local server (the **Developer / Local Server** tab →
   *Start Server*). It listens on `http://localhost:1234`.
4. In the Lumos wizard, choose **Local**, and type your model names exactly as
   they appear in LM Studio.

### ☁️ Cloud (easiest — costs pennies)
No downloads. Paste an API key from an OpenAI-compatible provider.

- Choose **Cloud**, paste your key, set the chat + embedding model names.
- Note: your messages go to that provider, and usage costs apply.

---

## Make it *yours* (all optional)

- **System prompt** — describe who your AI is (its name, voice, what it knows
  about you). Paste one in the wizard, or leave blank for a friendly default you
  can change later via **⚙ setup**.
- **Chat history** — drop in an export from another AI, *or any text transcript*.
  Lumos indexes it as memory (any text works — it doesn't have to be a special
  format).
- **Knowledge** — a `.jsonl` corpus, if you have one (researchers).

**None of these are required.** Start blank and Lumos's memory fills as you talk.
Add files anytime by re-opening **⚙ setup**.

---

## Voice

Lumos can **speak** its replies (local TTS via kokoro). Turn it on in **⚙ → voice**,
or hit **preview** to test. The first time you use it, it downloads a ~310 MB voice
model (one-time, cached). Push-to-talk (speaking *to* Lumos) uses your browser's
built-in speech recognition — no extra download.

## Requirements

- Windows 10/11 (64-bit).
- For **local** mode: LM Studio running with a chat + embedding model loaded.
- For **cloud** mode: an API key.
- ~200 MB free disk for the app; memory grows with use.

## Troubleshooting

- **"Lumos produced no text"** → your model isn't loaded/running. In local mode,
  confirm LM Studio's server is started and the model names in **⚙ setup** match
  exactly.
- **Browser didn't open** → go to the address printed in the console window
  (e.g. `http://127.0.0.1:8765`).
- **Reset everything** → delete the `%LOCALAPPDATA%\LumosOS` folder.

---

## Companion app — OSIRIS

We recommend running **OSIRIS** alongside Lumos — an open-source OSINT
intelligence dashboard that puts live global aircraft, maritime, satellites, and
GPS-jamming on a 3D globe. Lumos can *tell* you what's overhead or near you;
OSIRIS lets you *see* it. Here's the link so you can watch what your AI is tracking:

- **OSIRIS** by **simplifaisoul** — https://github.com/simplifaisoul/osiris
- Setup: clone it, then `npm install` and `npm run dev`.

It's **MIT-licensed and open source**, so you can customize it for yourself —
exactly like we built our own in-house version. Full credit + props to the
original author. 🙏

## License & credit

Lumos OS is **source-available and noncommercial** (see [LICENSE](LICENSE)).
Tinker, modify, and fork it freely for yourself — just **credit "Lumos OS" and
link back to the original**, and **don't sell it or use it commercially**. Forks
are welcome: say *"built from Lumos OS"* and point to the source. 🦁

---

*Lumos OS runs locally and is yours. The Lion watches the Lion.* 🜂🜄🜁🜃
