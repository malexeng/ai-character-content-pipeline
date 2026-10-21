# AI Character Content Pipeline

> 📘 New here? Start with **[READ_THIS_OVERVIEW.md](READ_THIS_OVERVIEW.md)** for a plain-English walkthrough.

Turns one image of an AI-generated character into a vetted, captioned batch of social posts — with quality and safety checks built in.

## Workflow

![Workflow diagram](docs/workflow.png)

> **Diagram key:** blue = input · purple = AI model · grey = processing · amber = storage · red = safety / guard rail · green = output

<details><summary>Diagram source (renders on GitHub)</summary>

```mermaid
flowchart TD
  IN([One seed image of an AI-generated character]):::input --> G[[Image model · style variations]]:::ai
  G --> Q{{"Quality check<br/>resolution · sharpness · brightness · contrast"}}:::guard
  Q -->|fail| X1[Discard]:::guard
  Q -->|pass| SF{{"Safety check · always on"}}:::guard
  SF -->|flagged| X2[Discard]:::guard
  SF -->|clean| V[[Vision model · describes and tags the image]]:::ai
  V --> C[[Text model · writes caption + hashtags]]:::ai
  C --> OUT([Upload queue · image + caption per post]):::output

  classDef input fill:#e8f1ff,stroke:#3b82f6,color:#0b2a5b,stroke-width:1.5px
  classDef ai fill:#f1e8ff,stroke:#8b5cf6,color:#2e1065,stroke-width:1.5px
  classDef step fill:#f4f4f5,stroke:#71717a,color:#18181b,stroke-width:1.2px
  classDef store fill:#fff6e0,stroke:#d97706,color:#5a3300,stroke-width:1.2px
  classDef output fill:#e7f8ee,stroke:#16a34a,color:#073b1c,stroke-width:1.5px
  classDef guard fill:#ffecec,stroke:#dc2626,color:#5c0a0a,stroke-width:1.2px
```
</details>

## What it does
- Generates several style variations of a seed character image.
- Drops anything blurry, too dark, too bright or low-contrast using measurable checks.
- Runs every image through a safety classifier that can't be turned off from the config.
- Describes each image with a local vision model, then writes platform-specific captions and hashtags.
- Exports a ready-to-review upload queue.

## What this project demonstrates
- Multi-model pipeline (image, vision, text)
- Automated quality gates
- Content-safety enforcement
- Local-first LLM use via Ollama

## Quick start
1. Install Ollama and pull `llava` and `qwen2.5:7b`.
2. `pip install -r requirements.txt`; set `STABILITY_API_KEY` in your environment for the variation step.
3. `python ai_character_pipeline.py`

## Project layout
```
ai_character_pipeline.py   ← all stages
```

## Notes
Built for fully AI-generated characters only.

## License
MIT — see [LICENSE](LICENSE).
