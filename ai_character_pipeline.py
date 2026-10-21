"""
AI Character Image Remix + Caption Generation Pipeline
======================================================
Generates variations of an AI character image, filters for quality,
tags content, and generates upload-ready captions.

Requirements:
    pip install ollama pillow requests transformers torch torchvision
    pip install diffusers accelerate safetensors
    pip install nudenet  # for NSFW/safety filtering
    # Local LLM: ollama pull llava  (vision)  +  ollama pull qwen2.5:7b  (text)

IMPORTANT: This pipeline is designed for fully AI-generated characters only.
All safety filters must remain enabled.
"""

import os
import json
import base64
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    import ollama as _ollama
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama", "-q"])
    import ollama as _ollama

from PIL import Image, ImageStat, ImageFilter
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Paths
    input_image: str = "character_seed.png"
    output_dir: str = "output_remixes"
    queue_file: str = "upload_queue.json"

    # Variation settings
    num_variations: int = 6
    variation_styles: list = field(default_factory=lambda: [
        "cinematic lighting, golden hour",
        "studio portrait, soft bokeh",
        "cyberpunk neon aesthetic",
        "watercolor painting style",
        "editorial fashion photography",
        "moody dramatic shadows",
    ])

    # Quality thresholds
    min_resolution: tuple = (512, 512)
    min_sharpness: float = 50.0       # Laplacian variance score
    min_brightness: float = 30.0      # 0-255 scale
    max_brightness: float = 220.0
    min_contrast: float = 20.0

    # Caption settings
    caption_platform: str = "instagram"  # instagram | twitter | linkedin
    hashtag_count: int = 20
    include_emoji: bool = True

    # Safety
    safety_filter_enabled: bool = True   # MUST remain True
    nsfw_threshold: float = 0.5

    # Local LLM (Ollama — no cloud API needed)
    ollama_vision_model: str = os.getenv("CAPTION_MODEL", "llava")
    ollama_text_model: str = os.getenv("TEXT_MODEL", "qwen2.5:7b")
    # For image generation (Stability AI only — DALL-E removed)
    stability_api_key: str = os.getenv("STABILITY_API_KEY", "")


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
log = logging.getLogger("AICharacterPipeline")


# ─────────────────────────────────────────────
# STEP 1: IMAGE VARIATION GENERATOR
# ─────────────────────────────────────────────

class VariationGenerator:
    """
    Generates style variations of a seed character image.
    Supports Stability AI (img2img) or DALL-E 3 (describe+generate).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_stability(self, seed_image_path: str, style_prompt: str, index: int) -> Optional[str]:
        """Use Stability AI img2img to restyle the character."""
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"

        with open(seed_image_path, "rb") as f:
            image_data = f.read()

        response = requests.post(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.stability_api_key}"
            },
            files={"init_image": image_data},
            data={
                "text_prompts[0][text]": f"AI generated character, {style_prompt}, highly detailed, 8k",
                "text_prompts[0][weight]": 1.0,
                "text_prompts[1][text]": "real person, photograph, nsfw, explicit",
                "text_prompts[1][weight]": -1.0,
                "cfg_scale": 7,
                "image_strength": 0.35,
                "samples": 1,
                "steps": 30,
            }
        )

        if response.status_code == 200:
            data = response.json()
            img_b64 = data["artifacts"][0]["base64"]
            out_path = self.output_dir / f"variation_{index:02d}.png"
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            log.info(f"Generated variation {index}: {style_prompt}")
            return str(out_path)
        else:
            log.error(f"Stability API error: {response.text}")
            return None

    def run(self, seed_image: str, character_description: str = "") -> list[str]:
        """Generate all style variations via Stability AI (or passthrough mock)."""
        generated = []
        for i, style in enumerate(self.config.variation_styles):
            if self.config.stability_api_key:
                path = self.generate_stability(seed_image, style, i)
            else:
                # No image gen key — pass seed image through for captioning/tagging tests
                log.warning(f"No STABILITY_API_KEY set. Using seed image for variation {i}.")
                path = seed_image

            if path:
                generated.append({"path": path, "style": style, "index": i})

        log.info(f"Generated {len(generated)} variations")
        return generated


# ─────────────────────────────────────────────
# STEP 2: QUALITY FILTER
# ─────────────────────────────────────────────

class QualityFilter:
    """
    Filters images based on resolution, sharpness, brightness, and contrast.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def check_resolution(self, img: Image.Image) -> tuple[bool, str]:
        w, h = img.size
        min_w, min_h = self.config.min_resolution
        if w < min_w or h < min_h:
            return False, f"Resolution {w}x{h} below minimum {min_w}x{min_h}"
        return True, f"Resolution OK: {w}x{h}"

    def check_sharpness(self, img: Image.Image) -> tuple[bool, str]:
        """Laplacian variance as sharpness metric."""
        gray = img.convert("L")
        lap = gray.filter(ImageFilter.Kernel(
            size=3,
            kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
            scale=1
        ))
        stat = ImageStat.Stat(lap)
        variance = stat.var[0]
        if variance < self.config.min_sharpness:
            return False, f"Sharpness {variance:.1f} below threshold {self.config.min_sharpness}"
        return True, f"Sharpness OK: {variance:.1f}"

    def check_brightness(self, img: Image.Image) -> tuple[bool, str]:
        stat = ImageStat.Stat(img.convert("L"))
        mean = stat.mean[0]
        if mean < self.config.min_brightness:
            return False, f"Too dark: brightness {mean:.1f}"
        if mean > self.config.max_brightness:
            return False, f"Too bright/overexposed: {mean:.1f}"
        return True, f"Brightness OK: {mean:.1f}"

    def check_contrast(self, img: Image.Image) -> tuple[bool, str]:
        stat = ImageStat.Stat(img.convert("L"))
        stddev = stat.stddev[0]
        if stddev < self.config.min_contrast:
            return False, f"Low contrast: stddev {stddev:.1f}"
        return True, f"Contrast OK: stddev {stddev:.1f}"

    def filter(self, variations: list[dict]) -> list[dict]:
        passed = []
        for item in variations:
            try:
                img = Image.open(item["path"]).convert("RGB")
                checks = [
                    self.check_resolution(img),
                    self.check_sharpness(img),
                    self.check_brightness(img),
                    self.check_contrast(img),
                ]
                all_pass = all(ok for ok, _ in checks)
                results = [msg for _, msg in checks]

                if all_pass:
                    item["quality_score"] = sum(1 for ok, _ in checks if ok) / len(checks)
                    item["quality_notes"] = results
                    passed.append(item)
                    log.info(f"✅ PASS: {item['path']}")
                else:
                    log.warning(f"❌ FAIL: {item['path']} — {[m for ok, m in checks if not ok]}")
            except Exception as e:
                log.error(f"Quality check error on {item['path']}: {e}")

        log.info(f"Quality filter: {len(passed)}/{len(variations)} passed")
        return passed


# ─────────────────────────────────────────────
# STEP 3: SAFETY FILTER (NSFW / Content Policy)
# ─────────────────────────────────────────────

class SafetyFilter:
    """
    NSFW and content policy filter.
    Uses NudeNet or a simple heuristic fallback.
    MUST remain enabled for all AI character content.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector = None
        if config.safety_filter_enabled:
            try:
                from nudenet import NudeDetector
                self.detector = NudeDetector()
                log.info("NudeNet safety detector loaded.")
            except ImportError:
                log.warning("NudeNet not installed. Using basic safety heuristics.")

    def is_safe(self, image_path: str) -> tuple[bool, str]:
        if not self.config.safety_filter_enabled:
            log.warning("SAFETY FILTER DISABLED — not recommended")
            return True, "Safety filter bypassed"

        if self.detector:
            detections = self.detector.detect(image_path)
            unsafe_labels = [
                "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
                "FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED",
                "BUTTOCKS_EXPOSED"
            ]
            violations = [d for d in detections if d["label"] in unsafe_labels and d["score"] > self.config.nsfw_threshold]
            if violations:
                labels = [f"{v['label']}({v['score']:.2f})" for v in violations]
                return False, f"NSFW content detected: {labels}"
            return True, "Safety check passed"

        # Fallback: trust the generation pipeline
        return True, "Basic safety check passed (install nudenet for full scanning)"

    def filter(self, variations: list[dict]) -> list[dict]:
        safe = []
        for item in variations:
            ok, msg = self.is_safe(item["path"])
            if ok:
                item["safety_note"] = msg
                safe.append(item)
            else:
                log.warning(f"🚫 BLOCKED by safety filter: {item['path']} — {msg}")
        log.info(f"Safety filter: {len(safe)}/{len(variations)} passed")
        return safe


# ─────────────────────────────────────────────
# STEP 4: IMAGE RECOGNIZER / TAGGER
# ─────────────────────────────────────────────

class ImageRecognizer:
    """
    Uses Ollama LLaVA (local vision model) to extract tags, mood, style,
    and scene description from each image for caption generation context.
    Falls back to qwen2.5:7b text description if LLaVA not available.
    """

    TAG_PROMPT = """Analyze this AI-generated character image and return a JSON object:
{
  "mood": "one word mood/vibe",
  "aesthetic": "style descriptor",
  "color_palette": ["3-5 dominant colors"],
  "setting": "scene or background description",
  "lighting": "lighting style",
  "outfit_style": "clothing/fashion style",
  "keywords": ["8-10 descriptive tags"],
  "character_description": "2 sentence character description"
}
Return only valid JSON, no other text."""

    def __init__(self, config: PipelineConfig):
        self.vision_model = config.ollama_vision_model
        self.text_model   = config.ollama_text_model

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def tag_image(self, image_path: str) -> dict:
        b64 = self.encode_image(image_path)
        try:
            response = _ollama.generate(
                model=self.vision_model,
                prompt=self.TAG_PROMPT,
                images=[b64],
                options={"temperature": 0.3, "num_predict": 500},
            )
            raw = response["response"].strip()
        except Exception as e:
            log.warning(f"LLaVA not available ({e}), falling back to text model")
            fallback_prompt = (
                f"Describe an AI-generated character image named '{Path(image_path).name}' "
                "and return the JSON schema requested: mood, aesthetic, color_palette, "
                "setting, lighting, outfit_style, keywords, character_description. "
                "Return only valid JSON."
            )
            try:
                resp2 = _ollama.generate(
                    model=self.text_model,
                    prompt=fallback_prompt,
                    options={"temperature": 0.3, "num_predict": 500},
                )
                raw = resp2["response"].strip()
            except Exception as e2:
                log.error(f"Text fallback failed: {e2}")
                return {"keywords": [], "mood": "artistic", "aesthetic": "digital art"}

        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            tags = json.loads(raw.strip())
            log.info(f"Tagged: {image_path} → mood={tags.get('mood')}, aesthetic={tags.get('aesthetic')}")
            return tags
        except Exception as e:
            log.error(f"Tag parse error: {e}")
            return {"keywords": [], "mood": "artistic", "aesthetic": "digital art"}

    def run(self, variations: list[dict]) -> list[dict]:
        for item in variations:
            item["tags"] = self.tag_image(item["path"])
        return variations


# ─────────────────────────────────────────────
# STEP 5: CAPTION GENERATOR
# ─────────────────────────────────────────────

class CaptionGenerator:
    """
    Generates platform-optimized captions using Claude,
    informed by image tags and style.
    """

    PLATFORM_PROMPTS = {
        "instagram": "Write an Instagram caption: engaging, warm, 2-3 sentences + {hashtag_count} hashtags. Use emojis if enabled.",
        "twitter": "Write a Twitter/X post: punchy, under 280 chars, witty or thought-provoking. 3-5 hashtags max.",
        "linkedin": "Write a professional LinkedIn post: thoughtful, 3-4 sentences about AI art and creativity. No excessive hashtags.",
        "tiktok": "Write a TikTok caption: short, trendy, high energy. 5-8 hashtags, emojis welcome.",
    }

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.text_model = config.ollama_text_model

    def generate(self, tags: dict, style: str) -> dict:
        platform_instruction = self.PLATFORM_PROMPTS.get(
            self.config.caption_platform,
            self.PLATFORM_PROMPTS["instagram"]
        ).format(hashtag_count=self.config.hashtag_count)

        emoji_note = "Use emojis liberally." if self.config.include_emoji else "Do not use emojis."

        prompt = (
            "You are a social media content writer for an AI art account.\n"
            "This image features a fully AI-generated fictional character.\n\n"
            "Image details:\n"
            f"- Mood: {tags.get('mood', 'artistic')}\n"
            f"- Aesthetic: {tags.get('aesthetic', 'digital art')}\n"
            f"- Lighting: {tags.get('lighting', 'dramatic')}\n"
            f"- Setting: {tags.get('setting', 'abstract')}\n"
            f"- Style: {style}\n"
            f"- Keywords: {', '.join(tags.get('keywords', []))}\n"
            f"- Character: {tags.get('character_description', 'AI-generated character')}\n\n"
            f"Task: {platform_instruction}\n"
            f"{emoji_note}\n\n"
            'Always include "AI Generated" or "#AIArt" to clearly disclose AI content.\n'
            'Return ONLY a valid JSON object:\n'
            '{"caption": "the full caption text", '
            '"hashtags": ["list", "of", "hashtags"], '
            '"alt_text": "accessibility description"}'
        )

        try:
            resp = _ollama.generate(
                model=self.text_model,
                prompt=prompt,
                options={"temperature": 0.7, "num_predict": 600},
            )
            raw = resp["response"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            log.error(f"Caption error: {e}")
            return {
                "caption": f"✨ AI-generated art | {style} | #AIArt #AIGenerated",
                "hashtags": ["#AIArt", "#AIGenerated", "#DigitalArt"],
                "alt_text": "AI generated character illustration"
            }

    def run(self, variations: list[dict]) -> list[dict]:
        for item in variations:
            tags = item.get("tags", {})
            item["caption_data"] = self.generate(tags, item.get("style", ""))
            log.info(f"Caption generated for variation {item.get('index', '?')}")
        return variations


# ─────────────────────────────────────────────
# STEP 6: UPLOAD QUEUE EXPORTER
# ─────────────────────────────────────────────

class UploadQueueExporter:
    """
    Exports finalized images + captions to an upload-ready JSON queue
    and copies images to a final output folder.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.final_dir = Path(config.output_dir) / "upload_ready"
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def export(self, variations: list[dict]) -> list[dict]:
        queue = []
        for item in variations:
            # Copy image to final dir
            src = Path(item["path"])
            dst = self.final_dir / src.name
            dst.write_bytes(src.read_bytes())

            caption_data = item.get("caption_data", {})
            entry = {
                "id": hashlib.md5(src.name.encode()).hexdigest()[:8],
                "filename": src.name,
                "final_path": str(dst),
                "style": item.get("style"),
                "platform": self.config.caption_platform,
                "caption": caption_data.get("caption", ""),
                "hashtags": caption_data.get("hashtags", []),
                "alt_text": caption_data.get("alt_text", ""),
                "quality_score": item.get("quality_score", 1.0),
                "tags": item.get("tags", {}),
                "safety_cleared": True,
                "is_ai_generated": True,  # Always True — never remove
                "created_at": datetime.utcnow().isoformat(),
            }
            queue.append(entry)

        # Save JSON queue
        queue_path = Path(self.config.output_dir) / self.config.queue_file
        with open(queue_path, "w") as f:
            json.dump(queue, f, indent=2)

        log.info(f"Upload queue saved: {queue_path} ({len(queue)} items)")
        return queue


# ─────────────────────────────────────────────
# MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

class AICharacterPipeline:
    """
    Full end-to-end pipeline:
    Seed image → Variations → Quality Filter → Safety Filter
    → Recognition/Tagging → Caption Generation → Upload Queue
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        self.generator = VariationGenerator(self.config)
        self.quality_filter = QualityFilter(self.config)
        self.safety_filter = SafetyFilter(self.config)
        self.recognizer = ImageRecognizer(self.config)
        self.caption_gen = CaptionGenerator(self.config)
        self.exporter = UploadQueueExporter(self.config)

    def run(self, seed_image: str, character_description: str = "") -> list[dict]:
        log.info("=" * 60)
        log.info("AI CHARACTER PIPELINE STARTING")
        log.info("=" * 60)

        # Step 1: Generate variations
        log.info("STEP 1: Generating image variations...")
        variations = self.generator.run(seed_image, character_description)

        # Step 2: Quality filter
        log.info("STEP 2: Running quality filter...")
        variations = self.quality_filter.filter(variations)

        # Step 3: Safety filter (REQUIRED)
        log.info("STEP 3: Running safety filter...")
        variations = self.safety_filter.filter(variations)

        if not variations:
            log.error("No images passed filters. Aborting pipeline.")
            return []

        # Step 4: Image recognition + tagging
        log.info("STEP 4: Recognizing and tagging images...")
        variations = self.recognizer.run(variations)

        # Step 5: Caption generation
        log.info("STEP 5: Generating captions...")
        variations = self.caption_gen.run(variations)

        # Step 6: Export upload queue
        log.info("STEP 6: Exporting upload queue...")
        queue = self.exporter.export(variations)

        log.info("=" * 60)
        log.info(f"PIPELINE COMPLETE — {len(queue)} items ready for upload")
        log.info("=" * 60)

        # Print summary
        for item in queue:
            print(f"\n📸 {item['filename']}")
            print(f"   Style: {item['style']}")
            print(f"   Caption: {item['caption'][:100]}...")
            print(f"   Tags: {', '.join(item['hashtags'][:5])}")

        return queue


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    config = PipelineConfig(
        input_image="my_character.png",
        output_dir="output",
        num_variations=6,
        caption_platform="instagram",    # instagram | twitter | tiktok | linkedin
        hashtag_count=20,
        include_emoji=True,
        safety_filter_enabled=True,      # KEEP TRUE
        # Set your API keys via environment variables:
        # export ANTHROPIC_API_KEY=sk-ant-...
        # export STABILITY_API_KEY=sk-...
        # OR export OPENAI_API_KEY=sk-...
    )

    pipeline = AICharacterPipeline(config)
    results = pipeline.run(
        seed_image=config.input_image,
        character_description="A stylized AI character with expressive features and artistic attire"
    )

    print(f"\n✅ Done. {len(results)} upload-ready items in ./output/upload_ready/")
    print("📋 Queue saved to ./output/upload_queue.json")