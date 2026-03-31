"""Generate a realistic-style painting for the DynaSense-RAG project using Vertex AI Imagen."""
import os
import sys
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT:
    print("ERROR: GOOGLE_CLOUD_PROJECT not set")
    sys.exit(1)

vertexai.init(project=PROJECT, location="us-central1")

PROMPT = (
    "A photorealistic oil painting in the style of classic realism. "
    "A focused analyst in a crisp dark suit sits at a grand wooden desk inside a vast, dimly lit baroque library. "
    "Towering bookshelves stretch to the ceiling. "
    "Holographic glowing documents and golden neural network connection lines float around the analyst, "
    "gently illuminating their face in blue and gold light. "
    "The atmosphere blends 19th-century classical library grandeur with cutting-edge AI data retrieval technology. "
    "Dust motes float in shafts of light. Deep blue, warm gold, and mahogany tones. "
    "Masterful chiaroscuro lighting. Cinematic composition. "
    "The painting conveys precision, trust, and the power of knowledge."
)

OUTPUT_PATH = "docs/project_painting.png"

for model_name in ["imagen-3.0-generate-001", "imagegeneration@006"]:
    try:
        print(f"Trying model: {model_name} ...")
        model = ImageGenerationModel.from_pretrained(model_name)
        images = model.generate_images(
            prompt=PROMPT,
            number_of_images=1,
            aspect_ratio="16:9",
        )
        images[0].save(location=OUTPUT_PATH)
        print(f"SUCCESS: saved to {OUTPUT_PATH} using {model_name}")
        sys.exit(0)
    except Exception as e:
        print(f"  {model_name} failed: {e}")

print("ERROR: all models failed")
sys.exit(1)
