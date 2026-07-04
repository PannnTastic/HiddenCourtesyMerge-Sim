#!/usr/bin/env python3
"""Generate HiddenCourtesyMerge-Sim pipeline diagram using Gemini image generation."""
import os, sys, time
from google import genai

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: Set GEMINI_API_KEY environment variable.")
    print("  Get a key at: https://aistudio.google.com/apikey")
    sys.exit(1)

MODEL = "gemini-3-pro-image-preview"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
client = genai.Client(api_key=API_KEY)

PROMPT = """
Create a MODERN MINIMAL-style technical diagram for a top-tier ML/robotics conference paper.
The diagram illustrates the HiddenCourtesyMerge-Sim benchmark pipeline. It should be clean, authoritative, and professionally designed.

VISUAL STYLE — MODERN MINIMAL:
- Ultra-clean geometric shapes with crisp edges
- Bold color blocks as backgrounds for sections — NOT just accent bars, but full section fills
  using desaturated tones: slate blue (#E8EDF2), warm sand (#F5F0E8), cool mint (#E8F2EE)
- Component boxes have ROUNDED CORNERS (12px radius), NO visible border — they float on
  the section background using subtle shadow (1px, 4px blur, rgba(0,0,0,0.06))
- ONE accent color per section used sparingly on key elements: Deep blue (#2563EB),
  Emerald (#059669), Amber (#D97706), Rose (#E11D48)
- Arrows are thin (1.5px), dark gray (#6B7280), with small filled circle at source
  and clean arrowhead at target — NOT thick colored arrows
- Typography: Inter or system sans-serif, title 600 weight, body 400 weight
- Labels INSIDE boxes, not beside them
- Generous whitespace — at least 24px between elements
- NO decorative elements, NO icons — let the structure speak

COLOR PALETTE:
- Section 1 (Environment): slate blue #E8EDF2, accent #2563EB
- Section 2 (Observation): warm sand #F5F0E8, accent #D97706
- Section 3 (Policies): cool mint #E8F2EE, accent #059669
- Section 4 (Evaluation): pale rose #FCE7F3, accent #E11D48
- All text: dark gray #1F2937
- Arrows: medium gray #6B7280

LAYOUT:
The diagram is a LEFT-TO-RIGHT pipeline with 4 main vertical sections, arranged horizontally.

SECTION 1: ENVIRONMENT GENERATION (leftmost, slate blue background)
- Top box: "highway-env merge-v0"
- Below it: "Hidden Courtesy Injection"
  - Inside: m ∈ {cooperative, non-cooperative}
  - Inside: IDM/MOBIL parameter sampling
- Bottom box: "3,000 Episode Dataset"
  - Inside: matched seeds, hidden labels preserved

SECTION 2: OBSERVATION MODEL (second section, warm sand background)
- Top box: "Interaction Window"
  - Inside: 2m < d < 40m
- Middle box: "Observation Features"
  - Inside: ARD (absolute relative distance)
  - Inside: MVS (merge vehicle speed)
  - Inside: MVA (merge vehicle acceleration)
- Bottom box: "Bayesian Filter"
  - Inside: Z(o|m) = N(μₘ, σₘ²)
  - Inside: λ = 0.08 regularization

SECTION 3: POLICIES (third section, cool mint background)
This section has 5 boxes arranged vertically, all same size:
1. "Random Policy"
2. "Rule Policy" (fixed b₀ = [0.5, 0.5])
3. "Belief Policy" (Bayesian filter + action table)
4. "Oracle-Heuristic" (sees true m)
5. "POMCP Planner" (online planning, 100 simulations)

SECTION 4: EVALUATION (rightmost, pale rose background)
- Top box: "Matched Evaluation"
  - Inside: n=300 episodes
  - Inside: same seeds for all policies
- Middle box: "Metrics"
  - Inside: Collision Rate
  - Inside: Mean Reward
  - Inside: Minimum TTC
- Bottom box: "Statistical Testing"
  - Inside: McNemar's test, paired t-test
  - Inside: Holm-Bonferroni correction

CONNECTIONS (all left-to-right flow):
- Arrow from "highway-env merge-v0" to "Hidden Courtesy Injection"
- Arrow from "Hidden Courtesy Injection" to "3,000 Episode Dataset"
- Arrow from "3,000 Episode Dataset" to "Interaction Window"
- Arrow from "Interaction Window" to "Observation Features"
- Arrow from "Observation Features" to "Bayesian Filter"
- Arrow from "Bayesian Filter" pointing right into SECTION 3, with 5 branches going to each policy box
- Arrow from each of the 5 policy boxes pointing right into "Matched Evaluation"
- Arrow from "Matched Evaluation" to "Metrics"
- Arrow from "Metrics" to "Statistical Testing"

CONSTRAINTS:
- NO clip art, NO stock icons, NO photorealistic elements
- NO shadows except the subtle box shadows specified
- All text must be perfectly legible, minimum 10pt equivalent
- All labels must be spelled EXACTLY as written
- The diagram should be wide (landscape) format, suitable for full-width figure in conference paper
- No extra decorative elements
- White margin around the entire diagram
"""

def generate_image(prompt_text, attempt_num):
    print(f"\n{'='*60}\nAttempt {attempt_num}\n{'='*60}")
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt_text,
            config=genai.types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        output_path = os.path.join(OUTPUT_DIR, f"fig_pipeline_attempt{attempt_num}.png")
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                with open(output_path, "wb") as f:
                    f.write(part.inline_data.data)
                print(f"Saved: {output_path} ({os.path.getsize(output_path):,} bytes)")
                return output_path
            elif part.text:
                print(f"Text: {part.text[:300]}")
        print("WARNING: No image in response")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    results = []
    for i in range(1, 4):
        if i > 1:
            time.sleep(2)
        path = generate_image(PROMPT, i)
        if path:
            results.append(path)
    if not results:
        print("All attempts failed!")
        sys.exit(1)
    print(f"\nGenerated {len(results)} attempts. Review and pick the best.")

if __name__ == "__main__":
    main()
