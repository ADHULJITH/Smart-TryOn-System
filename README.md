# Smart Try-On System

An end-to-end computer-vision pipeline developed for the Xapien Innovatus (XIPL) SDE Intern Technical Assessment.

The project covers the implemented portions of the assessment:

- **Q1:** Garment and person attribute understanding using Florence-2
- **Q2:** Human parsing, agnostic-person generation, and garment segmentation
- **Q3:** End-to-end virtual try-on using CatVTON
- **Q4:** Automated quality evaluation — **attempted, but not completed because of model/dependency compatibility and limited Colab GPU resources**

The implementation was developed primarily in **Google Colab on an NVIDIA Tesla T4**. The notebooks and generated outputs are retained so that the work can be inspected and reproduced.

> **Important:** Q4 is intentionally documented as incomplete. No fabricated Q4 scores or evaluation CSV are claimed as completed.

---

## 1. Assessment Coverage

| Question | Task | Status |
|---|---|---|
| Q1 | Garment & body understanding with a VLM | ✅ Implemented |
| Q2 | Human parsing & garment segmentation | ✅ Implemented |
| Q3 | End-to-end try-on inference | ✅ Implemented |
| Q4 | Automated quality evaluation | ⚠️ Attempted / incomplete |
| Q5 | Mini web demo | ❌ Not implemented |

The implementation prioritised getting Q1–Q3 working end-to-end within the available free-tier GPU environment and documenting the Q4 compatibility failure rather than submitting unsupported results.

---

# 2. Project Structure

The repository is organised around the actual working project rather than committing large model checkpoints.

```text
Smart-TryOn-System/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── Collab/
│   ├── Q1_*.ipynb
│   ├── Q2_*.ipynb
│   ├── Q3_*.ipynb
│   └── Q4_*.ipynb
│
├── Data/
│   ├── test_pairs/
│   │   ├── person/
│   │   ├── garment/
│   │   └── pairs_manifest.csv
│   │
│   ├── edge_cases/
│   │   └── ...
│   │
│   └── Output/
│       ├── Q1/
│       ├── Q2/
│       │   ├── parsing_maps/
│       │   ├── agnostic/
│       │   ├── garment_masks/
│       │   └── visualizations/
│       │
│       └── Q3/
│           ├── tryon_results/
│           ├── inputs/
│           ├── catvton_masks/
│           ├── visualizations/
│           └── q3_constraints_log.json
│
└── Utils/
    ├── config.py
    ├── file_utils.py
    ├── image_utils.py
    └── model_loader.py
```

The exact notebook names may differ depending on the final Colab export.

### Model checkpoints

Large model weights are **not committed to GitHub**. They are downloaded from their official repositories during notebook execution.

---

# 3. Environment

The main development environment was:

```text
Platform      : Google Colab
GPU           : NVIDIA Tesla T4
Python        : 3.12.x
PyTorch       : 2.x CUDA build
CUDA          : Colab-provided CUDA runtime
```

The most important practical limitation throughout the project was **GPU VRAM**.

Because CatVTON is a diffusion-based model, the implementation uses conservative settings and processes one pair at a time.

---

# 4. Installation

Clone the repository:

```bash
git clone https://github.com/ADHULJITH/Smart-TryOn-System.git
cd Smart-TryOn-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For Google Colab:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Set the project root according to the location of the repository:

```python
from pathlib import Path

PROJECT_ROOT = Path("/content/Smart-TryOn-System")
```

If the repository is stored in Google Drive instead:

```python
PROJECT_ROOT = Path(
    "/content/drive/MyDrive/Smart-TryOn-System"
)
```

The notebooks define the project-specific paths from this root.

---

# 5. Q1 — Garment & Body Understanding

## Objective

Q1 requires a vision-language model to understand person and garment images and produce structured information.

The required attributes are:

- Garment type
- Sleeve length
- Neckline
- Primary color
- Pattern
- Person pose
- Upper-body visibility
- Lower-body visibility

## Model

**Microsoft Florence-2 Base**

```text
microsoft/Florence-2-base
```

License: **MIT**

Florence-2 was selected because it is an open-source VLM suitable for free-tier experimentation and was already successfully used for the Q1 image-understanding pipeline.

## Pipeline

```text
Person / Garment Image
          │
          ▼
     Florence-2
          │
          ▼
 <DETAILED_CAPTION>
          │
          ▼
 Caption text
          │
          ▼
 Rule-based attribute extraction
          │
          ▼
 Structured JSON
```

Florence-2 did not reliably behave like a general chat model when given arbitrary structured-output prompts. Therefore, the implementation uses Florence-2's native captioning task and converts the resulting description into structured attributes using a lightweight parser.

## Example output

```json
{
    "garment_type": "T-Shirt",
    "sleeve_length": "Unknown",
    "neckline": "Unknown",
    "primary_color": "Purple",
    "secondary_color": "Black",
    "pattern": "Logo",
    "pose": "Front-facing",
    "upper_body_visible": true,
    "lower_body_visible": true
}
```

When an attribute cannot be reliably inferred, the pipeline uses `"Unknown"` rather than inventing a value.

## Q1 outputs

Generated JSON files are stored under:

```text
Data/Output/Q1/
```

The Q1 Colab notebook is stored under:

```text
Collab/
```

---

# 6. Q2 — Human Parsing & Garment Segmentation

## Objective

Q2 prepares the person and garment images for virtual try-on.

It produces:

1. A human parsing map
2. An agnostic person representation with upper-body clothing removed/masked
3. A clean garment segmentation/mask

## Models

### SCHP — Self-Correction for Human Parsing

Repository:

https://github.com/GoGoDuck912/Self-Correction-Human-Parsing

License: **MIT**

SCHP provides semantic human parsing classes including:

- Background
- Hair
- Face
- Upper-clothes
- Skirt
- Pants
- Dress
- Arms
- Shoes
- Bag
- Scarf

### rembg

Repository:

https://github.com/danielgatis/rembg

License: **MIT**

`rembg` is used for garment background removal. Its U²-Net-based model isolates the garment from the product-image background.

## Person pipeline

```text
Person Image
     │
     ▼
    SCHP
     │
     ▼
Human Parsing Map
     │
     ├───────────────┐
     │               │
     ▼               ▼
Body regions     Clothing regions
                     │
                     ▼
             Upper-body clothing
                     │
                     ▼
              Clothing mask
                     │
                     ▼
          Agnostic representation
```

## Garment pipeline

```text
Garment Product Image
          │
          ▼
        rembg
          │
          ▼
     Alpha mask
          │
          ▼
    Binary mask
          │
          ▼
  Isolated garment
```

## Edge cases

The assessment specifically includes difficult cases:

### Hair over shoulders

`person_02` and `person_03` contain hair overlapping the shoulder region.

The parsing stage attempts to keep hair separate from upper-body clothing.

### Strappy garment

`garment_03` contains thin straps.

Thin structures are difficult for background-removal models, so the garment mask was visually checked.

### Crossed arms

The provided crossed-arm edge case is processed separately.

The agnostic-image generation attempts to preserve detected arm regions while removing upper-body clothing.

Required outputs include:

```text
Data/Output/Q2/parsing_maps/person_crossed_arms.png
Data/Output/Q2/agnostic/person_crossed_arms_agnostic.png
```

## Q2 outputs

```text
Data/Output/Q2/
├── parsing_maps/
├── agnostic/
├── garment_masks/
└── visualizations/
```

## Important Q2 compatibility issue

The original SCHP implementation depends on an older `inplace_abn` C++/CUDA extension.

In the current Colab environment, this produced compatibility problems involving the legacy extension and modern Python/PyTorch/CUDA versions.

The issue was documented rather than hidden because the assessment explicitly allows partial credit for well-documented failures.

---

# 7. Q3 — End-to-End Virtual Try-On

## Objective

Q3 connects the Q2 preprocessing stage to an actual virtual try-on model.

The selected model is **CatVTON**.

## Why CatVTON?

CatVTON was selected because it was the most practical of the allowed try-on models for a free Google Colab T4 environment.

Model:

```text
zhengchong/CatVTON
```

Base model:

```text
stable-diffusion-v1-5/stable-diffusion-inpainting
```

Official repositories:

- https://github.com/Zheng-Chong/CatVTON
- https://huggingface.co/zhengchong/CatVTON
- https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting

## Q3 configuration

```text
Resolution       : 384 × 512
Batch size       : 1
Precision        : FP16
Diffusion steps  : 30
Guidance scale   : 2.5
Seed             : 555
Device           : CUDA / Tesla T4
```

The resolution and number of diffusion steps were intentionally reduced to keep inference practical under limited VRAM.

## Q2 → Q3 integration

```text
Q2
Human Parsing
     │
     ▼
Agnostic Representation
     │
     ▼
CatVTON Mask
     │
     ├──────────────────┐
     │                  │
     ▼                  ▼
Person Image       Garment Image
     │                  │
     └────────┬─────────┘
              ▼
          CatVTON
              │
              ▼
       Try-On Result
```

The Q2 preprocessing is therefore part of the Q3 pipeline rather than a separate experiment.

## Test pairs

Five pairs were used:

| Pair | Person | Garment | Notes |
|---|---|---|---|
| pair_01 | person_01.png | garment_01.jpg | Baseline |
| pair_02 | person_02.png | garment_02.jpg | Hair over shoulder |
| pair_03 | person_03.png | garment_03.jpg | Hair + strappy garment |
| pair_04 | person_04.png | garment_04.jpg | Candidate-sourced pair |
| pair_05 | person_05.png | garment_05.jpg | Candidate-sourced pair |

The manifest was completed so all five pairs could be processed.

## Output

```text
Data/Output/Q3/tryon_results/
├── pair_01_tryon.png
├── pair_02_tryon.png
├── pair_03_tryon.png
├── pair_04_tryon.png
└── pair_05_tryon.png
```

Additional artifacts:

```text
Data/Output/Q3/
├── catvton_masks/
├── visualizations/
└── q3_constraints_log.json
```

---

# 8. GPU and Memory Constraints

GPU memory was the main engineering constraint throughout Q2/Q3 development.

Several runtime failures occurred because Colab's available VRAM is limited compared with dedicated inference hardware.

The following workarounds were used.

## FP16

CatVTON was loaded using:

```python
torch.float16
```

This reduces memory consumption compared with FP32.

## Batch size 1

The five pairs were processed sequentially:

```text
pair_01 → GPU → result
pair_02 → GPU → result
pair_03 → GPU → result
pair_04 → GPU → result
pair_05 → GPU → result
```

rather than loading all five together.

## Reduced resolution

Inference was performed at:

```text
384 × 512
```

This was a deliberate trade-off between visual detail and VRAM usage.

## Reduced diffusion steps

The pipeline uses:

```text
30 steps
```

instead of a larger default configuration.

## CUDA cleanup

Between processing stages:

```python
import gc
import torch

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

Peak GPU memory was also monitored during Q3 inference.

## Other issues encountered

### Q3 agnostic-image path mismatch

The initial Q3 code searched for an agnostic image using an exact filename that did not match the generated Q2 filename.

This produced:

```text
FileNotFoundError:
No agnostic image found for person_01.png
```

The path-resolution logic was corrected to use the actual Q2 output naming.

### CatVTON mask path mismatch

The initial visualization code expected CatVTON masks in a different directory from the one actually used by the project.

The code was updated so the Q3 stages consistently use:

```text
Data/Output/Q3/catvton_masks/
```

These failures were resolved before final Q3 processing.

---

# 9. Q4 — Automated Quality Evaluation

## Status

**Q4 was attempted but is not complete.**

The assessment requires three evaluation axes:

1. Garment fidelity
2. Identity preservation
3. VLM-as-judge

The planned implementation used lightweight methods suitable for a T4.

### A. Garment fidelity

Planned method:

```text
OpenCLIP ViT-B/32
        │
        ▼
Garment embedding
        +
Generated garment-region embedding
        │
        ▼
Cosine similarity
```

### B. Identity preservation

Planned method:

```text
Face detection
      │
      ▼
Original face crop
      +
Try-on face crop
      │
      ▼
SSIM
```

This was chosen instead of InsightFace to avoid loading another relatively heavy face-embedding model.

### C. VLM-as-judge

The planned judge uses the same Q1 model family:

```text
Florence-2
```

with a rubric covering:

- Fit realism
- Texture transfer
- Garment appearance
- Artifacts
- Overall realism

The intended score is:

```text
1–10
```

with a short explanation.

## Why Q4 was not completed

The main blocker was **Florence-2 compatibility in the current Colab environment**, combined with limited GPU memory.

During Q4 setup, multiple compatibility errors occurred while loading Florence-2 through the installed Transformers stack, including:

```text
AttributeError:
'Florence2LanguageConfig' object has no attribute
'forced_bos_token_id'
```

and:

```text
AttributeError:
RobertaTokenizer has no attribute
additional_special_tokens
```

and:

```text
AttributeError:
'Florence2ForConditionalGeneration' object has no attribute
'_supports_sdpa'
```

These errors prevented the Q4 VLM judge from being reliably loaded.

Because the evaluation CSV must contain genuine model-generated results, unsupported or manually invented Q4 scores were **not** added to the repository.

This is an intentional documented limitation.

## Planned Q4 output

If the compatibility issue is resolved, the expected structure is:

```text
Data/Output/Q4/
├── evaluation_template_q4.csv
├── q4_evaluation_log.json
├── garment_fidelity/
├── identity/
└── vlm_judge/
```

The final CSV should contain one row for each of the five pairs.

---

# 10. Q4 Judge Rubric

The intended VLM judge rubric is:

```text
You are evaluating a virtual try-on result.

You are given:
1. The original person image.
2. The input garment image.
3. The generated virtual try-on image.

Evaluate only the quality of the generated try-on result.

Score the result from 1 to 10.

9-10:
Highly realistic try-on. The garment fits naturally, follows
body geometry, transfers texture and appearance accurately,
and contains almost no visible artifacts.

7-8:
Good try-on. The garment is recognizable and mostly realistic,
with minor issues in fit, texture, boundaries, folds, or artifacts.

5-6:
Moderate result. The garment is recognizable but there are
noticeable problems with fit, texture transfer, alignment,
occlusion, or visual artifacts.

3-4:
Poor result. Major distortions, incorrect garment placement,
weak texture transfer, unrealistic fit, or obvious artifacts.

1-2:
Very poor result. The garment is missing, severely distorted,
incorrectly placed, or the generated image is clearly broken.

Evaluate:

A. Fit realism
B. Texture and appearance transfer
C. Generation artifacts
D. Overall photographic realism

Return JSON:

{
    "score": 1,
    "reasons": "brief explanation"
}
```

This rubric is documented for transparency even though the Q4 evaluation could not be completed.

---

# 11. Model and Tool Licenses

| Component | Use | License |
|---|---|---|
| Florence-2 Base | Q1 VLM | MIT |
| SCHP | Q2 human parsing | MIT |
| rembg | Q2 garment background removal | MIT |
| CatVTON | Q3 virtual try-on | CC BY-NC-SA 4.0 |
| Stable Diffusion v1.5 Inpainting | CatVTON base | CreativeML OpenRAIL-M |

Third-party licenses remain applicable to their respective models and repositories.

CatVTON is used here strictly for the non-commercial technical assessment/research context. Model licenses should be reviewed before any other use.

---

# 12. Reproducibility

The recommended execution order is:

```text
Q1
 ↓
Q2
 ↓
Q3
 ↓
Q4
```

### Q1

Run the Q1 Colab notebook and verify JSON outputs under:

```text
Data/Output/Q1/
```

### Q2

Run Q2 preprocessing and verify:

```text
Data/Output/Q2/
```

especially the crossed-arm outputs.

### Q3

Verify all five pairs and Q2 outputs first, then:

1. Load CatVTON.
2. Test `pair_01`.
3. Process the remaining four pairs.
4. Verify five try-on images.
5. Check the constraints log.

### Q4

The Q4 notebook contains the attempted evaluation pipeline, but the final automated evaluation is currently blocked by Florence-2/Transformers compatibility in the Colab environment.

---

# 13. Known Limitations

The project intentionally documents the following limitations:

- Florence-2 caption-based attribute extraction can miss attributes that are not explicitly mentioned.
- Rule-based extraction is less flexible than a conversational VLM.
- SCHP's original legacy implementation has compatibility issues with modern Colab environments.
- Human parsing can be difficult around hair, thin straps, occlusions, and crossed arms.
- CatVTON output quality can vary with pose, clothing boundaries, and mask quality.
- Lower-resolution inference can reduce fine garment details.
- Diffusion models may produce visual artifacts.
- Q4 automated scoring is incomplete because of Florence-2 loading/compatibility issues.
- Q5 was not implemented within the available time.

---

# 14. Current Deliverables

At the time of submission preparation:

```text
Q1  ✅ Working implementation + outputs
Q2  ✅ Working preprocessing pipeline + outputs
Q3  ✅ Working CatVTON inference + 5 results
Q4  ⚠️ Evaluation pipeline attempted; blocked by dependency/model compatibility
Q5  ❌ Not implemented
```

The repository therefore represents the actual state of the work rather than claiming completion of components that were not successfully executed.

---

# 15. Colab Notebooks

The Colab notebooks used during development are stored under:

```text
Collab/
```

Before submitting, add the shareable Google Colab links here:

```text
Q1: [ADD COLAB LINK]
Q2: [ADD COLAB LINK]
Q3: [ADD COLAB LINK]
Q4: [ADD COLAB LINK]
```

Set the notebook sharing permission to the level required by the company so the evaluators can open them.

---

# 16. Submission Checklist

Before pushing the repository:

- [ ] `README.md` committed
- [ ] `requirements.txt` committed
- [ ] `Collab/` notebooks committed
- [ ] Q1 JSON outputs committed
- [ ] Q2 parsing maps committed
- [ ] Q2 agnostic images committed
- [ ] Q2 garment masks committed
- [ ] Crossed-arm Q2 outputs committed
- [ ] Q3 five try-on images committed
- [ ] Q3 constraints log committed
- [ ] CatVTON masks/visualizations committed where required
- [ ] Large model checkpoints excluded from Git
- [ ] Q4 incomplete status clearly documented
- [ ] No fabricated Q4 scores
- [ ] Colab links added
- [ ] Repository access checked

---

# 17. References

### Florence-2

https://huggingface.co/microsoft/Florence-2-base

### SCHP

https://github.com/GoGoDuck912/Self-Correction-Human-Parsing

### rembg

https://github.com/danielgatis/rembg

### CatVTON

https://github.com/Zheng-Chong/CatVTON

https://huggingface.co/zhengchong/CatVTON

### Stable Diffusion v1.5 Inpainting

https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting

### CatVTON Paper

https://arxiv.org/abs/2407.15886

---

# 18. Final Note

This repository demonstrates the implemented portion of the assessment as an end-to-end pipeline:

```text
Person + Garment
       │
       ▼
      Q1
Visual Understanding
       │
       ▼
      Q2
Human Parsing
+ Agnostic Representation
+ Garment Segmentation
       │
       ▼
      Q3
CatVTON Virtual Try-On
       │
       ▼
Five Try-On Results
       │
       ▼
      Q4
Automated Evaluation
(attempted; compatibility blocked)
```

The main engineering challenge was running multiple computer-vision and diffusion components within the memory and dependency constraints of a free Google Colab T4 environment. The repository documents those constraints and the resulting trade-offs explicitly.
