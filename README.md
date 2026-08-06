# Q1 – Garment and Person Attribute Extraction using Florence-2

## Objective

The objective of Q1 is to extract visual attributes from a person image and a garment image using the Florence-2 Vision-Language Model. The extracted information is converted into a structured JSON format for use in later stages of the Smart Try-On System.

---

## Model

* **Model:** Microsoft Florence-2 Base
* **Framework:** Hugging Face Transformers
* **Device:** CUDA (GPU) if available, otherwise CPU

---

## Dataset

The dataset contains:

* Person images
* Garment images
* `pairs_manifest.csv` mapping person images to garment images

Project structure:

```
Data/
├── test_pairs/
│   ├── person/
│   ├── garment/
│   └── pairs_manifest.csv
└── Output/
    └── Q1/
```

---

## Workflow

### 1. Load Dataset

The image pairs are read from `pairs_manifest.csv`.

### 2. Load Images

The corresponding person and garment images are loaded using the image utility functions.

### 3. Generate Image Captions

Florence-2 is used with the `<DETAILED_CAPTION>` task to generate descriptive captions for both images.

Example garment caption:

```
The image shows a purple and black Vans T-shirt with the iconic Vans logo on the front.
```

Example person caption:

```
The image shows a model wearing a white tank top and blue jeans, standing against a cream-colored background.
```

### 4. Extract Attributes

The generated captions are processed using rule-based text extraction to identify relevant attributes.

Garment attributes:

* Garment Type
* Sleeve Length
* Neckline
* Primary Color
* Secondary Color
* Pattern

Person attributes:

* Pose
* Upper Body Visible
* Lower Body Visible

### 5. Generate JSON

The extracted attributes are combined into a structured JSON object.

Example:

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

### 6. Batch Processing

After validating the pipeline on a single image pair, all image pairs listed in `pairs_manifest.csv` are processed automatically.

A JSON file is generated for each processed pair.

---

## Output

Output files are stored in:

```
Data/Output/Q1/
```

Example:

```
person_01.json
person_02.json
person_03.json
person_04.json
person_05.json
```

---

## Utilities Used

* `config.py` – Project configuration and paths
* `model_loader.py` – Loads the Florence-2 model
* `image_utils.py` – Image loading and visualization
* `file_utils.py` – CSV and JSON operations

---

## Notes

* Florence-2 is used with its native caption generation capability (`<DETAILED_CAPTION>`).
* Structured garment and person attributes are extracted from the generated captions using a lightweight rule-based parser.
* The generated JSON files serve as structured metadata for subsequent stages of the Smart Try-On pipeline.

---

## Pipeline

```
pairs_manifest.csv
        │
        ▼
 Load Image Pair
        │
        ▼
 Florence-2 (<DETAILED_CAPTION>)
        │
        ▼
 Caption Generation
        │
        ▼
 Attribute Extraction
        │
        ▼
 JSON Generation
        │
        ▼
 Save Results
```
## Challenges and Troubleshooting

During the implementation of Q1, several issues were encountered while developing the image understanding pipeline. The following summarizes the problems, their causes, and the adopted solutions.

### 1. Model Selection

Initially, MiniCPM-V 2.6 was considered because of its strong instruction-following capability and ability to generate structured responses directly from natural language prompts. However, to align with the project implementation, Microsoft Florence-2 Base was selected as the primary Vision-Language Model.

### 2. Florence-2 Prompt Behavior

The first implementation attempted to use prompts such as:

```text
Analyze this garment image.

Return only:

Garment Type:
Sleeve Length:
Neckline:
Primary Color:
Pattern:
```

Expected output:

```text
Garment Type: Shirt
Sleeve Length: Long
Neckline: Collar
Primary Color: Blue
Pattern: Plain
```

Observed output:

```text
Analyze this garment image.

Return only:

Garment Type:
Sleeve Length:
Neckline:
Primary Color:
Pattern:
```

Instead of generating the requested attributes, Florence-2 reproduced the prompt template without providing values.

**Reason**

Florence-2 is primarily designed around predefined vision tasks such as captioning, OCR, grounding, and object detection rather than arbitrary conversational instruction following.

**Solution**

The implementation was redesigned to use Florence-2's `<DETAILED_CAPTION>` task. The generated captions were then converted into structured attributes using a lightweight rule-based extraction module.

---

### 3. Caption Parsing

The first parser expected responses in key-value format.

Example:

```text
Garment Type: Shirt
Sleeve Length: Long
```

Since Florence-2 produced descriptive captions instead of structured text, the parser returned empty values.

Example:

```python
{
    "garment_type": "",
    "sleeve_length": "",
    "neckline": "",
    "primary_color": ""
}
```

**Solution**

A new parser was developed to identify garment-related information directly from the generated captions.

---

### 4. Attribute Extraction Accuracy

Some generated captions contained additional scene information.

Example:

```text
The image shows a purple and black Vans T-shirt with the iconic Vans logo on the front. The background is a crisp white.
```

Initial extraction incorrectly considered **white** as a garment color because it appeared in the caption.

**Solution**

Background-related phrases were removed before attribute extraction, and color detection was limited to garment descriptions whenever possible.

---

### 5. Incomplete Sleeve and Neckline Information

Certain captions did not explicitly describe sleeve length or neckline.

Example:

```text
The image shows a purple and black Vans T-shirt.
```

Since no sleeve or neckline information was present, these attributes could not be inferred reliably.

**Solution**

When the required information was absent from the caption, the implementation stored `"Unknown"` instead of making unsupported assumptions.

---

### 6. Dataset Manifest Issues

While batch processing the dataset, execution stopped with a `TypeError`.

```
TypeError: unsupported operand type(s) for /: 'PosixPath' and 'float'
```

Investigation showed that some rows in `pairs_manifest.csv` contained missing (`NaN`) values instead of valid image paths.

**Solution**

Only rows containing valid person and garment image paths were processed during batch execution. This prevented runtime failures while preserving valid image pairs.

---

### 7. Path Construction Errors

During initial testing, image loading failed because duplicate directory names were included while constructing file paths.

Example:

```text
.../person/person/person_01.png
```

instead of

```text
.../person/person_01.png
```

**Solution**

The path construction logic was updated to match the directory structure defined in the project configuration.

---

### 8. Rule-Based Attribute Extraction

Florence-2 provides descriptive captions but does not directly generate structured metadata.

To bridge this gap, a lightweight rule-based extraction module was implemented to identify:

* Garment type
* Sleeve length
* Neckline
* Primary color
* Secondary color
* Pattern
* Person pose
* Upper body visibility
* Lower body visibility

This approach allows Florence-2 outputs to be converted into the JSON format required for the remaining stages of the project.

---

## Limitations

* Attribute extraction depends on the quality and completeness of Florence-2 captions.
* Some visual properties may not be mentioned explicitly in the generated description.
* Rule-based extraction is less flexible than using a conversational Vision-Language Model capable of directly producing structured outputs.
* Complex garment designs or uncommon fashion terminology may not always be identified correctly.
