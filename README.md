# Explainable AI for Agriculture and Healthcare

Most AI models in medicine and farming give an answer without saying why. This project builds interpretable prediction models for three tasks, from **images** (three pretrained CNNs compared, explained with Grad-CAM) and from **measurements** (random forests, explained with exact Shapley values):

- **Disease diagnosis**: is a breast tumour sample malignant or benign?
- **Crop health**: is a field healthy, short of nutrients, water stressed or at risk of disease?
- **Food quality**: is a raw milk sample high, medium or low grade?

Every prediction comes with an exact breakdown of how much each input measurement raised or lowered the result.

![Disease diagnosis with Shapley explanation](docs/screenshots/disease.png)

## What it does

| | |
| --- | --- |
| **Predict** | A random forest per domain returns a class and the probability of every class. |
| **Explain locally** | Exact Shapley values split the predicted probability into one signed contribution per input, measured in percentage points. |
| **Explain globally** | Permutation importance ranks which inputs the model depends on across the whole test set. |
| **Summarise** | The top supporting and opposing factors are written out as short sentences, built directly from the numbers. |
| **What-if** | Moving any slider re-runs the prediction and explanation in about 20 ms. |
| **Validate** | You can load a held-out test sample and compare the prediction with the actual label. |

## How it works

```mermaid
flowchart LR
    U[User adjusts inputs] --> F[React UI]
    F -- POST /api/domains/id/analyze --> V[Input validation<br/>range and schema]
    V --> M[Random forest<br/>predict_proba]
    V --> S[Exact Shapley explainer<br/>2^n coalitions x 40 references]
    M --> R[Prediction and class probabilities]
    S --> R2[Per-feature contributions]
    R --> O[Response]
    R2 --> O
    O --> F
```

### Exact Shapley values

For an input `x` with `n` features, the explainer looks at every subset `S` of features (2ⁿ subsets). For each subset it builds 40 hybrid rows: features in `S` take their values from `x`, and the rest come from 40 reference samples drawn from the training data. The value of the subset, `v(S)`, is the model's average probability over those 40 rows. Each feature's contribution is its weighted average marginal effect over all subsets:

```
φᵢ = Σ  |S|! (n − |S| − 1)! / n!  ·  [ v(S ∪ {i}) − v(S) ]
    S ⊆ N \ {i}
```

This is calculated exactly, not estimated by sampling, and it works with any model. With 8 features that is 256 subsets × 40 references = 10,240 rows, all scored in one vectorised `predict_proba` call. Contributions always add up exactly:

```
reference rate  +  Σ φᵢ  =  predicted probability
```

The test suite checks this for every domain, and the benchmarks below measure the rounding error.

### Training pipeline

1. Load the dataset for the domain.
2. Split 75/25 into train and test, with the same class proportions in both (`random_state=42`).
3. Train `RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced")`.
4. Evaluate on the test split: accuracy, macro F1 and confusion matrix.
5. Compute permutation importance on the test split (10 repeats).
6. Save the model, reference samples, feature bounds and metrics to `backend/artifacts/<domain>.joblib`.

The server trains any missing model on first start (a few seconds). After that it loads the saved files.

## Image models

The same three ImageNet-pretrained CNNs are fine-tuned on one image dataset per domain and compared: **MobileNetV2**, **EfficientNet-B0** and **ResNet-50**. Every prediction is explained with a **Grad-CAM** heatmap. In the web app you upload an image (or pick a held-out test image) and see all three models side by side.

| Domain | Task | Dataset | Classes | Train / val / test |
| --- | --- | --- | --- | --- |
| Healthcare | Pneumonia detection | Paediatric chest X-rays (Kermany et al., 2018), 5,824 images | Normal, Pneumonia | 4,078 / 883 / 863 |
| Agriculture | Bean leaf disease | iBean, Makerere AI Lab, 1,295 field photos | Angular leaf spot, Bean rust, Healthy | 1,034 / 133 / 128 |
| Food | Fresh vs rotten produce | Fruits and vegetables, 12 types, 5,382 images | Fresh, Rotten | 1,500* / 823 / 835 |

Data handling (`python -m app.vision.prepare`):

- **X-rays are split by patient.** The public release shares patients between its train and test folders (278 test patients also appear in training), which would let a model recognise a patient instead of the disease. All images are pooled and re-split 70/15/15 by patient, stratified by class.
- **Duplicates are removed** from the produce dataset (194 found by comparing small thumbnails) before it is split 70/15/15 per source class.
- All images are resized to 224 x 224.

Training (`python -m app.vision.train`): new output layer on ImageNet weights, AdamW (learning rate 2e-4 for pretrained layers, 1e-3 for the new layer, one-cycle schedule), 8 epochs, batch size 32, random crop, rotation and brightness/contrast (horizontal flip for leaf and produce only), class-weighted cross-entropy with label smoothing 0.05, seed 42. The epoch with the best validation macro F1 is kept and the test set is used once. To save memory the produce models were trained with `--train-limit 1500` (a class-balanced random subset of the 3,724 training images); validation and test sets are unchanged. Runs on the Apple GPU (MPS), CUDA or CPU.

### Image benchmarks

Held-out test images, measured on an Apple M4. Time is the median for one image (batch of 1). Bold = best macro F1 per task.

| Task | Model | Test accuracy | Macro F1 | Recall per class | Size | Time / image | Grad-CAM deletion drop (highlighted vs random) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chest X-ray | **MobileNetV2** | **97.6%** | **0.969** | 95.3 / 98.4 | 4.6 MB | 4.7 ms | 21.1 vs 12.3 pp |
|  | EfficientNet-B0 | 96.4% | 0.956 | 98.3 / 95.7 | 8.2 MB | 6.8 ms | 12.9 vs 5.8 pp |
|  | ResNet-50 | 97.0% | 0.961 | 90.3 / 99.5 | 47.2 MB | 12.6 ms | 21.9 vs 12.6 pp |
| Bean leaf | MobileNetV2 | 96.1% | 0.961 | 93.0 / 97.7 / 97.6 | 4.6 MB | 5.4 ms | 26.9 vs 5.0 pp |
|  | EfficientNet-B0 | 95.3% | 0.953 | 88.4 / 97.7 / 100.0 | 8.2 MB | 15.0 ms | 47.2 vs 31.6 pp |
|  | **ResNet-50** | **96.9%** | **0.969** | 95.3 / 95.3 / 100.0 | 47.2 MB | 26.0 ms | 26.0 vs 8.0 pp |
| Produce freshness | MobileNetV2 | 94.0% | 0.940 | 93.3 / 94.7 | 4.6 MB | 4.9 ms | 27.1 vs 3.3 pp |
|  | EfficientNet-B0 | 94.5% | 0.945 | 93.5 / 95.5 | 8.2 MB | 6.4 ms | 28.8 vs 4.8 pp |
|  | **ResNet-50** | **96.2%** | **0.962** | 96.4 / 95.9 | 47.2 MB | 11.1 ms | 33.2 vs 9.1 pp |

Grad-CAM deletion test: the 20% of pixels with the highest heat are blanked and the drop in the predicted-class probability is compared with blanking a random blob of the same size (200 test images per model). A faithful heatmap should hurt the prediction more than a random region.

The leaf test set has only 128 images, so one image is 0.8 points and small differences between models are not meaningful. Training times are not reported because they were measured on a shared laptop.

| Chest X-ray: three models and their Grad-CAM heatmaps | Bean leaf |
| --- | --- |
| ![X-ray analysis](docs/screenshots/image-xray.png) | ![Leaf analysis](docs/screenshots/image-leaf.png) |

![Model comparison](docs/screenshots/image-models.png)

## Measurement datasets

| Domain | Source | Samples | Features | Classes |
| --- | --- | --- | --- | --- |
| Disease Diagnosis | [Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (UCI, bundled with scikit-learn) | 569 | mean radius, texture, smoothness, compactness, concavity, concave points, symmetry, worst area | Malignant (212), Benign (357) |
| Crop Health | Synthetic, generated from agronomic threshold rules (`backend/app/ml/domains/crop.py`) | 3,000 | soil moisture, N, P, K, soil pH, air temperature, relative humidity, leaf chlorophyll (SPAD) | Healthy, Nutrient Deficiency, Water Stress, Disease Risk |
| Food Quality | Synthetic, generated from dairy spoilage and grading rules (`backend/app/ml/domains/food.py`) | 3,000 | pH, storage temperature, time since milking, fat %, bacterial load (log CFU/ml), turbidity, odour score | High, Medium, Low |

The synthetic generators are fixed-seed and encode known relationships. In the milk data, for example, bacterial load grows with storage time × temperature, which then lowers pH and odour score. Label noise is added so the task is not trivial. These datasets are for demonstrating the explanation pipeline. They are not agronomic or food-safety ground truth.

## Benchmarks

All numbers are reproducible with `python -m app.ml.benchmark`. They were measured on an Apple M4 with Python 3.9 and scikit-learn 1.6.1.

### Model performance

| Domain | Model | Test accuracy | Test macro F1 | 5-fold CV accuracy |
| --- | --- | --- | --- | --- |
| Disease Diagnosis | Majority class | 62.9% | 0.386 | 62.7% ± 0.4% |
| Disease Diagnosis | Logistic regression | 92.3% | 0.918 | 95.1% ± 1.6% |
| Disease Diagnosis | **Random forest (deployed)** | **97.2%** | **0.970** | **95.6% ± 1.5%** |
| Crop Health | Majority class | 47.9% | 0.162 | 47.9% ± 0.1% |
| Crop Health | Logistic regression | 87.7% | 0.871 | 87.3% ± 1.6% |
| Crop Health | **Random forest (deployed)** | **85.3%** | **0.837** | **84.3% ± 1.3%** |
| Food Quality | Majority class | 38.0% | 0.184 | 38.0% ± 0.0% |
| Food Quality | Logistic regression | 90.7% | 0.908 | 92.2% ± 0.6% |
| Food Quality | **Random forest (deployed)** | **89.2%** | **0.894** | **90.6% ± 0.6%** |

On the real clinical data, the random forest beats logistic regression by about 5 points of test accuracy. On the two synthetic domains, logistic regression is 1.5–2.5 points ahead, because those labels come from rules that are close to linear. The random forest is used everywhere so all three domains share one model family and one explanation path. Swapping in another model only needs a `predict_proba` function.

Disease confusion matrix on the 143 test samples: 51/53 malignant and 88/90 benign classified correctly.

### Explainer

| Domain | Features | Coalitions | Median latency | p95 latency | Max additivity error |
| --- | --- | --- | --- | --- | --- |
| Disease Diagnosis | 8 | 256 | 15 ms | 18 ms | 1.1e-15 |
| Crop Health | 8 | 256 | 35 ms | 38 ms | 1.0e-15 |
| Food Quality | 7 | 128 | 18 ms | 20 ms | 1.0e-15 |

### Most important features (permutation importance, drop in test accuracy)

| Domain | 1st | 2nd | 3rd |
| --- | --- | --- | --- |
| Disease Diagnosis | Worst area (9.7 pp) | Mean texture (2.9 pp) | Mean concave points (2.8 pp) |
| Crop Health | Relative humidity (24.5 pp) | Soil moisture (21.4 pp) | Air temperature (7.8 pp) |
| Food Quality | Bacterial load (33.6 pp) | pH (7.3 pp) | Odour score (3.3 pp) |

## Screenshots

| Overview | Crop health |
| --- | --- |
| ![Overview](docs/screenshots/overview.png) | ![Crop health](docs/screenshots/crop.png) |

| Model evaluation |
| --- |
| ![Models](docs/screenshots/models.png) |

## Tech stack

| Layer | Tools |
| --- | --- |
| Image models | PyTorch 2.8, torchvision (MobileNetV2, EfficientNet-B0, ResNet-50), Pillow |
| ML | scikit-learn 1.6 (random forest, permutation importance), NumPy |
| Explainability | Grad-CAM (`backend/app/vision/gradcam.py`) and a custom exact Shapley explainer (`backend/app/ml/explainer.py`), no SHAP dependency |
| API | FastAPI, Pydantic v2, Uvicorn |
| Frontend | React 18, React Router 6, Vite 6, plain CSS with light and dark themes |
| Testing | pytest, FastAPI TestClient |

## Project structure

```
backend/
  app/
    main.py               FastAPI app, CORS, model loading at startup
    api/
      router.py           /api prefix
      deps.py             domain lookup and store dependencies
      routes/             health and domain endpoints
    ml/
      spec.py             Feature and DomainSpec definitions
      domains/            disease, crop, food: data loaders and metadata
      trainer.py          split, fit, evaluate, permutation importance
      explainer.py        exact Shapley values
      store.py            train-or-load cache in artifacts/
      train.py            retrain all models
      benchmark.py        baselines, cross-validation, explainer timing
    services/analysis.py  validation, prediction, contribution ranking, summary
    vision/
      prepare.py          download, de-duplicate, patient-level split, resize
      models.py           MobileNetV2, EfficientNet-B0, ResNet-50 with new heads
      train.py            fine-tune, benchmark, deletion test
      gradcam.py          Grad-CAM and heatmap overlay
      service.py          load models, predict, explain
      domains.py          the three image tasks
    schemas.py            request and response models
  tests/test_api.py
frontend/
  src/
    router.jsx            route table
    pages/                Home, Analyze, Models, NotFound
    components/           Layout, FeatureField, ContributionChart, BarList, Status
    api.js                API client
docs/screenshots/
```

## Setup

Requirements: Python 3.9+ and Node.js 18+.

**Backend**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The first start trains the three random forest models into `backend/artifacts/`.

For the image models (needs about 1 GB of disk for the datasets and a few GB of memory):

```bash
python -m app.vision.prepare   # download and prepare images into data/vision/
python -m app.vision.train     # fine-tune and benchmark the 9 image models
```

**Frontend** (in a second terminal)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server forwards `/api` requests to port 8000.

**Other commands**

```bash
cd backend
python -m pytest              # 20 tests: additivity, endpoints, validation, image models, Grad-CAM
python -m app.ml.train        # retrain and overwrite saved models
python -m app.ml.benchmark    # regenerate the benchmark tables
```

## Routes

| Page | Path |
| --- | --- |
| Overview | `/` |
| Chest X-ray, bean leaf, produce images | `/vision/xray`, `/vision/leaf`, `/vision/fruit` |
| Disease diagnosis | `/analyze/disease` |
| Crop health | `/analyze/crop` |
| Food quality | `/analyze/food` |
| Model evaluation | `/models` |

| Method | Endpoint | Returns |
| --- | --- | --- |
| GET | `/api/health` | Status and loaded models |
| GET | `/api/domains` | Domain list with test accuracy |
| GET | `/api/domains/{id}` | Features with bounds and defaults, classes, metrics, confusion matrix, importance |
| GET | `/api/domains/{id}/sample` | A random held-out test sample and its true label |
| POST | `/api/domains/{id}/analyze` | Prediction, probabilities, reference rate, contributions, summary |
| GET | `/api/vision/domains` | Image tasks with the test metrics of every model |
| GET | `/api/vision/domains/{id}` | Task details and held-out test images |
| POST | `/api/vision/domains/{id}/predict` | Multipart image upload; returns each model's prediction, probabilities and Grad-CAM heatmap |

Interactive API docs are at http://localhost:8000/docs.

Example:

```bash
curl -X POST localhost:8000/api/domains/crop/analyze \
  -H "Content-Type: application/json" \
  -d '{"values":{"soil_moisture":12,"nitrogen":80,"phosphorus":40,"potassium":160,
       "soil_ph":6.6,"temperature":36,"humidity":35,"chlorophyll":38}}'
```

```json
{
  "prediction": "Water Stress",
  "confidence": 0.96,
  "base_value": 0.09,
  "contributions": [
    { "label": "Soil moisture", "value": 12.0, "contribution": 0.459 },
    { "label": "Air temperature", "value": 36.0, "contribution": 0.165 },
    { "label": "Relative humidity", "value": 35.0, "contribution": 0.163 }
  ],
  "summary": [
    "Predicted Water Stress with 96% probability, against a reference rate of 9% for Water Stress across the background data.",
    "Strongest evidence for Water Stress: Soil moisture = 12.0 % (+45.9 pp), Air temperature = 36.0 °C (+16.5 pp), Relative humidity = 35.0 % (+16.3 pp)."
  ]
}
```

The response above is shortened. Inputs outside the training range, or with missing or unknown features, return `422` with a message naming the problem field.

## Adding a domain

1. Create `backend/app/ml/domains/<name>.py` that defines its `Feature`s, class names, a `load(seed)` function returning `(X, y)`, and a `SPEC`.
2. Register it in `backend/app/ml/domains/__init__.py`.
3. Add a nav link in `frontend/src/components/Layout.jsx`.

The form, charts, evaluation page and API work automatically from the spec.

## Limitations

- Crop and food results come from synthetic data and show that the method works, not how it performs in the field.
- Exact Shapley cost grows as 2ⁿ. Beyond about 15 features, a sampling-based or tree-specific estimator is needed.
- Contributions are measured against the reference set. When features are strongly correlated, hybrid rows can combine values that never occur together.
- Image models are trained on one public dataset per task; results on images from other hospitals, farms or cameras are untested.
- Grad-CAM heatmaps are coarse (a 7 x 7 grid) and show where the model looked, not exact boundaries.
- This is a research prototype and is not validated for clinical, agricultural or food-safety decisions.

## Roadmap

- Replace the synthetic data with field datasets (soil and crop sensor logs, milk quality lab data)
- Test the image models on external datasets and try sharper explanations (Grad-CAM++)
- Counterfactuals: the smallest change to the inputs that flips the prediction
- Exportable per-sample explanation reports
