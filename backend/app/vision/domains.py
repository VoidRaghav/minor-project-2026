from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class VisionDomain:
    id: str
    name: str
    sector: str
    task: str
    description: str
    data_source: str
    classes: Tuple[str, ...]
    flip: bool  # whether horizontal flips are a valid augmentation


VISION_DOMAINS = {
    d.id: d
    for d in (
        VisionDomain(
            id="xray",
            name="Chest X-ray Diagnosis",
            sector="Healthcare",
            task="Pneumonia detection from paediatric chest X-ray images",
            description="Classifies a chest X-ray as normal or showing pneumonia, with a heatmap of the lung regions the model relied on.",
            data_source="Paediatric chest X-ray dataset (Kermany et al., 2018), 5,824 images re-split by patient",
            classes=("Normal", "Pneumonia"),
            flip=False,
        ),
        VisionDomain(
            id="leaf",
            name="Bean Leaf Health",
            sector="Agriculture",
            task="Crop disease detection from bean leaf photographs",
            description="Classifies a bean leaf photo as healthy, angular leaf spot or bean rust, with a heatmap of the lesions the model relied on.",
            data_source="iBean dataset, Makerere AI Lab, field photos from Uganda (1,295 images)",
            classes=("Angular Leaf Spot", "Bean Rust", "Healthy"),
            flip=True,
        ),
        VisionDomain(
            id="fruit",
            name="Produce Freshness",
            sector="Food",
            task="Food quality grading of fruit and vegetable photographs",
            description="Classifies a photo of fruit or vegetables as fresh or rotten, with a heatmap of the regions the model relied on.",
            data_source="Fresh and rotten fruits and vegetables dataset, 12 produce types (5,382 images after removing duplicates)",
            classes=("Fresh", "Rotten"),
            flip=True,
        ),
    )
}
