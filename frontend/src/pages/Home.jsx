import { Link } from "react-router-dom";
import { api } from "../api.js";
import { percent } from "../format.js";
import { useRequest } from "../hooks/useRequest.js";
import Status from "../components/Status.jsx";

const steps = [
  {
    title: "Give an image",
    text: "Upload a chest X-ray, bean leaf or produce photo, or pick a held-out test image.",
  },
  {
    title: "Models predict",
    text: "Three ImageNet-pretrained CNNs, MobileNetV2, EfficientNet-B0 and ResNet-50, each classify the image.",
  },
  {
    title: "Prediction is explained",
    text: "A Grad-CAM heatmap shows which image regions drove each model, so the three can be compared side by side.",
  },
];

export default function Home() {
  const { data, error, loading } = useRequest(api.visionDomains, []);

  return (
    <div className="stack-lg">
      <section className="intro">
        <h1>Explainable AI for Agriculture and Healthcare</h1>
        <p className="lead">
          Transparent predictions for disease diagnosis, crop health assessment and food quality analysis. Every
          prediction comes with a heatmap of the image regions that drove it.
        </p>
      </section>

      <Status loading={loading} error={error}>
        <section className="grid-3">
          {data?.map((domain) => (
            <Link key={domain.id} to={`/vision/${domain.id}`} className="card card-link">
              <span className="eyebrow">{domain.sector}</span>
              <h2>{domain.name}</h2>
              <p className="muted">{domain.description}</p>
              {domain.models.length > 0 && (
                <span className="card-foot">
                  Best test accuracy {percent(Math.max(...domain.models.map((m) => m.accuracy)), 1)}
                </span>
              )}
            </Link>
          ))}
        </section>
      </Status>

      <section className="stack">
        <h2>How it works</h2>
        <ol className="steps">
          {steps.map((step) => (
            <li key={step.title} className="card">
              <h3>{step.title}</h3>
              <p className="muted">{step.text}</p>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
