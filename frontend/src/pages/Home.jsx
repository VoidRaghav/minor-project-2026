import { Link } from "react-router-dom";
import { api } from "../api.js";
import { percent } from "../format.js";
import { useRequest } from "../hooks/useRequest.js";
import Status from "../components/Status.jsx";

const steps = [
  {
    title: "Give an input",
    text: "Upload a chest X-ray, leaf or produce photo, or adjust clinical, sensor and lab readings.",
  },
  {
    title: "Models predict",
    text: "Three pretrained CNNs (MobileNetV2, EfficientNet-B0, ResNet-50) classify images; random forests handle measurements.",
  },
  {
    title: "Prediction is explained",
    text: "Grad-CAM heatmaps show which image regions drove each CNN. Exact Shapley values split tabular predictions by input.",
  },
];

const loadHome = () => Promise.all([api.visionDomains(), api.domains()]);

export default function Home() {
  const { data, error, loading } = useRequest(loadHome, []);

  return (
    <div className="stack-lg">
      <section className="intro">
        <h1>Explainable AI for Agriculture and Healthcare</h1>
        <p className="lead">
          Transparent predictions for disease diagnosis, crop health assessment and food quality analysis. Every
          result shows which inputs drove it and by how much.
        </p>
      </section>

      <Status loading={loading} error={error}>
        <h2>Image analysis</h2>
        <section className="grid-3">
          {data?.[0].map((domain) => (
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
        <h2>Measurement analysis</h2>
        <section className="grid-3">
          {data?.[1].map((domain) => (
            <Link key={domain.id} to={`/analyze/${domain.id}`} className="card card-link">
              <span className="eyebrow">{domain.sector}</span>
              <h2>{domain.name}</h2>
              <p className="muted">{domain.description}</p>
              <span className="card-foot">Test accuracy {percent(domain.accuracy, 1)}</span>
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
