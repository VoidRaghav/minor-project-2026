import { Link } from "react-router-dom";
import { api } from "../api.js";
import { useRequest } from "../hooks/useRequest.js";
import ConfusionMatrix from "../components/ConfusionMatrix.jsx";
import Status from "../components/Status.jsx";
import VisionComparison from "../components/VisionComparison.jsx";

export default function Models() {
  const { data, error, loading } = useRequest(api.visionDomains, []);

  return (
    <div className="stack-lg">
      <header className="page-head">
        <h1>Models</h1>
        <p className="muted">
          ImageNet-pretrained CNNs fine-tuned on each dataset and evaluated on its held-out test images.
        </p>
      </header>

      <Status loading={loading} error={error}>
        <p className="muted small">
          Grad-CAM deletion: average drop in the predicted-class probability when the 20% most highlighted pixels are
          blanked, compared with blanking a random region of the same size. A larger gap means a more faithful heatmap.
        </p>
        {data?.map((domain) => (
          <section key={domain.id} className="card stack">
            <div className="section-head">
              <div>
                <span className="eyebrow">{domain.sector}</span>
                <h2>{domain.name}</h2>
              </div>
              <Link to={`/vision/${domain.id}`} className="button">
                Open
              </Link>
            </div>
            <VisionComparison domain={domain} />
            <div className="models-grid">
              {domain.models.map((m) => (
                <div key={m.arch} className="stack">
                  <h3>{m.name} confusion matrix</h3>
                  <ConfusionMatrix classes={domain.classes} matrix={m.confusion_matrix} />
                </div>
              ))}
            </div>
            <p className="muted small">Data: {domain.data_source}</p>
          </section>
        ))}

      </Status>
    </div>
  );
}
