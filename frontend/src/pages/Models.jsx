import { Link } from "react-router-dom";
import { api } from "../api.js";
import { percent } from "../format.js";
import { useRequest } from "../hooks/useRequest.js";
import BarList from "../components/BarList.jsx";
import ConfusionMatrix from "../components/ConfusionMatrix.jsx";
import Status from "../components/Status.jsx";
import VisionComparison from "../components/VisionComparison.jsx";

const loadAll = async () => {
  const [domains, vision] = await Promise.all([api.domains(), api.visionDomains()]);
  return { tabular: await Promise.all(domains.map((d) => api.domain(d.id))), vision };
};

export default function Models() {
  const { data, error, loading } = useRequest(loadAll, []);

  return (
    <div className="stack-lg">
      <header className="page-head">
        <h1>Models</h1>
        <p className="muted">
          Image models are ImageNet-pretrained CNNs fine-tuned on each dataset and evaluated on its held-out test
          images. Tabular models are random forests evaluated on a stratified 25% held-out split.
        </p>
      </header>

      <Status loading={loading} error={error}>
        <h2>Image models</h2>
        <p className="muted small">
          Grad-CAM deletion: average drop in the predicted-class probability when the 20% most highlighted pixels are
          blanked, compared with blanking a random region of the same size. A larger gap means a more faithful heatmap.
        </p>
        {data?.vision.map((domain) => (
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

        <h2>Tabular models</h2>
        {data?.tabular.map((domain) => (
          <section key={domain.id} className="card stack">
            <div className="section-head">
              <div>
                <span className="eyebrow">{domain.sector}</span>
                <h2>{domain.name}</h2>
              </div>
              <Link to={`/analyze/${domain.id}`} className="button">
                Open
              </Link>
            </div>

            <dl className="metrics">
              <div>
                <dt>Accuracy</dt>
                <dd>{percent(domain.metrics.accuracy, 1)}</dd>
              </div>
              <div>
                <dt>Macro F1</dt>
                <dd>{domain.metrics.macro_f1.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Train / test</dt>
                <dd>
                  {domain.metrics.n_train} / {domain.metrics.n_test}
                </dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{domain.model}</dd>
              </div>
            </dl>

            <div className="models-grid">
              <div className="stack">
                <h3>Global feature importance</h3>
                <BarList
                  label={`${domain.name} feature importance`}
                  items={domain.importance.map((i) => ({ key: i.key, label: i.label, value: i.mean }))}
                  format={(v) => `${(v * 100).toFixed(1)} pp`}
                />
              </div>
              <div className="stack">
                <h3>Confusion matrix</h3>
                <ConfusionMatrix classes={domain.classes} matrix={domain.metrics.confusion_matrix} />
              </div>
            </div>

            <p className="muted small">Data: {domain.data_source}</p>
          </section>
        ))}
      </Status>
    </div>
  );
}
