import { useEffect, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { percent } from "../format.js";
import { useRequest } from "../hooks/useRequest.js";
import BarList from "../components/BarList.jsx";
import Status from "../components/Status.jsx";
import VisionComparison from "../components/VisionComparison.jsx";

export default function Vision() {
  const { domainId } = useParams();
  const [searchParams] = useSearchParams();
  const { data: domain, error, loading } = useRequest(() => api.visionDomain(domainId), [domainId]);
  const [selected, setSelected] = useState(null); // { url, label, name }
  const [analysis, setAnalysis] = useState({ result: null, error: null, pending: false });
  const inputRef = useRef(null);
  const requestId = useRef(0);

  useEffect(() => {
    setSelected(null);
    setAnalysis({ result: null, error: null, pending: false });
  }, [domainId]);

  const run = async (blob, meta) => {
    const id = ++requestId.current;
    setSelected((previous) => {
      if (previous?.url) URL.revokeObjectURL(previous.url);
      return { ...meta, url: URL.createObjectURL(blob) };
    });
    setAnalysis((current) => ({ ...current, error: null, pending: true }));
    try {
      const result = await api.visionPredict(domainId, blob, meta.name);
      if (id === requestId.current) setAnalysis({ result, error: null, pending: false });
    } catch (err) {
      if (id === requestId.current) setAnalysis({ result: null, error: err, pending: false });
    }
  };

  const pickSample = async (sample) => {
    try {
      const blob = await api.visionSample(domainId, sample.name);
      run(blob, { name: `${sample.name}.jpg`, label: sample.label });
    } catch (err) {
      setAnalysis({ result: null, error: err, pending: false });
    }
  };

  useEffect(() => {
    const name = searchParams.get("sample");
    const sample = domain?.samples.find((s) => s.name === name);
    if (sample) pickSample(sample);
  }, [domain, searchParams]);

  const upload = (event) => {
    const file = event.target.files?.[0];
    if (file) run(file, { name: file.name, label: null });
    event.target.value = "";
  };

  const result = analysis.result;

  return (
    <Status loading={loading} error={error}>
      {domain && (
        <div className="stack-lg">
          <header className="page-head">
            <span className="eyebrow">{domain.sector} · Image</span>
            <h1>{domain.name}</h1>
            <p className="muted">{domain.task}</p>
          </header>

          <div className="analyze-grid">
            <section className="card stack" aria-labelledby="image-title">
              <div className="section-head">
                <h2 id="image-title">Image</h2>
                <div className="actions">
                  <button type="button" className="button" onClick={() => inputRef.current?.click()}>
                    Upload image
                  </button>
                  <input ref={inputRef} type="file" accept="image/*" hidden onChange={upload} />
                </div>
              </div>
              {selected ? (
                <figure className="preview">
                  <img src={selected.url} alt="Selected input" />
                  <figcaption className="muted small">
                    {selected.name}
                    {selected.label && ` · actual label: ${selected.label}`}
                  </figcaption>
                </figure>
              ) : (
                <p className="muted small">Upload a JPEG or PNG, or pick a held-out test image below.</p>
              )}
              <h3>Test images</h3>
              <div className="gallery">
                {domain.samples.map((sample) => (
                  <button
                    key={sample.name}
                    type="button"
                    className="thumb"
                    onClick={() => pickSample(sample)}
                    title={`Actual: ${sample.label}`}
                  >
                    <img src={api.visionSampleUrl(domainId, sample.name)} alt={sample.label} loading="lazy" />
                    <span>{sample.label}</span>
                  </button>
                ))}
              </div>
            </section>

            <section className="stack" aria-live="polite">
              {analysis.error && <p className="status status-error">{analysis.error.message}</p>}
              {!result && !analysis.error && (
                <p className="status">{analysis.pending ? "Analysing…" : "Choose an image to analyse."}</p>
              )}
              {result && (
                <div className={`stack${analysis.pending ? " is-stale" : ""}`}>
                  <div className="card result-head">
                    <div>
                      <span className="eyebrow">Consensus</span>
                      <p className="result-label">{result.consensus}</p>
                      {selected?.label && (
                        <p className={`small ${selected.label === result.consensus ? "match" : "mismatch"}`}>
                          Actual label: {selected.label} (
                          {selected.label === result.consensus ? "correct" : "incorrect"})
                        </p>
                      )}
                    </div>
                    <div className="result-confidence">
                      <span className="eyebrow">Agreement</span>
                      <p className="result-number">
                        {result.agreement}/{result.results.length}
                      </p>
                    </div>
                  </div>

                  <div className="vision-grid">
                    {result.results.map((r) => (
                      <article key={r.arch} className="card stack">
                        <div className="section-head">
                          <h3>{r.name}</h3>
                          <span className="muted small">{r.latency_ms.toFixed(0)} ms</span>
                        </div>
                        <p>
                          <strong>{r.prediction}</strong> <span className="muted">{percent(r.confidence, 1)}</span>
                        </p>
                        <img className="heatmap" src={r.heatmap} alt={`Grad-CAM heatmap from ${r.name}`} />
                        <BarList
                          label={`${r.name} class probabilities`}
                          items={r.probabilities.map((p) => ({ key: p.label, label: p.label, value: p.probability }))}
                          max={1}
                          highlight={r.prediction}
                          format={(v) => percent(v, 1)}
                        />
                      </article>
                    ))}
                  </div>

                  <div className="card stack">
                    <h3>Summary</h3>
                    <ul className="summary">
                      {result.summary.map((line) => (
                        <li key={line}>{line}</li>
                      ))}
                    </ul>
                    <p className="muted small">
                      Heatmaps are Grad-CAM: red areas raised the score of the predicted class the most.
                    </p>
                  </div>
                </div>
              )}
            </section>
          </div>

          <section className="card stack">
            <h2>Model comparison on held-out test images</h2>
            <VisionComparison domain={domain} />
          </section>

          <p className="muted small">
            Data: {domain.data_source}. <Link to="/models">Model details</Link>
          </p>
        </div>
      )}
    </Status>
  );
}
