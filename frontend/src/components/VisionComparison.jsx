import { percent } from "../format.js";

export default function VisionComparison({ domain }) {
  if (!domain.models.length) {
    return <p className="status">No image models trained yet. Run python -m app.vision.train in the backend.</p>;
  }
  const best = Math.max(...domain.models.map((m) => m.macro_f1));

  return (
    <div className="table-wrap">
      <table className="matrix compare">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Accuracy</th>
            <th scope="col">Macro F1</th>
            {domain.classes.map((c) => (
              <th key={c} scope="col">
                Recall: {c}
              </th>
            ))}
            <th scope="col">Parameters</th>
            <th scope="col">Size</th>
            <th scope="col">Latency</th>
            <th scope="col">Grad-CAM deletion vs random</th>
          </tr>
        </thead>
        <tbody>
          {domain.models.map((m) => (
            <tr key={m.arch}>
              <th scope="row">{m.name}</th>
              <td className={m.macro_f1 === best ? "diag" : undefined}>{percent(m.accuracy, 1)}</td>
              <td className={m.macro_f1 === best ? "diag" : undefined}>{m.macro_f1.toFixed(3)}</td>
              {m.recall.map((r, i) => (
                <td key={domain.classes[i]}>{percent(r, 1)}</td>
              ))}
              <td>{m.params_m.toFixed(1)} M</td>
              <td>{m.size_mb.toFixed(0)} MB</td>
              <td>{m.latency_ms.toFixed(1)} ms</td>
              <td>
                {percent(m.deletion_drop_gradcam, 1)} vs {percent(m.deletion_drop_random, 1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
