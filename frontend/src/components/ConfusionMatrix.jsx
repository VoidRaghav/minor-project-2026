export default function ConfusionMatrix({ classes, matrix }) {
  return (
    <div className="table-wrap">
      <table className="matrix">
        <thead>
          <tr>
            <th scope="col">Actual \ Predicted</th>
            {classes.map((c) => (
              <th key={c} scope="col">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={classes[i]}>
              <th scope="row">{classes[i]}</th>
              {row.map((count, j) => (
                <td key={classes[j]} className={i === j ? "diag" : undefined}>
                  {count}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
