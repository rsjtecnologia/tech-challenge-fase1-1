function renderInterpretationComparison(models) {
  const container = document.querySelector("#interpretationComparison");
  const metrics = [
    ["recall", "Recall"],
    ["precision", "Precisão"],
    ["f1", "F1"],
    ["accuracy", "Acurácia"],
  ];
  const table = document.createElement("table");
  const caption = table.createCaption();
  caption.textContent = "Resultados reais no mesmo conjunto de teste (holdout), não gerados pela LLM. Δ = otimizado − baseline em pontos percentuais (p.p.); valores negativos indicam queda.";
  const head = table.createTHead().insertRow();
  for (const label of ["Modelo / experimento GA", "Métrica", "Baseline", "Otimizado", "Δ (p.p.)"]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.appendChild(cell);
  }
  const body = table.createTBody();
  for (const [name, model] of Object.entries(models)) {
    for (const [metric, label] of metrics) {
      const row = body.insertRow();
      const baseline = model.baseline_metrics[metric];
      const optimized = model.optimized_metrics[metric];
      const difference = (optimized - baseline) * 100;
      const roundedDifference = Number(difference.toFixed(2));
      const values = [
        `${name} · ${model.selected_experiment}`,
        label,
        `${(baseline * 100).toFixed(2)}%`,
        `${(optimized * 100).toFixed(2)}%`,
        `${roundedDifference > 0 ? "+" : ""}${roundedDifference.toFixed(2)}`,
      ];
      for (const value of values) {
        row.insertCell().textContent = value;
      }
    }
  }
  container.replaceChildren(table);
}
