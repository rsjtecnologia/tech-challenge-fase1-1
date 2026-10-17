const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const state = { scenario: null, result: null, history: null };
const ROUTE_COLORS = ["#45d4e8", "#a78bfa", "#55d6a5", "#f8bd68", "#ff7b91"];

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || "Não foi possível carregar o plano.");
  return body;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 4500);
}

function formatKm(value) {
  return `${Number(value).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} km`;
}

function formatKg(value) {
  return `${Number(value).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} kg`;
}

function formatDuration(seconds) {
  return `${Number(seconds).toFixed(2)} s`;
}

function currentSolution() {
  return state.result.comparison[$("#routeView").value];
}

function renderSummary() {
  const baseline = state.result.comparison.baseline;
  const optimized = state.result.comparison.optimized;
  $("#deliveryCount").textContent = state.scenario.deliveries.length;
  $("#vehicleCount").textContent = state.scenario.vehicles.length;
  $("#routeDistance").textContent = Number(optimized.total_distance_km).toFixed(1);
  $("#baselineDistance").textContent = Number(baseline.total_distance_km).toFixed(1);
  $("#optimizedDistance").textContent = Number(optimized.total_distance_km).toFixed(1);
  $("#distanceSaved").textContent = formatKm(state.result.comparison.distance_saved_km);
  $("#distanceSavedPercent").textContent = `${state.result.comparison.distance_saved_percent}% comparado à base`;
  $("#servedCount").textContent = `${optimized.served_delivery_count}/${optimized.delivery_count}`;
  $("#lateCount").textContent = `${optimized.late_delivery_count} entrega(s) após o prazo estimado`;
  $("#scenarioNotice").textContent = state.result.scenario_description;
}

function projectPoint(point, bounds, width, height) {
  const margin = 55;
  const longitudeRange = Math.max(bounds.maxLon - bounds.minLon, 0.001);
  const latitudeRange = Math.max(bounds.maxLat - bounds.minLat, 0.001);
  return {
    x: margin + ((point.longitude - bounds.minLon) / longitudeRange) * (width - margin * 2),
    y: height - margin - ((point.latitude - bounds.minLat) / latitudeRange) * (height - margin * 2),
  };
}

function renderMap() {
  const solution = currentSolution();
  const width = 1000;
  const height = 590;
  const points = [state.result.depot, ...state.scenario.deliveries];
  const bounds = {
    minLat: Math.min(...points.map((point) => point.latitude)),
    maxLat: Math.max(...points.map((point) => point.latitude)),
    minLon: Math.min(...points.map((point) => point.longitude)),
    maxLon: Math.max(...points.map((point) => point.longitude)),
  };
  const depot = projectPoint(state.result.depot, bounds, width, height);
  let paths = "";
  const visited = new Set();
  let legend = "";

  solution.routes.forEach((route, index) => {
    const color = ROUTE_COLORS[index % ROUTE_COLORS.length];
    const coordinates = route.path.map((point) => projectPoint(point, bounds, width, height));
    const pointsString = coordinates.map((point) => `${point.x},${point.y}`).join(" ");
    paths += `<polyline points="${pointsString}" fill="none" stroke="${color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity=".82"/>`;
    route.stops.forEach((stop, stopIndex) => {
      const point = projectPoint(stop, bounds, width, height);
      visited.add(stop.delivery_id);
      paths += `
        <circle cx="${point.x}" cy="${point.y}" r="15" fill="#111625" stroke="${color}" stroke-width="4"/>
        <text x="${point.x}" y="${point.y + 5}" text-anchor="middle" fill="#fff" font-size="12" font-weight="700">${stopIndex + 1}</text>
      `;
    });
    legend += `<span><i style="--route-color:${color}"></i>${escapeHtml(route.vehicle_name)} · ${formatKm(route.distance_km)}</span>`;
  });

  const deliveryMarkers = state.scenario.deliveries
    .filter((delivery) => !visited.has(delivery.id))
    .map((delivery) => {
      const point = projectPoint(delivery, bounds, width, height);
      return `<circle cx="${point.x}" cy="${point.y}" r="10" fill="#ff7b91" stroke="#fff" stroke-width="2"/>`;
    })
    .join("");

  $("#routeMap").innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Rotas esquemáticas com ${solution.routes.length} veículos">
      <defs><pattern id="routeGrid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,.04)" stroke-width="1"/></pattern></defs>
      <rect width="${width}" height="${height}" fill="url(#routeGrid)"/>
      <text x="25" y="32" fill="#8892a8" font-size="13">Esquema geográfico · não representa vias ou trânsito</text>
      ${paths}
      ${deliveryMarkers}
      <path d="M ${depot.x} ${depot.y - 15} L ${depot.x + 13} ${depot.y + 12} L ${depot.x - 13} ${depot.y + 12} Z" fill="#f8bd68" stroke="#111625" stroke-width="2"/>
      <text x="${depot.x + 18}" y="${depot.y + 5}" fill="#f8bd68" font-size="13" font-weight="700">Central</text>
    </svg>
  `;
  $("#routeLegend").innerHTML = `${legend}<span><i class="route-depot-legend"></i>Central de distribuição</span>`;
}

function renderRoutes() {
  const solution = currentSolution();
  $("#vehicleRoutes").innerHTML = solution.routes.map((route, routeIndex) => `
    <article class="route-vehicle-card">
      <div class="route-vehicle-heading">
        <span class="route-color-dot" style="--route-color:${ROUTE_COLORS[routeIndex % ROUTE_COLORS.length]}"></span>
        <div><h4>${escapeHtml(route.vehicle_name)}</h4><small>${formatKm(route.distance_km)} · ${route.stops.length} parada(s)</small></div>
      </div>
      <div class="route-vehicle-capacity">
        <span>Carga ${formatKg(route.load_kg)} / ${formatKg(route.capacity_kg)}</span>
        <span>Autonomia restante ${formatKm(route.remaining_range_km)}</span>
      </div>
      <ol>${route.stops.map((stop) => `
        <li>
          <span class="priority-dot priority-${escapeHtml(stop.priority)}"></span>
          <div><strong>${escapeHtml(stop.name)}</strong><small>${escapeHtml(stop.arrival_time)} · prazo ${escapeHtml(state.scenario.deliveries.find((delivery) => delivery.id === stop.delivery_id)?.deadline_time || "—")} · ${formatKg(stop.demand_kg)}</small></div>
          ${stop.lateness_minutes > 0 ? `<span class="late-label">+${Math.round(stop.lateness_minutes)} min</span>` : ""}
        </li>
      `).join("")}</ol>
    </article>
  `).join("");
}

function renderDeliveries() {
  const solution = currentSolution();
  const assignments = new Map();
  solution.routes.forEach((route) => route.stops.forEach((stop) => assignments.set(stop.delivery_id, {
    vehicle: route.vehicle_name,
    arrival: stop.arrival_time,
    lateness: stop.lateness_minutes,
  })));
  $("#deliveryRows").innerHTML = state.scenario.deliveries.map((delivery) => {
    const assignment = assignments.get(delivery.id);
    return `
      <tr>
        <td><strong>${escapeHtml(delivery.name)}</strong><small>${escapeHtml(delivery.items.join(", "))}</small></td>
        <td><span class="priority-badge priority-${escapeHtml(delivery.priority)}">${escapeHtml(delivery.priority)}</span></td>
        <td>${formatKg(delivery.demand_kg)}</td>
        <td>${escapeHtml(delivery.deadline_time)}</td>
        <td>${delivery.requires_refrigeration ? "Sim" : "Não"}</td>
        <td>${assignment ? `${escapeHtml(assignment.vehicle)} · ${escapeHtml(assignment.arrival)}` : "Não atribuída"}</td>
      </tr>
    `;
  }).join("");
}

function renderConvergence() {
  const history = state.result.algorithm.history;
  const width = 600;
  const height = 190;
  const pad = 35;
  const values = history.map((point) => point.best_fitness);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1);
  const points = values.map((value, index) => {
    const x = pad + (index / Math.max(values.length - 1, 1)) * (width - pad * 2);
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  $("#gaChart").innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Convergência do fitness ao longo de ${history.length} gerações">
      <path d="M ${pad} ${pad} V ${height - pad} H ${width - pad}" fill="none" stroke="rgba(255,255,255,.18)"/>
      <polyline points="${points}" fill="none" stroke="#45d4e8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      <text x="${pad}" y="${height - 8}" fill="#8d96aa" font-size="12">Geração 1</text>
      <text x="${width - pad}" y="${height - 8}" fill="#8d96aa" font-size="12" text-anchor="end">Geração ${history.length}</text>
      <text x="${pad}" y="18" fill="#8d96aa" font-size="12">Fitness menor é melhor · distância + atraso ponderado</text>
    </svg>
  `;
  const config = state.result.algorithm.config;
  $("#gaParameters").innerHTML = `
    <span>População: <strong>${config.population_size}</strong></span>
    <span>Gerações: <strong>${config.generations}</strong></span>
    <span>Mutação: <strong>${Math.round(config.mutation_rate * 100)}%</strong></span>
    <span>Avaliações únicas: <strong>${state.result.algorithm.evaluated_chromosomes}</strong></span>
    <span>Tempo: <strong>${formatDuration(state.result.elapsed_seconds)}</strong></span>
  `;
}

function renderAll() {
  renderSummary();
  renderMap();
  renderRoutes();
  renderDeliveries();
  renderConvergence();
}

async function optimizeRoutes() {
  const button = $("#optimizeButton");
  const seedInput = $("#routeSeed");
  if (!seedInput.checkValidity()) {
    seedInput.reportValidity();
    return;
  }
  button.disabled = true;
  $("#scenarioNotice").textContent = "Executando busca genética…";
  try {
    state.result = await request("/api/rotas/otimizar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seed: Number(seedInput.value) }),
    });
    renderAll();
    await refreshHistory();
    showToast("Rotas otimizadas e salvas localmente.");
  } catch (error) {
    $("#scenarioNotice").textContent = error.message;
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
}

async function refreshHistory() {
  const notice = $("#routeHistoryNotice");
  try {
    state.history = await request("/api/rotas/historico");
    const count = state.history.run_count;
    const runLabel = count === 1 ? "simulação sintética registrada" : "simulações sintéticas registradas";
    notice.textContent = count >= state.history.minimum_runs_for_trend
      ? `${count} ${runLabel} nos últimos ${state.history.window_days} dias. O comparativo mede execuções do algoritmo, não operações reais.`
      : `${count} ${count === 1 ? "simulação registrada" : "simulações registradas"} nos últimos ${state.history.window_days} dias. Registre ao menos mais uma execução para comparar variações; isso não representa uma tendência operacional.`;
  } catch (error) {
    notice.textContent = `Não foi possível carregar o histórico: ${error.message}`;
  }
}

function renderAssistantResult(result) {
  const content = result.content;
  const output = $("#assistantOutput");
  const mode = result.source === "local"
    ? "Resposta local, baseada nos dados do plano (sem LLM)."
    : "Resposta gerada pela LLM configurada.";
  if (result.action === "instructions") {
    output.innerHTML = `
      <p class="assistant-mode">${escapeHtml(mode)}</p>
      <p>${escapeHtml(content.summary)}</p>
      ${content.vehicle_instructions.map((vehicle) => `
        <h4>${escapeHtml(vehicle.vehicle)}</h4>
        <ol>${vehicle.ordered_stops.map((stop) => `<li>${escapeHtml(typeof stop === "string" ? stop : JSON.stringify(stop))}</li>`).join("")}</ol>
        <p>${escapeHtml(Array.isArray(vehicle.notes) ? vehicle.notes.join(" ") : vehicle.notes)}</p>
      `).join("")}
      <h4>Verificações operacionais</h4>
      <ul>${content.operational_checks.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <p><strong>${escapeHtml(content.disclaimer)}</strong></p>
    `;
    return;
  }
  if (result.action === "daily_report" || result.action === "weekly_report") {
    output.innerHTML = `
      <p class="assistant-mode">${escapeHtml(mode)}</p>
      <p><strong>${escapeHtml(content.period)}</strong></p>
      <p>${escapeHtml(content.summary)}</p>
      <h4>Métricas</h4>
      <ul>${Object.entries(content.key_metrics).map(([key, value]) => `<li><strong>${escapeHtml(key)}:</strong> ${escapeHtml(typeof value === "string" ? value : JSON.stringify(value))}</li>`).join("")}</ul>
      <h4>Melhorias propostas</h4>
      <ul>${content.improvements.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <h4>Limitações</h4>
      <ul>${content.caveats.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      <p><strong>${escapeHtml(content.disclaimer)}</strong></p>
    `;
    return;
  }
  output.innerHTML = `
    <p class="assistant-mode">${escapeHtml(mode)}</p>
    <p>${escapeHtml(content.answer)}</p>
    <h4>Dados que sustentam a resposta</h4>
    <ul>${content.supporting_facts.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <h4>Informações não disponíveis</h4>
    <ul>${content.missing_information.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    <p><strong>${escapeHtml(content.disclaimer)}</strong></p>
  `;
}

async function askAssistant(action, question = null, button = null) {
  const output = $("#assistantOutput");
  if (button) button.disabled = true;
  output.textContent = "Gerando resposta fundamentada no plano…";
  try {
    const result = await request("/api/rotas/assistente", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, question }),
    });
    renderAssistantResult(result);
  } catch (error) {
    output.textContent = error.message;
    showToast(error.message);
  } finally {
    if (button) button.disabled = false;
  }
}

async function loadPage() {
  try {
    state.scenario = await request("/api/rotas/cenario");
    $("#scenarioNotice").textContent = state.scenario.description;
    try {
      state.result = await request("/api/rotas/resultado");
    } catch (_error) {
      await optimizeRoutes();
    }
    if (state.result) renderAll();
    await refreshHistory();
  } catch (error) {
    $("#scenarioNotice").textContent = error.message;
    showToast(error.message);
  }
}

function bindEvents() {
  $("#optimizeButton").addEventListener("click", optimizeRoutes);
  $("#routeView").addEventListener("change", () => {
    if (!state.result) return;
    renderMap();
    renderRoutes();
    renderDeliveries();
  });
  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => askAssistant(button.dataset.action, null, button));
  });
  $("#routeQuestionForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("#routeQuestion").value.trim();
    if (!question) return;
    askAssistant("question", question, event.submitter);
  });
  $("#menuButton").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadPage();
});
