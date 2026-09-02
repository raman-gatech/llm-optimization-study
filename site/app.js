const formatNumber = new Intl.NumberFormat("en-US");

function configureRepositoryLink() {
  if (!location.hostname.endsWith("github.io")) return;
  const owner = location.hostname.split(".")[0];
  const repository = location.pathname.split("/").filter(Boolean)[0];
  if (repository) document.querySelector("#repo-link").href = `https://github.com/${owner}/${repository}`;
}

async function loadMetrics() {
  const response = await fetch("data/summary.json");
  if (!response.ok) throw new Error(`summary request failed: ${response.status}`);
  const summary = await response.json();
  const scale = summary.experiment_scale;
  document.querySelector("#experiments").textContent = formatNumber.format(scale.result_json_files);
  document.querySelector("#requests").textContent = formatNumber.format(scale.completed_requests);
  document.querySelector("#tokens").textContent = `${(scale.total_tokens / 1e6).toFixed(1)}M`;
  document.querySelector("#completion").textContent = `${scale.recorded_completion_rate_pct}%`;
}

async function loadFigures() {
  const response = await fetch("data/figures.json");
  if (!response.ok) throw new Error(`figure request failed: ${response.status}`);
  const names = await response.json();
  const preferred = names.filter((name) => ["improvement_summary_heatmap_all.png", "tail_latency_qps_grid_2x2_best.png", "student_vs_teacher_speedup_heatmap.png", "efficiency_frontier_qps_sweep_logy.png"].includes(name));
  const gallery = document.querySelector("#gallery");
  gallery.replaceChildren(...preferred.map((name) => {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = `figures/${name}`;
    image.alt = name.replaceAll("_", " ").replace(".png", "");
    image.loading = "lazy";
    const caption = document.createElement("figcaption");
    caption.textContent = image.alt;
    figure.append(image, caption);
    return figure;
  }));
}

configureRepositoryLink();
Promise.all([loadMetrics(), loadFigures()]).catch((error) => console.error(error));
