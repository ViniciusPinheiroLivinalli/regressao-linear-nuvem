/* Gráficos simples (Chart.js): treino e teste com rótulos.
 * Teste sem rótulos não tem gráfico (só tabela de resultado e download). */
(() => {
  const charts = {};
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  function parseCsv(text) {
    const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
    const header = lines[0].split(",").map((s) => s.trim());
    const rows = lines.slice(1).map((l) => l.split(",").map(Number));
    const col = (name) => rows.map((r) => r[header.indexOf(name)]);
    return { has: (n) => header.includes(n), col };
  }

  const toPoints = (ys) => ys.map((y, i) => ({ x: i, y }));

  function line(label, data, color, dash = []) {
    return {
      label, data, borderColor: color, borderWidth: 2, borderDash: dash,
      pointRadius: 0, tension: 0, normalized: true,
    };
  }

  function draw(key, title, datasets) {
    // Sem Chart.js (CDN fora do ar ou bloqueado) a página segue funcionando, só sem gráfico.
    if (typeof Chart === "undefined") {
      console.warn("Chart.js não carregou; gráfico de " + key + " omitido.");
      clear(key);
      return;
    }
    const wrap = document.getElementById(key + "-chart-wrap");
    wrap.hidden = false; // precisa estar visível antes de criar o gráfico
    if (charts[key]) charts[key].destroy();

    Chart.defaults.color = css("--muted");
    Chart.defaults.borderColor = css("--border");
    charts[key] = new Chart(document.getElementById(key + "-chart"), {
      type: "line",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        parsing: false,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        scales: {
          x: { type: "linear", title: { display: true, text: "Linha / tempo" } },
          y: { title: { display: true, text: "Valor" } },
        },
        plugins: {
          title: { display: true, text: title },
          legend: { position: "bottom" },
          decimation: { enabled: true, algorithm: "lttb", samples: 1000 },
        },
      },
    });
  }

  function clear(key) {
    if (charts[key]) { charts[key].destroy(); charts[key] = null; }
    document.getElementById(key + "-chart-wrap").hidden = true;
  }

  window.SeriesPlots = {
    /** CSV de previsões gerado a partir do arquivo de treino (tem `time`). */
    setTrain(csvText) {
      const d = parseCsv(csvText);
      draw("train", "Treino: série real × ajuste do modelo (dados já vistos pelo modelo)", [
        line("Série real", toPoints(d.col("time")), "#6b7280"),
        line("Ajuste do modelo", toPoints(d.col("prediction")), "#2457c5"),
      ]);
    },

    /** CSV de previsões do teste. Só desenha se tiver `time` (rótulos). */
    setTest(csvText) {
      const d = parseCsv(csvText);
      if (!d.has("time")) { clear("test"); return; } // sem rótulos: sem gráfico
      draw("test", "Teste: série real × predição do modelo", [
        line("Série real", toPoints(d.col("time")), "#f59e0b"),
        line("Predição", toPoints(d.col("prediction")), "#dc2626", [6, 4]),
      ]);
    },

    reset() { clear("train"); clear("test"); },
  };
})();
