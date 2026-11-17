document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("image-input");
  const fileStatus = document.getElementById("file-status");
  const sampleTiles = document.querySelectorAll(".sample-tile");
  const radioCards = document.querySelectorAll(".radio-card");
  const autoTrimBox = document.getElementById("auto-trim");
  const tauSlider = document.getElementById("tau-slider");
  const tauDisplay = document.getElementById("tau-display");
  const kpiTauVal = document.getElementById("kpi-tau-val");

  let activeSample = document.querySelector(".sample-tile.selected")?.dataset.sample || "test_class_14.png";
  let uploadedFile = null;

  function renderPrediction(data) {
    document.getElementById("stage-orig").src = data.stages.orig_uri;
    document.getElementById("stage-resized").src = data.stages.resized_uri;
    document.getElementById("stage-clahe").src = data.stages.clahe_uri;
    document.getElementById("stage-hog").src = data.stages.hog_uri;

    const trimBadge = document.getElementById("trim-badge");
    trimBadge.textContent = data.auto_trim_margin ? "Đã cắt lề 10%" : "Giữ toàn khung";

    document.getElementById("diag-size").textContent = `${data.diagnostics.width} × ${data.diagnostics.height} px`;
    document.getElementById("diag-res-note").textContent = data.diagnostics.resolution_status;
    document.getElementById("diag-bright").textContent = `${data.diagnostics.mean_brightness} / 255`;
    document.getElementById("diag-bright-note").textContent = data.diagnostics.brightness_status;

    ["red", "blue", "yellow", "dark"].forEach((k) => {
      const val = data.diagnostics.color_ratios[k];
      document.getElementById(`col-${k}-bar`).style.width = `${val}%`;
      document.getElementById(`col-${k}-val`).textContent = `${val}%`;
    });

    const safetyPill = document.getElementById("safety-pill");
    safetyPill.className = `safety-pill ${data.safety_level}`;
    safetyPill.textContent = data.safety_label;

    document.getElementById("pred-category").textContent =
      `${data.primary.category} · F1 lớp trên Test: ${data.primary.knn_test_f1}%`;
    document.getElementById("pred-title").textContent =
      `Lớp ${data.primary.class_id} – ${data.primary.name_vi}`;
    document.getElementById("pred-en").textContent = data.primary.name_en;
    document.getElementById("pred-score").textContent = `${data.primary.confidence.toFixed(2)}%`;
    document.getElementById("pred-meta").textContent = `${data.model_choice} · ${data.latency_ms} ms`;

    document.getElementById("top5-list").innerHTML = data.top_predictions
      .map(
        (r, i) => `
      <div class="top5-row ${i === 0 ? "top1" : ""}">
        <div class="top5-name">#${r.rank} · Lớp ${r.class_id} – ${r.name_vi}</div>
        <div class="top5-track"><div class="top5-fill" style="width: ${r.confidence}%;"></div></div>
        <div class="top5-pct">${r.confidence.toFixed(2)}%</div>
      </div>`
      )
      .join("");

    document.getElementById("agree-ratio").textContent = `${data.agreement_count}/${data.agreement_total}`;
    document.getElementById("consensus-tbody").innerHTML = data.consensus
      .map(
        (c) => `
      <tr>
        <td>${c.choice}</td>
        <td>Lớp ${c.class_id} – ${c.name_vi}</td>
        <td>${c.category}</td>
        <td class="num">${c.confidence.toFixed(2)}%</td>
        <td class="num">${c.latency_ms.toFixed(2)} ms</td>
      </tr>`
      )
      .join("");

    document.getElementById("calibration-note").innerHTML =
      `<strong>Lưu ý hiệu chuẩn xác suất:</strong> Giá trị phần trăm là ${data.score_type} (chưa hiệu chuẩn xác suất hậu nghiệm). Do đó cần kết hợp ngưỡng từ chối τ và bảng đồng thuận đa mô hình trước khi đưa ra cảnh báo ADAS.`;
  }

  async function submitPrediction() {
    const formData = new FormData();
    const chosenRadio = document.querySelector('input[name="model_choice"]:checked');
    formData.append("model_choice", chosenRadio ? chosenRadio.value : "K-NN F3-PCA 128 (Tối ưu K-NN)");
    formData.append("auto_trim", autoTrimBox.checked ? "true" : "false");
    formData.append("tau", tauSlider.value);

    if (uploadedFile) {
      formData.append("image", uploadedFile);
    } else {
      formData.append("sample_name", activeSample);
    }

    const resp = await fetch("/api/predict", { method: "POST", body: formData });
    if (!resp.ok) return;
    const data = await resp.json();
    renderPrediction(data);
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      uploadedFile = e.dataTransfer.files[0];
      fileStatus.textContent = `Tệp tải lên: ${uploadedFile.name}`;
      sampleTiles.forEach((t) => t.classList.remove("selected"));
      submitPrediction();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      uploadedFile = fileInput.files[0];
      fileStatus.textContent = `Tệp tải lên: ${uploadedFile.name}`;
      sampleTiles.forEach((t) => t.classList.remove("selected"));
      submitPrediction();
    }
  });

  sampleTiles.forEach((tile) => {
    tile.addEventListener("click", () => {
      uploadedFile = null;
      fileInput.value = "";
      activeSample = tile.dataset.sample;
      autoTrimBox.checked = tile.dataset.autoTrim === "true";
      fileStatus.textContent = `Đang sử dụng mẫu chuẩn: ${activeSample}`;
      sampleTiles.forEach((t) => t.classList.remove("selected"));
      tile.classList.add("selected");
      submitPrediction();
    });
  });

  radioCards.forEach((card) => {
    card.addEventListener("click", () => {
      radioCards.forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      submitPrediction();
    });
  });

  autoTrimBox.addEventListener("change", submitPrediction);
  tauSlider.addEventListener("input", () => {
    tauDisplay.textContent = parseFloat(tauSlider.value).toFixed(2);
    kpiTauVal.textContent = parseFloat(tauSlider.value).toFixed(2);
  });
  tauSlider.addEventListener("change", submitPrediction);

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    submitPrediction();
  });
});
