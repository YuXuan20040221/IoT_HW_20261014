import {
  calculateMetrics,
  fitLinearRegression,
  predict,
  seededShuffleSplit,
  selectForwardFeatures,
} from "./regression.mjs";

const MAX_FILE_BYTES = 50 * 1024 * 1024;
const MAX_ROWS = 50000;
const MAX_CATEGORIES = 12;
const state = { headers: [], rows: [], featureColumns: [], defaults: [], result: null };
const $ = selector => document.querySelector(selector);
const elements = {
  file: $("#csv-file"),
  fileStatus: $("#file-status"),
  message: $("#app-message"),
  controls: $("#analysis-controls"),
  summary: $("#dataset-summary"),
  target: $("#target-select"),
  testSize: $("#test-size"),
  featureCount: $("#feature-count"),
  randomSeed: $("#random-seed"),
  method: $("#selection-method"),
  featureList: $("#feature-list"),
  selectedCount: $("#selected-count"),
  featureCountHint: $("#feature-count-hint"),
  notes: $("#column-notes"),
  run: $("#run-analysis"),
  results: $("#results"),
};

function showMessage(text, kind = "") {
  elements.message.textContent = text;
  elements.message.className = `message ${kind}`.trim();
  elements.message.hidden = !text;
}

function parseNumeric(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const text = String(value ?? "").trim();
  if (!text) return null;
  const normalized = text.replace(/[$£€¥,\s%]/g, "");
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(normalized)) return null;
  const number = Number(normalized);
  return Number.isFinite(number) ? number : null;
}

function normalizeRecords(records) {
  const rawHeaders = records.meta.fields.map(header => String(header).trim());
  const counts = new Map();
  state.headers = rawHeaders.map((header, index) => {
    const base = header || `欄位_${index + 1}`;
    const count = counts.get(base) || 0;
    counts.set(base, count + 1);
    return count ? `${base}_${count + 1}` : base;
  });
  const rawRows = records.data.slice(0, MAX_ROWS);
  const numericTargets = [];
  const featureColumns = [];
  const skipped = [];

  state.headers.forEach((header, columnIndex) => {
    const values = rawRows.map(row => row[rawHeaders[columnIndex]]);
    const idLike = /^(?:id|idx|index|record|serial|row(?: ?number)?|unnamed(?:: ?0)?|.*(?:_id|_idx|_index|_serial))$/i.test(header.trim());
    const numericValues = values.map(parseNumeric);
    const nonEmpty = values.filter(value => String(value ?? "").trim() !== "");
    const numericCount = numericValues.filter(value => value !== null).length;
    const numericRatio = nonEmpty.length ? numericCount / nonEmpty.length : 0;
    if (numericRatio >= 0.8 && numericCount > 1) {
      const distinctNumeric = new Set(numericValues.filter(value => value !== null)).size;
      if (!idLike && distinctNumeric >= 3) numericTargets.push(header);
      if (!idLike && distinctNumeric > 1) featureColumns.push({
        name: header,
        source: header,
        values: numericValues,
        recommended: true,
      });
      else if (!idLike) skipped.push(`${header}（常數欄位）`);
      else skipped.push(`${header}（識別碼）`);
      return;
    }

    const dateValues = values.map(value => {
      const text = String(value ?? "").trim();
      if (!text || !/[-/:T]/.test(text)) return null;
      const date = Date.parse(text);
      return Number.isFinite(date) ? date / 86400000 : null;
    });
    const dateCount = dateValues.filter(value => value !== null).length;
    if (dateCount >= Math.max(2, Math.ceil(nonEmpty.length * 0.8))) {
      const distinctDates = new Set(dateValues.filter(value => value !== null)).size;
      if (!idLike && distinctDates > 1) featureColumns.push({
        name: `${header}（日期）`,
        source: header,
        values: dateValues,
        recommended: true,
      });
      else if (!idLike) skipped.push(`${header}（常數欄位）`);
      else skipped.push(`${header}（識別碼）`);
      return;
    }

    const categories = [...new Set(nonEmpty.map(value => String(value).trim()))];
    if (categories.length >= 2 && categories.length <= MAX_CATEGORIES && !idLike) {
      categories.slice(1).forEach(category => {
        featureColumns.push({
          name: `${header} = ${category}`,
          source: header,
          values: values.map(value => String(value ?? "").trim() === category ? 1 : 0),
          recommended: true,
        });
      });
    } else {
      skipped.push(`${header}${idLike ? "（識別碼）" : categories.length < 2 ? "（常數欄位）" : "（非數值或類別過多）"}`);
    }
  });

  state.featureColumns = featureColumns;
  state.defaults = featureColumns.filter(feature => feature.recommended).slice(0, 20).map(feature => feature.name);
  state.rows = rawRows.map((row, index) => {
    const normalizedRow = {};
    state.headers.forEach((header, columnIndex) => {
      normalizedRow[header] = row[rawHeaders[columnIndex]];
    });
    normalizedRow.__rowNumber = index + 2;
    return normalizedRow;
  });
  return { numericTargets, skipped, truncated: records.data.length > MAX_ROWS };
}

function renderFeatureOptions() {
  elements.featureList.replaceChildren();
  state.featureColumns.filter(feature => feature.name !== elements.target.value).forEach(feature => {
    const label = document.createElement("label");
    label.className = "feature-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = feature.name;
    checkbox.checked = state.defaults.includes(feature.name);
    checkbox.addEventListener("change", updateFeatureCount);
    const text = document.createElement("span");
    text.textContent = feature.name;
    label.append(checkbox, text);
    elements.featureList.append(label);
  });
  updateFeatureCount();
}

function updateFeatureCount() {
  const count = elements.featureList.querySelectorAll("input:checked").length;
  elements.selectedCount.textContent = `${count} 個已選`;
  const maxFeatures = Math.min(39, count);
  elements.featureCount.max = String(maxFeatures || 1);
  if (Number(elements.featureCount.value) > maxFeatures) elements.featureCount.value = String(maxFeatures);
  elements.featureCountHint.textContent = `最多 ${maxFeatures} 個；預設最多 ${Math.min(5, maxFeatures)} 個。`;
  if (count > 0 && !elements.featureCount.dataset.userChanged) {
    elements.featureCount.value = String(Math.min(5, maxFeatures));
  }
}

function readConfiguration() {
  const selected = [...elements.featureList.querySelectorAll("input:checked")].map(input => input.value);
  if (!elements.target.value) throw new Error("請先選擇一個連續數值 Target。");
  if (!selected.length) throw new Error("請至少勾選一個輸入特徵。");
  if (selected.length > 39) throw new Error("輸入特徵最多可選 39 個。");

  const testPercent = elements.testSize.value.trim() === "" ? 20 : Number(elements.testSize.value);
  if (!Number.isInteger(testPercent) || testPercent < 10 || testPercent > 40) {
    throw new Error("測試資料比例請輸入 10 到 40 之間的整數。");
  }
  const requestedCount = elements.featureCount.value.trim() === "" ? Math.min(5, selected.length) : Number(elements.featureCount.value);
  if (!Number.isInteger(requestedCount) || requestedCount < 1 || requestedCount > Math.min(39, selected.length)) {
    throw new Error(`最多使用特徵數須為 1 到 ${Math.min(39, selected.length)} 之間的整數。`);
  }
  const seed = elements.randomSeed.value.trim() === "" ? 42 : Number(elements.randomSeed.value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 999999) {
    throw new Error("隨機種子須為 0 到 999999 之間的整數。");
  }
  return { selected, testRatio: testPercent / 100, requestedCount, seed };
}

function getRows(target, featureNames) {
  const columnsByName = new Map(state.featureColumns.map(feature => [feature.name, feature]));
  const rows = [];
  const targetValues = [];
  const rowNumbers = [];
  for (let index = 0; index < state.rows.length; index += 1) {
    const targetValue = parseNumeric(state.rows[index][target]);
    const values = featureNames.map(name => columnsByName.get(name).values[index]);
    if (targetValue === null || values.some(value => !Number.isFinite(value))) continue;
    rows.push(values);
    targetValues.push(targetValue);
    rowNumbers.push(state.rows[index].__rowNumber);
  }
  return { rows, targets: targetValues, rowNumbers };
}

function runAnalysis() {
  elements.results.hidden = true;
  try {
    const config = readConfiguration();
    const data = getRows(elements.target.value, config.selected);
    if (data.rows.length < 12) throw new Error(`清理缺失、格式錯誤或非有限值後只剩 ${data.rows.length} 筆資料；至少需要 12 筆。`);
    const split = seededShuffleSplit(data.rows.length, config.testRatio, config.seed);
    if (split.train.length < config.requestedCount + 2) {
      throw new Error(`訓練資料只有 ${split.train.length} 筆，無法使用 ${config.requestedCount} 個特徵；請減少特徵或降低測試比例。`);
    }

    const trainRows = split.train.map(index => data.rows[index]);
    const trainTargets = split.train.map(index => data.targets[index]);
    const testRows = split.test.map(index => data.rows[index]);
    const testTargets = split.test.map(index => data.targets[index]);
    const validationSplit = seededShuffleSplit(trainRows.length, 0.2, config.seed + 1);
    const innerTrainRows = validationSplit.train.map(index => trainRows[index]);
    const innerTrainTargets = validationSplit.train.map(index => trainTargets[index]);
    const validationRows = validationSplit.test.map(index => trainRows[index]);
    const validationTargets = validationSplit.test.map(index => trainTargets[index]);
    let featureOrder = config.selected;
    if (elements.method.value === "forward") {
      const rankingData = downsampleTraining(innerTrainRows, innerTrainTargets, 2500);
      const selection = selectForwardFeatures(rankingData.rows, rankingData.targets, config.selected, config.requestedCount);
      featureOrder = selection.selectedIndices.map(index => config.selected[index]);
    } else {
      featureOrder = config.selected.slice(0, config.requestedCount);
    }

    const maxCount = Math.min(config.requestedCount, featureOrder.length, innerTrainRows.length - 2);
    if (maxCount < 1) throw new Error("訓練資料不足以保留驗證資料；請增加資料筆數或調高測試比例。");
    const models = [];
    for (let count = 1; count <= maxCount; count += 1) {
      const names = featureOrder.slice(0, count);
      const indices = names.map(name => config.selected.indexOf(name));
      const innerTrainSubset = innerTrainRows.map(row => indices.map(index => row[index]));
      const validationSubset = validationRows.map(row => indices.map(index => row[index]));
      const testSubset = testRows.map(row => indices.map(index => row[index]));
      const validationModel = fitLinearRegression(innerTrainSubset, innerTrainTargets, names);
      const validationPredictions = validationSubset.map(row => predict(validationModel, row));
      const validationMetrics = calculateMetrics(validationTargets, validationPredictions);
      const trainSubset = trainRows.map(row => indices.map(index => row[index]));
      const model = fitLinearRegression(trainSubset, trainTargets, names);
      const predictions = testSubset.map(row => predict(model, row));
      models.push({
        names,
        model,
        predictions,
        metrics: calculateMetrics(testTargets, predictions),
        validationRmse: validationMetrics.rmse,
      });
    }
    const best = models.reduce((previous, current) => current.validationRmse < previous.validationRmse ? current : previous);
    state.result = {
      best,
      models,
      testTargets,
      testRows: split.test.map(index => data.rows[index]),
      testRowNumbers: split.test.map(index => data.rowNumbers[index]),
      trainCount: split.train.length,
      skippedRows: state.rows.length - data.rows.length,
      featureNames: best.names,
      fullFeatureNames: config.selected,
      target: elements.target.value,
      featureOrder,
      seed: config.seed,
    };
    renderResults();
    showMessage(
      `分析完成：使用 ${data.rows.length.toLocaleString()} 筆有效資料，${state.result.skippedRows.toLocaleString()} 筆因缺少或無效值排除。`,
      "success",
    );
  } catch (error) {
    showMessage(error instanceof Error ? error.message : "分析失敗，請檢查 CSV 欄位與參數。", "error");
  }
}

function downsampleTraining(rows, targets, limit) {
  if (rows.length <= limit) return { rows, targets };
  const sampledRows = [];
  const sampledTargets = [];
  for (let index = 0; index < limit; index += 1) {
    const source = Math.floor(index * (rows.length - 1) / (limit - 1));
    sampledRows.push(rows[source]);
    sampledTargets.push(targets[source]);
  }
  return { rows: sampledRows, targets: sampledTargets };
}

function formatNumber(value, digits = 4) {
  if (value === null || !Number.isFinite(value)) return "N/A";
  return new Intl.NumberFormat("zh-TW", { maximumFractionDigits: digits }).format(value);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function renderResults() {
  const result = state.result;
  const metrics = result.best.metrics;
  $("#metric-r2").textContent = formatNumber(metrics.r2, 5);
  $("#metric-mae").textContent = formatNumber(metrics.mae, 4);
  $("#metric-mse").textContent = formatNumber(metrics.mse, 4);
  $("#metric-rmse").textContent = formatNumber(metrics.rmse, 4);
  $("#active-features").textContent = result.featureNames.join(" · ");
  $("#split-summary").textContent = `驗證集選出 ${result.featureNames.length} 個特徵 · 訓練 ${result.trainCount.toLocaleString()} 筆 · 測試 ${result.testTargets.length.toLocaleString()} 筆`;
  $("#equation").textContent = `${result.target} = ${formatNumber(result.best.model.intercept, 5)}`
    + result.best.names.map((name, index) => ` ${result.best.model.coefficients[index] >= 0 ? "+" : "−"} ${formatNumber(Math.abs(result.best.model.coefficients[index]), 5)} × (${name})`).join("");

  $("#coefficient-rows").innerHTML = [
    `<tr><td><strong>截距（Intercept）</strong></td><td class="number">${formatNumber(result.best.model.intercept, 6)}</td><td>基準值</td></tr>`,
    ...result.best.names.map((name, index) => {
      const coefficient = result.best.model.coefficients[index];
      return `<tr><td>${escapeHtml(name)}</td><td class="number">${formatNumber(coefficient, 6)}</td><td class="${coefficient >= 0 ? "positive" : "negative"}">${coefficient >= 0 ? "正向" : "負向"}</td></tr>`;
    }),
  ].join("");
  renderChart($("#rmse-chart"), result.models.map(model => ({ x: model.names.length, y: model.metrics.rmse })), "RMSE");
  renderChart($("#r2-chart"), result.models.map(model => ({ x: model.names.length, y: model.metrics.r2 })), "R²");
  renderPredictor();
  renderResiduals();
  elements.results.hidden = false;
}

function renderChart(container, points, label) {
  const valid = points.filter(point => Number.isFinite(point.y));
  if (valid.length < 2) {
    container.innerHTML = '<div class="chart-empty">至少比較兩種特徵數，才會顯示趨勢圖。</div>';
    return;
  }
  const width = 560;
  const height = 235;
  const plot = { left: 55, top: 18, right: 18, bottom: 36 };
  const min = Math.min(...valid.map(point => point.y));
  const max = Math.max(...valid.map(point => point.y));
  const spread = max - min || Math.max(Math.abs(max) * 0.05, 1);
  const yMin = min - spread * 0.12;
  const yMax = max + spread * 0.12;
  const x = index => plot.left + index * (width - plot.left - plot.right) / Math.max(1, valid.length - 1);
  const y = value => plot.top + (yMax - value) * (height - plot.top - plot.bottom) / (yMax - yMin);
  const grid = Array.from({ length: 4 }, (_, index) => {
    const gy = plot.top + index * (height - plot.top - plot.bottom) / 3;
    const value = yMax - index * (yMax - yMin) / 3;
    return `<line x1="${plot.left}" y1="${gy}" x2="${width - plot.right}" y2="${gy}" stroke="#e8edf4"/><text x="${plot.left - 8}" y="${gy + 3}" text-anchor="end" fill="#8995a8" font-size="9">${escapeHtml(formatNumber(value, 3))}</text>`;
  }).join("");
  const path = valid.map((point, index) => `${index ? "L" : "M"} ${x(index)} ${y(point.y)}`).join(" ");
  const circles = valid.map((point, index) => `<circle cx="${x(index)}" cy="${y(point.y)}" r="4" fill="#5145e5"><title>${point.x} 個特徵：${escapeHtml(formatNumber(point.y, 5))}</title></circle>`).join("");
  const ticks = valid.map((point, index) => `<text x="${x(index)}" y="${height - 12}" text-anchor="middle" fill="#8995a8" font-size="9">${point.x}</text>`).join("");
  container.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(label)} 比較圖">${grid}<path d="${path}" fill="none" stroke="#7269f4" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>${circles}${ticks}<text x="${width / 2}" y="${height - 1}" text-anchor="middle" fill="#71809b" font-size="9">特徵數</text></svg>`;
}

function renderPredictor() {
  const result = state.result;
  const container = $("#predictor-fields");
  container.replaceChildren();
  const means = new Map(result.fullFeatureNames.map((name, index) => [
    name,
    result.testRows.reduce((sum, row) => sum + row[index], 0) / result.testRows.length,
  ]));
  result.featureNames.forEach(name => {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = String(Number(means.get(name).toPrecision(7)));
    input.dataset.feature = name;
    input.addEventListener("input", updatePrediction);
    const label = document.createElement("label");
    label.className = "predictor-field";
    const caption = document.createElement("span");
    caption.textContent = name;
    label.append(caption, input);
    container.append(label);
  });
  updatePrediction();
}

function updatePrediction() {
  const result = state.result;
  if (!result) return;
  const values = result.featureNames.map(name => {
    const input = $(`#predictor-fields input[data-feature="${CSS.escape(name)}"]`);
    return parseNumeric(input?.value);
  });
  if (values.some(value => value === null)) {
    $("#live-prediction").textContent = "請輸入有效數值";
    return;
  }
  $("#live-prediction").textContent = formatNumber(predict(result.best.model, values), 4);
}

function renderResiduals() {
  const result = state.result;
  const rows = result.testTargets.map((actual, index) => {
    const selectedValues = result.featureNames.map(name => {
      const fullIndex = result.fullFeatureNames.indexOf(name);
      return result.testRows[index][fullIndex];
    });
    const prediction = predict(result.best.model, selectedValues);
    const residual = actual - prediction;
    return { actual, prediction, residual, rowNumber: result.testRowNumbers[index] };
  }).sort((left, right) => Math.abs(right.residual) - Math.abs(left.residual));
  const cutoff = result.best.metrics.rmse * 2;
  $("#residual-count").textContent = `${rows.length.toLocaleString()} 筆測試資料`;
  $("#residual-rows").innerHTML = rows.slice(0, 10).map((row, index) => {
    const isHigh = Math.abs(row.residual) > cutoff;
    return `<tr><td>#${index + 1}</td><td>第 ${row.rowNumber} 列</td><td class="number">${formatNumber(row.actual, 4)}</td><td class="number">${formatNumber(row.prediction, 4)}</td><td class="number ${row.residual >= 0 ? "positive" : "negative"}">${row.residual >= 0 ? "+" : ""}${formatNumber(row.residual, 4)}</td><td class="number">${formatNumber(Math.abs(row.residual), 4)}</td><td><span class="status-pill ${isHigh ? "status-high" : "status-normal"}">${isHigh ? "高殘差" : "一般範圍"}</span></td></tr>`;
  }).join("");
}

function downloadPredictions() {
  const result = state.result;
  if (!result) return;
  const header = ["row_number", "actual", "predicted", "residual", "absolute_error"];
  const lines = [header, ...result.testTargets.map((actual, index) => {
    const values = result.featureNames.map(name => result.testRows[index][result.fullFeatureNames.indexOf(name)]);
    const prediction = predict(result.best.model, values);
    return [result.testRowNumbers[index], actual, prediction, actual - prediction, Math.abs(actual - prediction)];
  })].map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(","));
  const blob = new Blob(["\uFEFF", lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "regression_predictions.csv";
  anchor.click();
  URL.revokeObjectURL(url);
}

elements.file.addEventListener("change", () => {
  const file = elements.file.files?.[0];
  if (!file) return;
  elements.results.hidden = true;
  elements.controls.hidden = true;
  state.result = null;
  showMessage("");
  if (file.size > MAX_FILE_BYTES) {
    showMessage("CSV 檔案超過 50 MB，請先縮小資料檔。", "error");
    elements.file.value = "";
    return;
  }
  elements.fileStatus.textContent = `正在讀取 ${file.name}…`;
  if (!window.Papa) {
    showMessage("CSV 解析元件載入失敗；請確認網路連線後重新整理頁面。", "error");
    elements.fileStatus.textContent = "無法讀取 CSV";
    return;
  }
  window.Papa.parse(file, {
    header: true,
    skipEmptyLines: "greedy",
    dynamicTyping: false,
    complete: result => {
      const records = result.data.filter(row => row && Object.values(row).some(value => String(value ?? "").trim()));
      if (!records.length || !result.meta.fields?.length) {
        showMessage("CSV 沒有可用資料，請確認第一列是欄位名稱且檔案包含資料列。", "error");
        elements.fileStatus.textContent = "CSV 格式無效";
        return;
      }
      if (result.errors.length) {
        const blockingErrors = result.errors.filter(error => error.code !== "TooFewFields");
        if (blockingErrors.length) {
          const first = blockingErrors[0];
          showMessage(`CSV 解析時發現 ${blockingErrors.length} 個格式問題（第 ${first.row ?? "?"} 列：${first.message}）。請修正格式後重新上傳。`, "error");
          elements.fileStatus.textContent = "CSV 格式需要修正";
          return;
        }
        showMessage(`有 ${result.errors.length} 列欄位不完整；這些列會在分析時依所選欄位排除。`);
      }
      const normalized = normalizeRecords({ ...result, data: records });
      elements.fileStatus.textContent = `${file.name} · ${state.rows.length.toLocaleString()} 筆 · ${state.headers.length} 個欄位`;
      elements.controls.hidden = false;
      delete elements.featureCount.dataset.userChanged;
      elements.summary.textContent = `${state.rows.length.toLocaleString()} 筆資料 · ${state.headers.length} 個欄位`;
      elements.target.replaceChildren();
      normalized.numericTargets.forEach(target => {
        const option = document.createElement("option");
        option.value = target;
        option.textContent = target;
        elements.target.append(option);
      });
      if (!normalized.numericTargets.length) {
        elements.controls.hidden = true;
        showMessage("找不到連續數值目標欄位。類別文字目標不適合直接用線性回歸，請先整理成適當的連續數值目標。", "error");
        return;
      }
      renderFeatureOptions();
      elements.notes.textContent = `自動略過欄位：${normalized.skipped.length ? normalized.skipped.join("、") : "無"}。${normalized.truncated ? `資料超過 ${MAX_ROWS.toLocaleString()} 列，為維持瀏覽器效能，只分析前 ${MAX_ROWS.toLocaleString()} 列。` : ""}`;
      if (state.featureColumns.length > 39) showMessage(`偵測到 ${state.featureColumns.length} 個可用特徵；可選擇其中最多 39 個。`);
      else if (!result.errors.length) showMessage("");
      elements.run.disabled = false;
    },
    error: error => {
      showMessage(`無法讀取 CSV：${error.message}`, "error");
      elements.fileStatus.textContent = "CSV 讀取失敗";
    },
  });
});

$("#select-defaults").addEventListener("click", () => {
  const defaultNames = new Set(state.defaults);
  elements.featureList.querySelectorAll("input").forEach(input => { input.checked = defaultNames.has(input.value); });
  updateFeatureCount();
});
elements.target.addEventListener("change", renderFeatureOptions);
elements.featureCount.addEventListener("input", () => { elements.featureCount.dataset.userChanged = "true"; });
elements.run.addEventListener("click", runAnalysis);
$("#download-results").addEventListener("click", downloadPredictions);
