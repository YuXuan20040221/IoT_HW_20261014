export function seededShuffleSplit(length, testRatio, seed) {
  if (!Number.isInteger(length) || length < 3) {
    throw new Error("至少需要 3 筆有效資料才能切分訓練集與測試集。");
  }
  const indices = Array.from({ length }, (_, index) => index);
  let state = (Number(seed) >>> 0) || 1;
  for (let index = indices.length - 1; index > 0; index -= 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const other = state % (index + 1);
    [indices[index], indices[other]] = [indices[other], indices[index]];
  }
  const testCount = Math.max(1, Math.min(length - 2, Math.round(length * testRatio)));
  return {
    train: indices.slice(testCount),
    test: indices.slice(0, testCount),
  };
}

export function fitLinearRegression(rows, targets, featureNames) {
  const rowCount = rows.length;
  const featureCount = featureNames.length;
  if (featureCount === 0 || rowCount <= featureCount + 1) {
    throw new Error("有效資料筆數必須大於特徵數加 1，請減少特徵或增加資料。");
  }

  const means = Array(featureCount).fill(0);
  const scales = Array(featureCount).fill(0);
  let targetMean = 0;
  for (let row = 0; row < rowCount; row += 1) {
    targetMean += targets[row];
    for (let col = 0; col < featureCount; col += 1) means[col] += rows[row][col];
  }
  targetMean /= rowCount;
  for (let col = 0; col < featureCount; col += 1) means[col] /= rowCount;

  const matrix = Array.from({ length: rowCount }, () => Array(featureCount));
  const residualTargets = new Float64Array(rowCount);
  for (let row = 0; row < rowCount; row += 1) {
    residualTargets[row] = targets[row] - targetMean;
    for (let col = 0; col < featureCount; col += 1) {
      const delta = rows[row][col] - means[col];
      scales[col] += delta * delta;
      matrix[row][col] = delta;
    }
  }
  for (let col = 0; col < featureCount; col += 1) {
    scales[col] = Math.sqrt(scales[col] / rowCount);
    if (!Number.isFinite(scales[col]) || scales[col] <= Number.EPSILON) {
      throw new Error(`特徵「${featureNames[col]}」是常數欄位，不能用於回歸。`);
    }
    for (let row = 0; row < rowCount; row += 1) matrix[row][col] /= scales[col];
  }

  for (let pivot = 0; pivot < featureCount; pivot += 1) {
    let norm = 0;
    for (let row = pivot; row < rowCount; row += 1) norm = Math.hypot(norm, matrix[row][pivot]);
    if (!Number.isFinite(norm) || norm < 1e-10) {
      throw new Error("部分特徵完全重複或高度共線，請移除重複／高度相關欄位後再試。");
    }
    const alpha = matrix[pivot][pivot] >= 0 ? -norm : norm;
    const vector = new Float64Array(rowCount - pivot);
    vector[0] = matrix[pivot][pivot] - alpha;
    for (let row = pivot + 1; row < rowCount; row += 1) vector[row - pivot] = matrix[row][pivot];
    let vectorNormSquared = 0;
    for (const value of vector) vectorNormSquared += value * value;
    const factor = 2 / vectorNormSquared;

    for (let col = pivot; col < featureCount; col += 1) {
      let projection = 0;
      for (let row = pivot; row < rowCount; row += 1) {
        projection += vector[row - pivot] * matrix[row][col];
      }
      projection *= factor;
      for (let row = pivot; row < rowCount; row += 1) {
        matrix[row][col] -= projection * vector[row - pivot];
      }
    }
    let targetProjection = 0;
    for (let row = pivot; row < rowCount; row += 1) {
      targetProjection += vector[row - pivot] * residualTargets[row];
    }
    targetProjection *= factor;
    for (let row = pivot; row < rowCount; row += 1) {
      residualTargets[row] -= targetProjection * vector[row - pivot];
    }
    matrix[pivot][pivot] = alpha;
    for (let row = pivot + 1; row < rowCount; row += 1) matrix[row][pivot] = 0;
  }

  const standardizedCoefficients = Array(featureCount).fill(0);
  for (let row = featureCount - 1; row >= 0; row -= 1) {
    let value = residualTargets[row];
    for (let col = row + 1; col < featureCount; col += 1) {
      value -= matrix[row][col] * standardizedCoefficients[col];
    }
    standardizedCoefficients[row] = value / matrix[row][row];
  }
  const coefficients = standardizedCoefficients.map((value, index) => value / scales[index]);
  const intercept = targetMean - coefficients.reduce((sum, value, index) => sum + value * means[index], 0);
  return { intercept, coefficients, featureNames: [...featureNames] };
}

export function predict(model, row) {
  return model.intercept + model.coefficients.reduce((sum, coefficient, index) => sum + coefficient * row[index], 0);
}

export function calculateMetrics(actual, predicted) {
  const count = actual.length;
  if (!count || count !== predicted.length) throw new Error("沒有可供評估的測試資料。");
  const mean = actual.reduce((sum, value) => sum + value, 0) / count;
  let squaredError = 0;
  let absoluteError = 0;
  let totalVariation = 0;
  for (let index = 0; index < count; index += 1) {
    const error = actual[index] - predicted[index];
    squaredError += error * error;
    absoluteError += Math.abs(error);
    totalVariation += (actual[index] - mean) ** 2;
  }
  return {
    mse: squaredError / count,
    rmse: Math.sqrt(squaredError / count),
    mae: absoluteError / count,
    r2: totalVariation === 0 ? null : 1 - squaredError / totalVariation,
  };
}

export function selectForwardFeatures(rows, targets, featureNames, maxFeatures) {
  const remaining = new Set(featureNames.map((_, index) => index));
  const selected = [];
  const scores = [];
  const limit = Math.min(maxFeatures, featureNames.length);
  for (let step = 0; step < limit; step += 1) {
    let bestIndex = -1;
    let bestMse = Infinity;
    for (const candidate of remaining) {
      const trial = [...selected, candidate];
      try {
        const names = trial.map(index => featureNames[index]);
        const model = fitLinearRegression(rows.map(row => trial.map(index => row[index])), targets, names);
        const mse = targets.reduce((sum, target, index) => {
          const prediction = predict(model, trial.map(featureIndex => rows[index][featureIndex]));
          return sum + (target - prediction) ** 2;
        }, 0) / targets.length;
        if (mse < bestMse) {
          bestMse = mse;
          bestIndex = candidate;
        }
      } catch (error) {
        if (!(error instanceof Error) || !/完全重複|高度共線/.test(error.message)) throw error;
      }
    }
    if (bestIndex < 0) break;
    selected.push(bestIndex);
    remaining.delete(bestIndex);
    scores.push({ count: selected.length, mse: bestMse, features: selected.map(index => featureNames[index]) });
  }
  if (!scores.length) throw new Error("無法從目前資料選出有效特徵；請檢查常數欄位或重複資料。");
  return { selectedIndices: selected, scores };
}
