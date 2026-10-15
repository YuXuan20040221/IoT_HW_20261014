import test from "node:test";
import assert from "node:assert/strict";
import {
  calculateMetrics,
  fitLinearRegression,
  predict,
  seededShuffleSplit,
  selectForwardFeatures,
} from "../regression.mjs";

test("fits a multiple linear model and predicts accurately", () => {
  const rows = Array.from({ length: 30 }, (_, index) => [index, (index * 7) % 11]);
  const targets = rows.map(([first, second]) => 12 + 3 * first - 2 * second);
  const model = fitLinearRegression(rows, targets, ["first", "second"]);
  assert.ok(Math.abs(model.intercept - 12) < 1e-9);
  assert.ok(Math.abs(model.coefficients[0] - 3) < 1e-9);
  assert.ok(Math.abs(model.coefficients[1] + 2) < 1e-9);
  assert.ok(Math.abs(predict(model, [9, 4]) - 31) < 1e-9);
});

test("splits deterministically without losing rows", () => {
  const first = seededShuffleSplit(100, 0.2, 42);
  const second = seededShuffleSplit(100, 0.2, 42);
  assert.deepEqual(first, second);
  assert.equal(first.train.length, 80);
  assert.equal(first.test.length, 20);
  assert.equal(new Set([...first.train, ...first.test]).size, 100);
});

test("calculates common regression evaluation metrics", () => {
  const metrics = calculateMetrics([1, 2, 3], [1, 3, 2]);
  assert.equal(metrics.mae, 2 / 3);
  assert.equal(metrics.mse, 2 / 3);
  assert.equal(metrics.rmse, Math.sqrt(2 / 3));
  assert.equal(metrics.r2, 0);
});

test("forward selection ranks the predictive feature before noise", () => {
  const rows = Array.from({ length: 80 }, (_, index) => [index, Math.sin(index * 17), Math.cos(index * 13)]);
  const targets = rows.map(([signal]) => 5 + 4 * signal);
  const selection = selectForwardFeatures(rows, targets, ["signal", "noise_a", "noise_b"], 2);
  assert.equal(selection.scores.length, 2);
  assert.equal(selection.scores[0].features[0], "signal");
});

test("rejects an insufficient sample for a fitted model", () => {
  assert.throws(() => fitLinearRegression([[1], [2]], [1, 2], ["x"]), /有效資料筆數/);
});
