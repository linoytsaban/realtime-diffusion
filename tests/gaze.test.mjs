// Pure gaze math (static/gaze.js). Run: node --test tests/
import test from "node:test";
import assert from "node:assert/strict";
import { gazeFeatures, GazeModel, OneEuro, Fixation, pursuitTarget } from "../static/gaze.js";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function face({ irisShift = 0, scale = 1 }) {
  // Two eyes 0.1 * scale wide; irises shifted along the eye axis by `irisShift` eye widths.
  const p = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
  const w = 0.1 * scale;
  Object.assign(p, {
    33: { x: 0.3, y: 0.4 }, 133: { x: 0.3 + w, y: 0.4 }, 468: { x: 0.3 + w / 2 + irisShift * w, y: 0.4 },
    263: { x: 0.7, y: 0.4 }, 362: { x: 0.7 - w, y: 0.4 }, 473: { x: 0.7 - w / 2 - irisShift * w, y: 0.4 },
  });
  return p;
}

test("eye features are centered, signed and scale invariant", () => {
  assert.deepEqual(gazeFeatures(face({}), IDENTITY).slice(0, 4).map((x) => Math.abs(x)), [0, 0, 0, 0]);
  const near = gazeFeatures(face({ irisShift: 0.2 }), IDENTITY), far = gazeFeatures(face({ irisShift: 0.2, scale: 0.5 }), IDENTITY);
  assert.ok(Math.abs(near[0] - 0.2) < 1e-9, "offset is in eye widths");
  assert.ok(Math.abs(near[0] - far[0]) < 1e-9, "same offset when the face is further away");
  assert.deepEqual(near.slice(4, 6), [0, -0], "identity pose: no yaw, no pitch");
});

test("calibration recovers a nonlinear gaze mapping", () => {
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  const truth = (f) => [0.5 + 0.3 * f[0] + 0.1 * f[4] + 0.05 * f[0] * f[1], 0.5 + 0.3 * f[1] - 0.1 * f[5] + 0.05 * f[1] ** 2];
  const sample = () => Array.from({ length: 8 }, rand);
  const train = Array.from({ length: 300 }, sample);
  const model = new GazeModel().fit(train, train.map((f) => truth(f).map((x) => x + 0.01 * rand())));
  const errors = Array.from({ length: 100 }, sample).map((f) => Math.hypot(...model.predict(f).map((x, i) => x - truth(f)[i])));
  const mean = errors.reduce((a, b) => a + b) / errors.length;
  assert.ok(mean < 0.02, `mean error ${mean.toFixed(4)} image widths`);
});

test("One-Euro filter cuts jitter at rest and follows a real move", () => {
  const filter = new OneEuro();
  let spread = 0;
  for (let i = 0; i < 60; i++) {
    const y = filter.filter(0.5 + (i % 2 ? 0.02 : -0.02), i / 30);
    if (i > 30) spread = Math.max(spread, Math.abs(y - 0.5));
  }
  assert.ok(spread < 0.01, `jitter ${spread.toFixed(4)} vs 0.02 raw`);
  let y;
  for (let i = 60; i < 75; i++) y = filter.filter(0.9, i / 30);
  assert.ok(y > 0.85, `followed a jump to 0.9 within 0.5 s (at ${y.toFixed(3)})`);
});

test("fixation needs a still gaze for a while and ends on a saccade", () => {
  const fix = new Fixation(0.5, 0.15);
  const states = [];
  for (let i = 0; i < 10; i++) states.push(fix.update([0.5, 0.5], i / 30));
  assert.equal(states[2], false, "not yet: too short");
  assert.equal(states.at(-1), true, "fixated after 0.3 s");
  assert.equal(fix.update([0.9, 0.5], 10 / 30), false, "a jump ends the fixation");
});

test("pursuit target stays inside the image", () => {
  for (let t = 0; t < 20; t += 0.05) for (const x of pursuitTarget(t)) assert.ok(x >= 0.1 - 1e-9 && x <= 0.9 + 1e-9);
});
