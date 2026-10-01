// Webcam gaze for the focus mask (GAZE.md). The math is pure and unit-tested in tests/gaze.test.mjs;
// only startFaceTracking touches MediaPipe.

// MediaPipe face-mesh indices: eye corners and iris centers (the subject's right eye appears on the image left).
const RIGHT_EYE = { outer: 33, inner: 133, iris: 468 };
const LEFT_EYE = { outer: 263, inner: 362, iris: 473 };
const NOSE_TIP = 1;

function eyeOffset(points, eye) {
  // Iris position in the eye's own frame, normalized by eye width: robust to distance from the camera.
  const a = points[eye.outer], b = points[eye.inner], iris = points[eye.iris];
  const ax = b.x - a.x, ay = b.y - a.y, width = Math.hypot(ax, ay);
  const dx = iris.x - (a.x + b.x) / 2, dy = iris.y - (a.y + b.y) / 2;
  return [(dx * ax + dy * ay) / (width * width), (dy * ax - dx * ay) / (width * width)];
}

// Feature vector: both irises in eye coordinates, head yaw and pitch, and where the face sits in the frame.
export function gazeFeatures(points, matrix) {
  const d = matrix; // 4x4 column-major facial transformation
  const yaw = Math.atan2(d[8], d[10]), pitch = Math.asin(Math.max(-1, Math.min(1, -d[9])));
  return [...eyeOffset(points, RIGHT_EYE), ...eyeOffset(points, LEFT_EYE), yaw, pitch, points[NOSE_TIP].x, points[NOSE_TIP].y];
}

// Ridge regression on standardized, second-order polynomial features: features -> (u, v) on the output image.
export class GazeModel {
  fit(features, targets, lambda = 1e-2) {
    const n = features[0].length;
    this.mean = Array.from({ length: n }, (_, j) => features.reduce((s, f) => s + f[j], 0) / features.length);
    // Floor the spread: a feature that barely moved during calibration (a very still head) must not amplify jitter.
    this.std = this.mean.map((m, j) => Math.max(0.01, Math.sqrt(features.reduce((s, f) => s + (f[j] - m) ** 2, 0) / features.length)));
    const X = features.map((f) => this.expand(f));
    const k = X[0].length;
    const A = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => X.reduce((s, x) => s + x[i] * x[j], 0) + (i === j ? lambda : 0)));
    const B = Array.from({ length: k }, (_, i) => [0, 1].map((c) => X.reduce((s, x, r) => s + x[i] * targets[r][c], 0)));
    this.weights = solve(A, B);
    return this;
  }
  expand(f) {
    const z = f.map((x, j) => (x - this.mean[j]) / this.std[j]);
    const out = [1, ...z];
    for (let i = 0; i < z.length; i++) for (let j = i; j < z.length; j++) out.push(z[i] * z[j]);
    return out;
  }
  predict(f) {
    const x = this.expand(f);
    return [0, 1].map((c) => x.reduce((s, xi, i) => s + xi * this.weights[i][c], 0));
  }
}

function solve(A, B) {
  // Gauss-Jordan with partial pivoting; A is k x k (symmetric positive definite), B is k x 2.
  const k = A.length, M = A.map((row, i) => [...row, ...B[i]]);
  for (let c = 0; c < k; c++) {
    let p = c;
    for (let r = c + 1; r < k; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < k; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let j = c; j < k + 2; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row, i) => [row[k] / row[i], row[k + 1] / row[i]]);
}

// One-Euro filter: smooths jitter when the gaze is still, follows quickly when it moves.
export class OneEuro {
  constructor(minCutoff = 1.0, beta = 0.5, dCutoff = 1.0) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: 0, t: null }); }
  static alpha(cutoff, dt) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  filter(value, t) {
    if (this.x === null) { this.x = value; this.t = t; return value; }
    const dt = Math.max(t - this.t, 1e-3);
    const dx = (value - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (dx - this.dx);
    this.x += OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (value - this.x);
    this.t = t;
    return this.x;
  }
}

// Velocity-threshold fixation detection (I-VT): fixated once the gaze has moved slower than `maxSpeed`
// (image widths per second) for `minDuration` seconds.
export class Fixation {
  constructor(maxSpeed = 0.5, minDuration = 0.15) { Object.assign(this, { maxSpeed, minDuration, last: null, since: null }); }
  update([u, v], t) {
    const speed = this.last ? Math.hypot(u - this.last[0], v - this.last[1]) / Math.max(t - this.last[2], 1e-3) : Infinity;
    this.last = [u, v, t];
    if (speed > this.maxSpeed) this.since = null;
    else if (this.since === null) this.since = t;
    return this.since !== null && t - this.since >= this.minDuration;
  }
}

// Smooth-pursuit calibration target: a Lissajous path over the image (the dot the user follows), in [0.1, 0.9]^2.
export function pursuitTarget(t) {
  return [0.5 + 0.4 * Math.sin((2 * Math.PI * t) / 5), 0.5 + 0.4 * Math.sin((2 * Math.PI * t) / 3.3 + Math.PI / 2)];
}

// MediaPipe FaceLandmarker on the camera <video>; calls onFace(features, timeSeconds) per video frame with a face.
export async function startFaceTracking(video, onFace) {
  const base = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1";
  const { FaceLandmarker, FilesetResolver } = await import(`${base}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${base}/wasm`);
  const landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
    runningMode: "VIDEO", numFaces: 1, outputFacialTransformationMatrixes: true,
  });
  const onFrame = (now) => {
    const result = landmarker.detectForVideo(video, now);
    if (result.faceLandmarks.length && result.facialTransformationMatrixes.length) {
      onFace(gazeFeatures(result.faceLandmarks[0], result.facialTransformationMatrixes[0].data), now / 1000);
    }
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
  return landmarker;
}
