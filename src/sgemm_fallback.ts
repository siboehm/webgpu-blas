
export function sgemm(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array, beta: number = 0.0, c?: Float32Array): Float32Array {
  // To improve performance on WebGPU unsupported devices, use WebGL or WebAssembly
  const result = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0.0;
      for (let j = 0; j < k; j++) {
        sum += a[row * k + j] * b[j * n + col];
      }
      result[row * n + col] = alpha * sum + (c ? beta * c[row * n + col] : 0.0);
    }
  }
  return result;
}
