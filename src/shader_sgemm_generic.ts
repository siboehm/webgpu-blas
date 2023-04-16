export const Shader = `
@group(0) @binding(0)
var<storage,read> array_a: array<f32>;

@group(0) @binding(1)
var<storage,read> array_b: array<f32>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<f32>;

struct CMeta {
  M: f32,
  N: f32,
  K: f32,
  alpha: f32,
}

@group(0) @binding(3)
var<storage,read> cmeta: CMeta;

const CHUNK_SIZE: u32 = 8u;

var<workgroup> As: array<f32, CHUNK_SIZE * CHUNK_SIZE>;
var<workgroup> Bs: array<f32, CHUNK_SIZE * CHUNK_SIZE>;

const WORKGROUP_SIZE: u32 = CHUNK_SIZE;

@compute @workgroup_size(WORKGROUP_SIZE,WORKGROUP_SIZE,1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) block_id: vec3<u32>
) {
  var M: u32 = u32(cmeta.M);
  var N: u32 = u32(cmeta.N);
  var K: u32 = u32(cmeta.K);

  var threadCol: u32 = local_id.x;
  var threadRow: u32 = local_id.y;

  var col: u32 = block_id.x * WORKGROUP_SIZE + threadCol;
  var row: u32 = block_id.y * WORKGROUP_SIZE + threadRow;

  // register accumulate
  var sum: f32 = 0.0;
  for(var k: u32 = 0u; k < K; k = k + CHUNK_SIZE) {
    let aCol = k + threadCol;
    let bRow = k + threadRow;
    if (row < M && aCol < K){
      As[threadRow * CHUNK_SIZE + threadCol] = array_a[row * K + aCol];
    } else {
      As[threadRow * CHUNK_SIZE + threadCol] = 0.0;
    }
    if (bRow < K && col < N) {
      Bs[threadRow * CHUNK_SIZE + threadCol] = array_b[bRow * N + col];
    } else {
      Bs[threadRow * CHUNK_SIZE + threadCol] = 0.0;
    }

    workgroupBarrier();

    for(var i: u32 = 0u; i < CHUNK_SIZE; i = i + 1) {
      sum = fma(As[threadRow * CHUNK_SIZE + i], Bs[i * CHUNK_SIZE + threadCol], sum);
    }
    workgroupBarrier();
  }
  if (row < M && col < N) {
    array_c[row * N + col] = fma(cmeta.alpha, sum, array_c[row * N + col]);
  }
}
`;
