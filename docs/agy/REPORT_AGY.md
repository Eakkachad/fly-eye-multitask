# Fly-Eye Multi-Task Deep Learning Baselines Report

This report summarizes the implementation, validation, and benchmarking of hexagonal deep-learning baseline models for the fly-eye multi-task video project.

---

## 1. Files Created

All work was performed strictly inside `/home/user/flyproj/course/baselines/`:

- [hex_models.py](file:///home/user/flyproj/course/baselines/hex_models.py): Hexagonal convolutional primitives and networks:
  - `HexConv(in_ch, out_ch, bias=True, extent=15)`: Convolution on the 7-neighbourhood hex lattice with zero-padding border boundary handling.
  - `HexConvGRUCell(in_ch, hid_ch, extent=15)`: Recurrent cell performing ConvGRU gating and state transition using `HexConv`.
  - `HexConvGRUNet(hid_ch, n_layers, head_ch, in_ch=1, extent=15)`: Complete multi-task network (HexConv encoder stack $\rightarrow$ temporal ConvGRU $\rightarrow$ optic flow and depth heads).
  - `make_small()` and `make_large()`: Standardized factories producing models within the required parameter ranges.
  - `get_neighbour_table(extent=15)`: Precomputed/cached $(N, 7)$ neighbour lookup table.
- [metrics.py](file:///home/user/flyproj/course/baselines/metrics.py): Pure PyTorch evaluation metrics supporting shape `(B, T, C, N)` and optional boolean mask `(B, T, 1, N)`:
  - `epe(pred_flow, gt_flow, mask=None)`: Mean end-point error ($L_2$ norm over 2 flow channels).
  - `angular_error_deg(pred_flow, gt_flow, mask=None)`: Mean angular error between $(u, v, 1)$ vectors in degrees.
  - `depth_rmse(pred, gt, mask=None)`: Root mean squared depth error.
  - `depth_absrel(pred, gt, mask=None, eps=1e-6)`: Absolute relative depth error.
  - `binned(metric_values_per_pixel, bin_values, edges)`: Error analysis conditioning metrics across speed or texture bins.
- [test_baselines.py](file:///home/user/flyproj/course/baselines/test_baselines.py): Comprehensive test suite covering parameter bounds, forward output shapes, gradient flow, hand-computed metric verification, and neighbour table correctness.
- [bench.py](file:///home/user/flyproj/course/baselines/bench.py): Timing and memory benchmark running forward + backward passes for $B=4, T=19$ with Adam on CUDA.
- [REPORT_AGY.md](file:///home/user/flyproj/course/baselines/REPORT_AGY.md): Technical report and performance summary.

---

## 2. Parameter Counts

Target parameter budgets:
- **Small model (`make_small`)**: Target range 3,000 to 6,000 parameters (parameter-matched to connectome model M1).
- **Large model (`make_large`)**: Target range 400,000 to 1,000,000 parameters (deep baseline M5).

### Achieved Parameter Counts

| Model | Parameters | Target Range | Status | Configuration |
|---|---|---|---|---|
| `make_small()` | **4,315** | [3,000, 6,000] | **PASS** | `hid_ch=8, n_layers=2, head_ch=8` |
| `make_large()` | **604,835** | [400,000, 1,000,000] | **PASS** | `hid_ch=96, n_layers=3, head_ch=64` |

### Parameter Breakdown

#### `make_small()` (Total: 4,315)
- **Encoder**: 2 layers
  - Layer 1 (`HexConv(1, 8)`): $8 \times 1 \times 7 + 8 = 64$
  - Layer 2 (`HexConv(8, 8)`): $8 \times 8 \times 7 + 8 = 456$
- **ConvGRU Cell**:
  - Reset & Update Gates (`HexConv(16, 16)`): $16 \times 16 \times 7 + 16 = 1,808$
  - Candidate State (`HexConv(16, 8)`): $8 \times 16 \times 7 + 8 = 904$
- **Decoders / Heads**:
  - Flow Head (`HexConv(8, 8) -> HexConv(8, 2)`): $(8 \times 8 \times 7 + 8) + (2 \times 8 \times 7 + 2) = 456 + 114 = 570$
  - Depth Head (`HexConv(8, 8) -> HexConv(8, 1)`): $(8 \times 8 \times 7 + 8) + (1 \times 8 \times 7 + 1) = 456 + 57 = 513$

#### `make_large()` (Total: 604,835)
- **Encoder**: 3 layers
  - Layer 1 (`HexConv(1, 96)`): $96 \times 1 \times 7 + 96 = 768$
  - Layer 2 (`HexConv(96, 96)`): $96 \times 96 \times 7 + 96 = 64,608$
  - Layer 3 (`HexConv(96, 96)`): $96 \times 96 \times 7 + 96 = 64,608$
- **ConvGRU Cell**:
  - Reset & Update Gates (`HexConv(192, 192)`): $192 \times 192 \times 7 + 192 = 258,240$
  - Candidate State (`HexConv(192, 96)`): $96 \times 192 \times 7 + 96 = 129,120$
- **Decoders / Heads**:
  - Flow Head (`HexConv(96, 64) -> HexConv(64, 2)`): $43,072 + 898 = 43,970$
  - Depth Head (`HexConv(96, 64) -> HexConv(64, 1)`): $43,072 + 449 = 43,521$

---

## 3. Test Suite Verification

Pytest was executed via `/home/user/flyproj/.venv/bin/python -m pytest -v test_baselines.py`.

```text
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0 -- /home/user/flyproj/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/user/flyproj/course/baselines
plugins: jaxtyping-0.3.7, hydra-core-1.3.7
collected 16 items

test_baselines.py::test_param_counts PASSED                              [  6%]
test_baselines.py::test_neighbour_table_correctness PASSED               [ 12%]
test_baselines.py::test_hexconv_border_zero_padding PASSED               [ 18%]
test_baselines.py::test_forward_shapes[cpu-make_small] PASSED            [ 25%]
test_baselines.py::test_forward_shapes[cpu-make_large] PASSED            [ 31%]
test_baselines.py::test_forward_shapes[cuda-make_small] PASSED           [ 37%]
test_baselines.py::test_forward_shapes[cuda-make_large] PASSED           [ 43%]
test_baselines.py::test_gradient_flow[cpu-make_small] PASSED             [ 50%]
test_baselines.py::test_gradient_flow[cpu-make_large] PASSED             [ 56%]
test_baselines.py::test_gradient_flow[cuda-make_small] PASSED            [ 62%]
test_baselines.py::test_gradient_flow[cuda-make_large] PASSED            [ 68%]
test_baselines.py::test_metric_epe PASSED                                [ 75%]
test_baselines.py::test_metric_angular_error_deg PASSED                  [ 81%]
test_baselines.py::test_metric_depth_rmse PASSED                         [ 87%]
test_baselines.py::test_metric_depth_absrel PASSED                       [ 93%]
test_baselines.py::test_metric_binned PASSED                             [100%]

============================== 16 passed in 4.56s ==============================
```

All 16 tests passed.

---

## 4. Benchmark Numbers

Benchmark environment:
- GPU: NVIDIA GeForce RTX 4060 (Compute Capability 8.9)
- PyTorch: 2.14.1+cu130
- Workload: Forward + Backward pass, Adam optimizer, $B=4, T=19, N=721$ hexals.

| Model | Trainable Params | Time per Iteration (`s/iter`) | Peak VRAM (MB) | Peak VRAM (GB) |
|---|---|---|---|---|
| `small` (`make_small`) | 4,315 | **0.0401 s** | **160.0 MB** | **0.156 GB** |
| `large` (`make_large`) | 604,835 | **0.3388 s** | **1667.8 MB** | **1.629 GB** |

Key observation: `make_small` runs at ~25 iterations per second with minimal VRAM overhead (~160 MB), allowing multiple concurrent runs on a single RTX 4060 GPU as planned in `PLAN.md`.

---

## 5. Architectural Implementation Notes & Limitations

### Implementation Highlights
1. **Zero-Padding Hexagonal Gather**: The lattice consists of $N=721$ hexals (radius 15). At each convolution step, an extra row of zeros is appended at index $N$. Boundary hexals pointing outside the grid gather from index $N$, naturally yielding zero-padding with no runtime branching.
2. **Fused Linear Operations**: By reshaping gathered 7-neighbourhood tensors `(B, N, C * 7)`, convolution executes as an optimized GEMM (`F.linear`), achieving high GPU utilization (>18x faster than naive einsum).
3. **Multi-Task Decoders**: The ConvGRU hidden state sequence `(B, T, hid_ch, N)` is projected to flow `(B, T, 2, N)` and depth `(B, T, 1, N)` at each time frame.

### Limitations
1. **Fixed Hexagonal Kernel Radius**: Current implementation restricts kernel neighbourhood to 1-ring axial neighbours ($k=7$). Dilated hexagonal convolutions or multi-scale hexagonal pooling (downsampling/upsampling) are not included.
2. **Sequential Temporal Recurrence**: While spatial convolutions across frames are parallelized across batch and time in the encoder and decoders, the recurrent GRU state transitions inherently require a sequential Python loop across time frames $t=0 \dots T-1$.
3. **Fixed Hexagonal Lattice Extent**: The lattice extent defaults to $R=15$ ($N=721$) matching the flyvis camera array; changing extent re-indexes neighbours via `get_neighbour_table(extent)`.
