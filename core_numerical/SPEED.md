# Core Numerical — speed comparisons

Generated from the Python ↔ Rust parity harnesses (`*_parity.compare`).

**Settings:** `size=64`, `iters=15`, `warmup=5`, `seed=42`, release `parity_runner` (+ `torch_micro_runner` for short kernels).

**Speedup** = Python median ns / Rust median ns (>1 means Rust is faster).

Raw JSON: [`results/`](results/).

## NumPy / rnumpy

80 ops · parity OK 79 · FAIL 1 · Rust faster on 60

| Op | Parity | Python | Rust | Speedup |
|----|--------|--------|------|---------|
| `zeros` | OK | 1.90 µs | 900 ns | 2.11× |
| `ones` | OK | 4.30 µs | 1.10 µs | 3.91× |
| `full` | OK | 3.20 µs | 1.10 µs | 2.91× |
| `arange` | OK | 1.10 µs | 1.10 µs | 1.00× |
| `linspace` | OK | 9.40 µs | 400 ns | 23.50× |
| `eye` | OK | 5.00 µs | 1.50 µs | 3.33× |
| `add` | OK | 3.00 µs | 2.50 µs | 1.20× |
| `add_broadcast` | OK | 10.20 µs | 2.20 µs | 4.64× |
| `subtract` | OK | 2.90 µs | 2.50 µs | 1.16× |
| `multiply` | OK | 5.60 µs | 2.40 µs | 2.33× |
| `divide` | OK | 3.30 µs | 3.10 µs | 1.06× |
| `power` | OK | 2.40 µs | 2.50 µs | 0.96× |
| `maximum` | OK | 3.20 µs | 2.80 µs | 1.14× |
| `minimum` | OK | 3.40 µs | 3.40 µs | 1.00× |
| `greater` | OK | 7.80 µs | 2.50 µs | 3.12× |
| `less` | OK | 8.30 µs | 3.40 µs | 2.44× |
| `equal` | OK | 4.60 µs | 1.90 µs | 2.42× |
| `not_equal` | OK | 30.70 µs | 7.30 µs | 4.21× |
| `sqrt` | OK | 17.50 µs | 10.80 µs | 1.62× |
| `exp` | OK | 3.90 µs | 2.70 µs | 1.44× |
| `log` | OK | 4.20 µs | 3.20 µs | 1.31× |
| `sin` | OK | 70.50 µs | 70.60 µs | 1.00× |
| `cos` | OK | 80.90 µs | 37.60 µs | 2.15× |
| `tan` | OK | 5.60 µs | 3.80 µs | 1.47× |
| `tanh` | OK | 210.30 µs | 105.10 µs | 2.00× |
| `negative` | OK | 4.70 µs | 5.50 µs | 0.85× |
| `abs` | OK | 4.50 µs | 2.90 µs | 1.55× |
| `sign` | OK | 10.00 µs | 3.00 µs | 3.33× |
| `square` | OK | 5.80 µs | 4.70 µs | 1.23× |
| `reciprocal` | OK | 9.20 µs | 4.60 µs | 2.00× |
| `floor` | OK | 3.90 µs | 2.60 µs | 1.50× |
| `ceil` | OK | 3.30 µs | 2.40 µs | 1.38× |
| `trunc` | OK | 3.10 µs | 2.40 µs | 1.29× |
| `round` | OK | 7.80 µs | 2.20 µs | 3.55× |
| `clip` | OK | 4.80 µs | 1.70 µs | 2.82× |
| `where` | OK | 10.60 µs | 5.60 µs | 1.89× |
| `sum` | OK | 5.30 µs | 1.30 µs | 4.08× |
| `sum_axis` | OK | 5.70 µs | 1.10 µs | 5.18× |
| `mean` | OK | 7.30 µs | 1.30 µs | 5.62× |
| `mean_axis` | OK | 9.40 µs | 2.30 µs | 4.09× |
| `min` | OK | 4.80 µs | 1.70 µs | 2.82× |
| `min_axis` | OK | 6.50 µs | 3.70 µs | 1.76× |
| `max` | OK | 4.80 µs | 1.80 µs | 2.67× |
| `max_axis` | OK | 13.50 µs | 2.30 µs | 5.87× |
| `var` | OK | 24.70 µs | 6.60 µs | 3.74× |
| `std` | OK | 26.30 µs | 6.50 µs | 4.05× |
| `argmin` | OK | 3.80 µs | 2.70 µs | 1.41× |
| `argmax` | OK | 3.00 µs | 3.40 µs | 0.88× |
| `cumsum` | OK | 23.50 µs | 20.00 µs | 1.18× |
| `cumsum_axis` | OK | 23.10 µs | 27.60 µs | 0.84× |
| `cumprod` | OK | 3.20 µs | 700 ns | 4.57× |
| `transpose` | OK | 800 ns | 200 ns | 4.00× |
| `reshape` | OK | 1.40 µs | 300 ns | 4.67× |
| `reshape_infer` | OK | 1.50 µs | 300 ns | 5.00× |
| `ravel` | OK | 700 ns | 300 ns | 2.33× |
| `concatenate` | OK | 4.60 µs | 14.50 µs | 0.32× |
| `stack` | OK | 7.30 µs | 18.90 µs | 0.39× |
| `broadcast_to` | OK | 7.00 µs | 2.70 µs | 2.59× |
| `swapaxes` | OK | 800 ns | 100 ns | 8.00× |
| `moveaxis` | OK | 4.50 µs | 300 ns | 15.00× |
| `matmul` | OK | 22.10 µs | 20.20 µs | 1.09× |
| `dot` | OK | 1.70 µs | 600 ns | 2.83× |
| `trace` | OK | 3.60 µs | 300 ns | 12.00× |
| `norm` | OK | 3.50 µs | 4.80 µs | 0.73× |
| `solve` | OK | 46.20 µs | 40.20 µs | 1.15× |
| `inv` | OK | 84.40 µs | 320.20 µs | 0.26× |
| `det` | OK | 35.70 µs | 44.80 µs | 0.80× |
| `qr` | OK | 90.30 µs | 305.70 µs | 0.30× |
| `svd` | OK | 303.80 µs | 1.80 ms | 0.17× |
| `svdvals` | OK | 122.00 µs | 1.41 ms | 0.09× |
| `eigvalsh` | OK | 295.40 µs | 17.58 ms | 0.02× |
| `eigvals` | OK | 158.40 µs | 28.02 ms | 0.01× |
| `eig` | FAIL | 72.20 µs | 3.49 ms | 0.02× |
| `take` | OK | 3.00 µs | 600 ns | 5.00× |
| `compress` | OK | 2.50 µs | 800 ns | 3.12× |
| `boolean_index` | OK | 12.80 µs | 8.10 µs | 1.58× |
| `fancy_index_2d` | OK | 2.00 µs | 700 ns | 2.86× |
| `take_along_axis` | OK | 26.50 µs | 227.40 µs | 0.12× |
| `slice` | OK | 2.50 µs | 100 ns | 25.00× |
| `astype_f32` | OK | 5.60 µs | 29.00 µs | 0.19× |

### Fastest Rust wins (top 10)

| Op | Speedup |
|----|---------|
| `slice` | 25.00× |
| `linspace` | 23.50× |
| `moveaxis` | 15.00× |
| `trace` | 12.00× |
| `swapaxes` | 8.00× |
| `max_axis` | 5.87× |
| `mean` | 5.62× |
| `sum_axis` | 5.18× |
| `reshape_infer` | 5.00× |
| `take` | 5.00× |

## SciPy / rscipy

66 ops · parity OK 66 · Rust faster on 62

| Op | Parity | Python | Rust | Speedup |
|----|--------|--------|------|---------|
| `erf` | OK | 2.90 µs | 1.10 µs | 2.64× |
| `erfc` | OK | 2.90 µs | 1.20 µs | 2.42× |
| `gamma` | OK | 1.60 µs | 2.80 µs | 0.57× |
| `gammaln` | OK | 5.70 µs | 2.20 µs | 2.59× |
| `expit` | OK | 6.00 µs | 1.10 µs | 5.45× |
| `logit` | OK | 4.60 µs | 1.20 µs | 3.83× |
| `logsumexp` | OK | 69.40 µs | 700 ns | 99.14× |
| `softmax` | OK | 14.90 µs | 1.60 µs | 9.31× |
| `i0` | OK | 7.40 µs | 1.10 µs | 6.73× |
| `ndtr` | OK | 3.30 µs | 1.20 µs | 2.75× |
| `ndtri` | OK | 4.20 µs | 1.00 µs | 4.20× |
| `lu` | OK | 98.50 µs | 156.80 µs | 0.63× |
| `lu_factor` | OK | 54.00 µs | 47.40 µs | 1.14× |
| `cholesky` | OK | 27.70 µs | 50.70 µs | 0.55× |
| `solve_triangular` | OK | 18.70 µs | 3.40 µs | 5.50× |
| `lstsq` | OK | 2.59 ms | 2.44 ms | 1.07× |
| `norm` | OK | 11.00 µs | 5.20 µs | 2.12× |
| `norm_1` | OK | 10.70 µs | 3.50 µs | 3.06× |
| `norm_inf` | OK | 11.60 µs | 3.30 µs | 3.52× |
| `expm` | OK | 96.70 µs | 14.30 µs | 6.76× |
| `nelder_mead` | OK | 4.50 ms | 27.60 µs | 162.93× |
| `lbfgsb` | OK | 5.07 ms | 246.10 µs | 20.61× |
| `least_squares` | OK | 689.10 µs | 1.40 µs | 492.21× |
| `norm_pdf` | OK | 77.00 µs | 1.00 µs | 77.00× |
| `norm_cdf` | OK | 63.20 µs | 1.20 µs | 52.67× |
| `norm_ppf` | OK | 102.90 µs | 900 ns | 114.33× |
| `entropy` | OK | 256.80 µs | 1.00 µs | 256.80× |
| `zscore` | OK | 232.10 µs | 1.00 µs | 232.10× |
| `rankdata` | OK | 112.50 µs | 2.70 µs | 41.67× |
| `pearsonr` | OK | 351.80 µs | 1.00 µs | 351.80× |
| `spearmanr` | OK | 456.30 µs | 6.10 µs | 74.80× |
| `ttest_ind` | OK | 1.96 ms | 1.20 µs | 1629.42× |
| `skew` | OK | 541.70 µs | 500 ns | 1083.40× |
| `kurtosis` | OK | 509.90 µs | 400 ns | 1274.75× |
| `sem` | OK | 399.30 µs | 600 ns | 665.50× |
| `csr_from_dense` | OK | 199.20 µs | 17.90 µs | 11.13× |
| `csr_matvec` | OK | 7.30 µs | 3.00 µs | 2.43× |
| `csr_matmat` | OK | 17.40 µs | 8.00 µs | 2.17× |
| `csr_transpose` | OK | 25.50 µs | 100 ns | 255.00× |
| `csr_add` | OK | 65.90 µs | 86.80 µs | 0.76× |
| `csr_eye` | OK | 41.00 µs | 1.00 µs | 41.00× |
| `csr_norm` | OK | 4.70 µs | 2.80 µs | 1.68× |
| `csr_to_csc` | OK | 54.50 µs | 36.00 µs | 1.51× |
| `spsolve` | OK | 403.30 µs | 122.10 µs | 3.30× |
| `cg` | OK | 332.00 µs | 50.00 µs | 6.64× |
| `butter` | OK | 195.80 µs | 1.60 µs | 122.38× |
| `filtfilt` | OK | 133.50 µs | 8.70 µs | 15.34× |
| `welch` | OK | 190.00 µs | 24.40 µs | 7.79× |
| `stft` | OK | 152.90 µs | 22.90 µs | 6.68× |
| `fft` | OK | 8.50 µs | 1.90 µs | 4.47× |
| `ifft` | OK | 8.40 µs | 2.00 µs | 4.20× |
| `rfft` | OK | 8.90 µs | 2.50 µs | 3.56× |
| `irfft` | OK | 11.90 µs | 2.10 µs | 5.67× |
| `fftfreq` | OK | 6.80 µs | 500 ns | 13.60× |
| `convolve` | OK | 25.00 µs | 2.10 µs | 11.90× |
| `fftconvolve` | OK | 87.60 µs | 8.70 µs | 10.07× |
| `correlate` | OK | 31.60 µs | 2.70 µs | 11.70× |
| `hann` | OK | 21.30 µs | 1.10 µs | 19.36× |
| `hamming` | OK | 19.80 µs | 1.10 µs | 18.00× |
| `blackman` | OK | 28.60 µs | 1.60 µs | 17.88× |
| `detrend` | OK | 128.40 µs | 900 ns | 142.67× |
| `trapezoid` | OK | 20.90 µs | 300 ns | 69.67× |
| `simpson` | OK | 40.00 µs | 400 ns | 100.00× |
| `cumulative_trapezoid` | OK | 13.10 µs | 900 ns | 14.56× |
| `quad` | OK | 12.90 µs | 3.00 µs | 4.30× |
| `solve_ivp` | OK | 1.77 ms | 14.30 µs | 123.55× |

### Fastest Rust wins (top 10)

| Op | Speedup |
|----|---------|
| `ttest_ind` | 1629.42× |
| `kurtosis` | 1274.75× |
| `skew` | 1083.40× |
| `sem` | 665.50× |
| `least_squares` | 492.21× |
| `pearsonr` | 351.80× |
| `entropy` | 256.80× |
| `csr_transpose` | 255.00× |
| `zscore` | 232.10× |
| `nelder_mead` | 162.93× |

## Pandas / rpandas

18 ops · parity OK 18 · Rust faster on 18

| Op | Parity | Python | Rust | Speedup |
|----|--------|--------|------|---------|
| `construct` | OK | 258.60 µs | 4.10 µs | 63.07× |
| `select` | OK | 320.30 µs | 600 ns | 533.83× |
| `head` | OK | 14.40 µs | 2.50 µs | 5.76× |
| `filter_gt` | OK | 208.10 µs | 2.90 µs | 71.76× |
| `sort_values` | OK | 162.10 µs | 4.20 µs | 38.60× |
| `dropna` | OK | 408.30 µs | 6.20 µs | 65.85× |
| `fillna` | OK | 307.40 µs | 2.90 µs | 106.00× |
| `sum` | OK | 456.40 µs | 4.40 µs | 103.73× |
| `mean` | OK | 381.00 µs | 3.20 µs | 119.06× |
| `describe` | OK | 3.76 ms | 7.30 µs | 515.59× |
| `groupby_sum` | OK | 4.13 ms | 4.60 µs | 898.15× |
| `merge_inner` | OK | 1.20 ms | 60.30 µs | 19.98× |
| `merge_left` | OK | 1.01 ms | 53.80 µs | 18.82× |
| `csv_roundtrip` | OK | 1.34 ms | 195.30 µs | 6.88× |
| `melt` | OK | 1.68 ms | 11.10 µs | 151.11× |
| `pivot_sum` | OK | 3.08 ms | 52.10 µs | 59.15× |
| `rolling_mean` | OK | 317.70 µs | 2.50 µs | 127.08× |
| `mixed_dtypes` | OK | 233.40 µs | 2.10 µs | 111.14× |

### Fastest Rust wins (top 10)

| Op | Speedup |
|----|---------|
| `groupby_sum` | 898.15× |
| `select` | 533.83× |
| `describe` | 515.59× |
| `melt` | 151.11× |
| `rolling_mean` | 127.08× |
| `mean` | 119.06× |
| `mixed_dtypes` | 111.14× |
| `fillna` | 106.00× |
| `sum` | 103.73× |
| `filter_gt` | 71.76× |

## PyTorch / rustorch

96 ops · parity OK 96 · Rust faster on 87

| Op | Parity | Python | Rust | Speedup |
|----|--------|--------|------|---------|
| `zeros` | OK | 3.10 µs | 900 ns | 3.44× |
| `add` | OK | 3.50 µs | 1.10 µs | 3.18× |
| `mul` | OK | 3.50 µs | 1.10 µs | 3.18× |
| `matmul` | OK | 12.80 µs | 13.30 µs | 0.96× |
| `sum` | OK | 7.30 µs | 5.10 µs | 1.43× |
| `mean` | OK | 15.60 µs | 4.70 µs | 3.32× |
| `relu` | OK | 3.90 µs | 2.60 µs | 1.50× |
| `sigmoid` | OK | 12.90 µs | 4.60 µs | 2.80× |
| `transpose` | OK | 3.30 µs | 500 ns | 6.60× |
| `reshape` | OK | 4.30 µs | 200 ns | 21.50× |
| `linear_forward` | OK | 51.10 µs | 5.80 µs | 8.81× |
| `mse_loss` | OK | 32.00 µs | 1.40 µs | 22.86× |
| `train_step` | OK | 3.63 ms | 182.30 µs | 19.91× |
| `exp` | OK | 11.50 µs | 4.20 µs | 2.74× |
| `log` | OK | 11.70 µs | 4.30 µs | 2.72× |
| `pow` | OK | 30.10 µs | 52.00 µs | 0.58× |
| `clamp` | OK | 6.80 µs | 1.60 µs | 4.25× |
| `broadcast_add` | OK | 5.00 µs | 2.60 µs | 1.92× |
| `cat` | OK | 5.20 µs | 2.10 µs | 2.48× |
| `stack` | OK | 5.30 µs | 900 ns | 5.89× |
| `index_select` | OK | 6.40 µs | 12.70 µs | 0.50× |
| `softmax` | OK | 7.90 µs | 4.90 µs | 1.61× |
| `cross_entropy` | OK | 10.30 µs | 14.90 µs | 0.69× |
| `dropout` | OK | 5.90 µs | 8.30 µs | 0.71× |
| `sequential_forward` | OK | 117.40 µs | 15.50 µs | 7.57× |
| `adam_train_step` | OK | 5.27 ms | 126.90 µs | 41.56× |
| `embedding_forward` | OK | 13.50 µs | 7.30 µs | 1.85× |
| `layernorm_forward` | OK | 35.70 µs | 12.50 µs | 2.86× |
| `conv2d_forward` | OK | 126.20 µs | 19.30 µs | 6.54× |
| `adamw_train_step` | OK | 5.73 ms | 209.60 µs | 27.32× |
| `steplr` | OK | 324.60 µs | 0 | ∞ |
| `tanh` | OK | 19.00 µs | 6.20 µs | 3.06× |
| `gelu` | OK | 22.60 µs | 7.70 µs | 2.94× |
| `batchnorm1d_forward` | OK | 62.00 µs | 3.60 µs | 17.22× |
| `max_pool2d_forward` | OK | 11.70 µs | 2.40 µs | 4.88× |
| `flatten_forward` | OK | 7.50 µs | 800 ns | 9.38× |
| `multisteplr` | OK | 380.70 µs | 100 ns | 3807.00× |
| `batchnorm2d_forward` | OK | 65.60 µs | 8.80 µs | 7.45× |
| `avg_pool2d_forward` | OK | 8.20 µs | 2.10 µs | 3.90× |
| `cosineannealinglr` | OK | 567.60 µs | 100 ns | 5676.00× |
| `dataloader_epoch` | OK | 394.90 µs | 73.90 µs | 5.34× |
| `leaky_relu` | OK | 6.60 µs | 1.80 µs | 3.67× |
| `gru_forward` | OK | 929.20 µs | 179.20 µs | 5.19× |
| `state_dict_roundtrip` | OK | 24.10 µs | 500 ns | 48.20× |
| `lstm_forward` | OK | 348.90 µs | 235.80 µs | 1.48× |
| `adaptive_avg_pool2d_forward` | OK | 10.60 µs | 1.20 µs | 8.83× |
| `adam_state_dict` | OK | 3.78 ms | 88.80 µs | 42.60× |
| `silu` | OK | 15.80 µs | 5.00 µs | 3.16× |
| `mha_forward` | OK | 353.40 µs | 43.30 µs | 8.16× |
| `transformer_encoder_layer_forward` | OK | 565.50 µs | 105.60 µs | 5.36× |
| `transformer_encoder_forward` | OK | 1.17 ms | 390.80 µs | 3.00× |
| `sdpa_causal` | OK | 86.40 µs | 8.20 µs | 10.54× |
| `transformer_decoder_layer_forward` | OK | 1.06 ms | 319.80 µs | 3.32× |
| `add_` | OK | 9.80 µs | 13.20 µs | 0.74× |
| `relu_` | OK | 6.30 µs | 12.10 µs | 0.52× |
| `narrow` | OK | 6.30 µs | 700 ns | 9.00× |
| `mha_key_padding` | OK | 329.70 µs | 82.80 µs | 3.98× |
| `dataloader_sequential` | OK | 365.00 µs | 83.10 µs | 4.39× |
| `create_graph_second` | OK | 360.00 µs | 18.10 µs | 19.89× |
| `default_collate` | OK | 62.60 µs | 10.20 µs | 6.14× |
| `gradcheck_square` | OK | 201.00 µs | 21.20 µs | 9.48× |
| `create_graph_pow` | OK | 268.80 µs | 15.20 µs | 17.68× |
| `device_cpu` | OK | 2.90 µs | 0 | ∞ |
| `dtype_float32` | OK | 1.70 µs | 100 ns | 17.00× |
| `dtype_float64` | OK | 14.10 µs | 1.20 µs | 11.75× |
| `dtype_int64` | OK | 14.50 µs | 4.70 µs | 3.09× |
| `dtype_bool` | OK | 13.30 µs | 6.60 µs | 2.02× |
| `view_transpose_reshape` | OK | 13.50 µs | 16.60 µs | 0.81× |
| `numpy_roundtrip` | OK | 13.60 µs | 5.70 µs | 2.39× |
| `pandas_roundtrip` | OK | 97.40 µs | 5.40 µs | 18.04× |
| `create_graph_conv2d` | OK | 1.36 ms | 59.70 µs | 22.82× |
| `create_graph_batchnorm1d` | OK | 698.30 µs | 24.40 µs | 28.62× |
| `create_graph_batchnorm2d` | OK | 791.70 µs | 21.30 µs | 37.17× |
| `create_graph_max_pool2d` | OK | 288.00 µs | 20.00 µs | 14.40× |
| `create_graph_avg_pool2d` | OK | 429.60 µs | 16.10 µs | 26.68× |
| `create_graph_adaptive_avg_pool2d` | OK | 289.20 µs | 15.40 µs | 18.78× |
| `create_graph_cross_entropy` | OK | 200.80 µs | 11.80 µs | 17.02× |
| `create_graph_narrow` | OK | 215.90 µs | 16.70 µs | 12.93× |
| `nested_to_padded` | OK | 300.70 µs | 3.50 µs | 85.91× |
| `device_cuda` | OK | 100 ns | 100 ns | 1.00× |
| `fused_linear_relu` | OK | 18.50 µs | 8.30 µs | 2.23× |
| `grad_scaler_step` | OK | 2.26 ms | 68.00 µs | 33.22× |
| `custom_square` | OK | 236.60 µs | 6.70 µs | 35.31× |
| `create_graph_exp` | OK | 222.40 µs | 13.80 µs | 16.12× |
| `create_graph_sigmoid` | OK | 256.70 µs | 12.80 µs | 20.05× |
| `create_graph_silu` | OK | 392.40 µs | 13.80 µs | 28.43× |
| `create_graph_gelu` | OK | 329.00 µs | 34.50 µs | 9.54× |
| `create_graph_clamp` | OK | 299.00 µs | 12.50 µs | 23.92× |
| `create_graph_linear` | OK | 536.20 µs | 31.00 µs | 17.30× |
| `create_graph_cat` | OK | 372.20 µs | 15.50 µs | 24.01× |
| `create_graph_stack` | OK | 352.30 µs | 17.80 µs | 19.79× |
| `create_graph_bmm` | OK | 393.70 µs | 58.00 µs | 6.79× |
| `create_graph_permute` | OK | 273.40 µs | 13.80 µs | 19.81× |
| `create_graph_dropout` | OK | 234.20 µs | 10.80 µs | 21.69× |
| `create_graph_index_select` | OK | 277.70 µs | 11.90 µs | 23.34× |
| `create_graph_layernorm` | OK | 1.13 ms | 192.20 µs | 5.85× |

### Fastest Rust wins (top 10)

| Op | Speedup |
|----|---------|
| `cosineannealinglr` | 5676.00× |
| `multisteplr` | 3807.00× |
| `nested_to_padded` | 85.91× |
| `state_dict_roundtrip` | 48.20× |
| `adam_state_dict` | 42.60× |
| `adam_train_step` | 41.56× |
| `create_graph_batchnorm2d` | 37.17× |
| `custom_square` | 35.31× |
| `grad_scaler_step` | 33.22× |
| `create_graph_batchnorm1d` | 28.62× |

