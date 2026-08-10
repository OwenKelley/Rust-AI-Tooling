//! CLI used by the Python PyTorch comparison harness.

use std::env;
use std::process;
use std::time::Instant;

use rtorch::{
    adaptive_avg_pool2d, add, add_, avg_pool2d, cat, clamp, cross_entropy, default_collate,
    dropout, exp, from_dataframe, from_numpy, fused_linear_relu, gelu,
    generate_square_subsequent_mask, grad, gradcheck_max_error, index_select, leaky_relu, linear,
    load_state_dict, log, matmul, max_pool2d, mean, mul, narrow, pow, relu, relu_, reshape,
    scaled_dot_product_attention_masked, seeded_uniform, sigmoid, silu, softmax, square_function,
    stack, state_dict, state_dict_checksum, sub, sum, tanh, to_dataframe, to_numpy, transpose, zeros,
    Adam, AdamW, BatchNorm1d, BatchNorm2d, Conv2d, CosineAnnealingLR, CrossEntropyLoss, DataLoader,
    Device, Dtype, Embedding, Flatten, GradScaler, GRU, LSTM, LayerNorm, Linear, Module, MultiStepLR,
    MultiheadAttention, MSELoss, ReLU, SGD, Sequential, StepLR, Tensor, TensorDataset,
    TransformerActivation, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer,
    bmm, nested_tensor, permute, full,
};

#[derive(Debug, Clone)]
enum Op {
    Zeros,
    Add,
    Mul,
    Matmul,
    Sum,
    Mean,
    Relu,
    Sigmoid,
    Transpose,
    Reshape,
    LinearForward,
    MseLoss,
    TrainStep,
    Exp,
    Log,
    Pow,
    Clamp,
    BroadcastAdd,
    Cat,
    Stack,
    IndexSelect,
    Softmax,
    CrossEntropy,
    Dropout,
    SequentialForward,
    AdamTrainStep,
    EmbeddingForward,
    LayerNormForward,
    Conv2dForward,
    AdamWTrainStep,
    StepLr,
    Tanh,
    Gelu,
    BatchNorm1dForward,
    MaxPool2dForward,
    FlattenForward,
    MultiStepLr,
    BatchNorm2dForward,
    AvgPool2dForward,
    CosineAnnealingLr,
    DataLoaderEpoch,
    LeakyRelu,
    GruForward,
    StateDictRoundtrip,
    LstmForward,
    AdaptiveAvgPool2dForward,
    AdamStateDictRoundtrip,
    Silu,
    MhaForward,
    TransformerEncoderLayerForward,
    TransformerEncoderForward,
    SdpaCausal,
    TransformerDecoderLayerForward,
    AddInplace,
    ReluInplace,
    Narrow,
    MhaKeyPadding,
    DataLoaderSequential,
    CreateGraphSecond,
    DefaultCollate,
    GradcheckSquare,
    CreateGraphPow,
    DeviceCpu,
    DtypeFloat32,
    DtypeFloat64,
    DtypeInt64,
    DtypeBool,
    ViewTransposeReshape,
    CustomSquare,
    CreateGraphExp,
    CreateGraphSigmoid,
    CreateGraphSilu,
    CreateGraphGelu,
    CreateGraphClamp,
    CreateGraphLinear,
    CreateGraphCat,
    CreateGraphStack,
    CreateGraphBmm,
    CreateGraphPermute,
    CreateGraphDropout,
    CreateGraphIndexSelect,
    CreateGraphLayernorm,
    NumpyRoundtrip,
    PandasRoundtrip,
    CreateGraphConv2d,
    CreateGraphBatchNorm1d,
    CreateGraphBatchNorm2d,
    CreateGraphMaxPool2d,
    CreateGraphAvgPool2d,
    CreateGraphAdaptiveAvgPool2d,
    CreateGraphCrossEntropy,
    CreateGraphNarrow,
    NestedToPadded,
    DeviceCuda,
    FusedLinearRelu,
    GradScalerStep,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "zeros" => Self::Zeros,
            "add" => Self::Add,
            "mul" => Self::Mul,
            "matmul" => Self::Matmul,
            "sum" => Self::Sum,
            "mean" => Self::Mean,
            "relu" => Self::Relu,
            "sigmoid" => Self::Sigmoid,
            "transpose" => Self::Transpose,
            "reshape" => Self::Reshape,
            "linear_forward" => Self::LinearForward,
            "mse_loss" => Self::MseLoss,
            "train_step" => Self::TrainStep,
            "exp" => Self::Exp,
            "log" => Self::Log,
            "pow" => Self::Pow,
            "clamp" => Self::Clamp,
            "broadcast_add" => Self::BroadcastAdd,
            "cat" => Self::Cat,
            "stack" => Self::Stack,
            "index_select" => Self::IndexSelect,
            "softmax" => Self::Softmax,
            "cross_entropy" => Self::CrossEntropy,
            "dropout" => Self::Dropout,
            "sequential_forward" => Self::SequentialForward,
            "adam_train_step" => Self::AdamTrainStep,
            "embedding_forward" => Self::EmbeddingForward,
            "layernorm_forward" => Self::LayerNormForward,
            "conv2d_forward" => Self::Conv2dForward,
            "adamw_train_step" => Self::AdamWTrainStep,
            "steplr" => Self::StepLr,
            "tanh" => Self::Tanh,
            "gelu" => Self::Gelu,
            "batchnorm1d_forward" => Self::BatchNorm1dForward,
            "max_pool2d_forward" => Self::MaxPool2dForward,
            "flatten_forward" => Self::FlattenForward,
            "multisteplr" => Self::MultiStepLr,
            "batchnorm2d_forward" => Self::BatchNorm2dForward,
            "avg_pool2d_forward" => Self::AvgPool2dForward,
            "cosineannealinglr" => Self::CosineAnnealingLr,
            "dataloader_epoch" => Self::DataLoaderEpoch,
            "leaky_relu" => Self::LeakyRelu,
            "gru_forward" => Self::GruForward,
            "state_dict_roundtrip" => Self::StateDictRoundtrip,
            "lstm_forward" => Self::LstmForward,
            "adaptive_avg_pool2d_forward" => Self::AdaptiveAvgPool2dForward,
            "adam_state_dict" => Self::AdamStateDictRoundtrip,
            "silu" => Self::Silu,
            "mha_forward" => Self::MhaForward,
            "transformer_encoder_layer_forward" => Self::TransformerEncoderLayerForward,
            "transformer_encoder_forward" => Self::TransformerEncoderForward,
            "sdpa_causal" => Self::SdpaCausal,
            "transformer_decoder_layer_forward" => Self::TransformerDecoderLayerForward,
            "add_" => Self::AddInplace,
            "relu_" => Self::ReluInplace,
            "narrow" => Self::Narrow,
            "mha_key_padding" => Self::MhaKeyPadding,
            "dataloader_sequential" => Self::DataLoaderSequential,
            "create_graph_second" => Self::CreateGraphSecond,
            "default_collate" => Self::DefaultCollate,
            "gradcheck_square" => Self::GradcheckSquare,
            "create_graph_pow" => Self::CreateGraphPow,
            "device_cpu" => Self::DeviceCpu,
            "dtype_float32" => Self::DtypeFloat32,
            "dtype_float64" => Self::DtypeFloat64,
            "dtype_int64" => Self::DtypeInt64,
            "dtype_bool" => Self::DtypeBool,
            "view_transpose_reshape" => Self::ViewTransposeReshape,
            "custom_square" => Self::CustomSquare,
            "create_graph_exp" => Self::CreateGraphExp,
            "create_graph_sigmoid" => Self::CreateGraphSigmoid,
            "create_graph_silu" => Self::CreateGraphSilu,
            "create_graph_gelu" => Self::CreateGraphGelu,
            "create_graph_clamp" => Self::CreateGraphClamp,
            "create_graph_linear" => Self::CreateGraphLinear,
            "create_graph_cat" => Self::CreateGraphCat,
            "create_graph_stack" => Self::CreateGraphStack,
            "create_graph_bmm" => Self::CreateGraphBmm,
            "create_graph_permute" => Self::CreateGraphPermute,
            "create_graph_dropout" => Self::CreateGraphDropout,
            "create_graph_index_select" => Self::CreateGraphIndexSelect,
            "create_graph_layernorm" => Self::CreateGraphLayernorm,
            "numpy_roundtrip" => Self::NumpyRoundtrip,
            "pandas_roundtrip" => Self::PandasRoundtrip,
            "create_graph_conv2d" => Self::CreateGraphConv2d,
            "create_graph_batchnorm1d" => Self::CreateGraphBatchNorm1d,
            "create_graph_batchnorm2d" => Self::CreateGraphBatchNorm2d,
            "create_graph_max_pool2d" => Self::CreateGraphMaxPool2d,
            "create_graph_avg_pool2d" => Self::CreateGraphAvgPool2d,
            "create_graph_adaptive_avg_pool2d" => Self::CreateGraphAdaptiveAvgPool2d,
            "create_graph_cross_entropy" => Self::CreateGraphCrossEntropy,
            "create_graph_narrow" => Self::CreateGraphNarrow,
            "nested_to_padded" => Self::NestedToPadded,
            "device_cuda" => Self::DeviceCuda,
            "fused_linear_relu" => Self::FusedLinearRelu,
            "grad_scaler_step" => Self::GradScalerStep,
            other => return Err(format!("unknown op '{other}'")),
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Zeros => "zeros",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Matmul => "matmul",
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Transpose => "transpose",
            Self::Reshape => "reshape",
            Self::LinearForward => "linear_forward",
            Self::MseLoss => "mse_loss",
            Self::TrainStep => "train_step",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Pow => "pow",
            Self::Clamp => "clamp",
            Self::BroadcastAdd => "broadcast_add",
            Self::Cat => "cat",
            Self::Stack => "stack",
            Self::IndexSelect => "index_select",
            Self::Softmax => "softmax",
            Self::CrossEntropy => "cross_entropy",
            Self::Dropout => "dropout",
            Self::SequentialForward => "sequential_forward",
            Self::AdamTrainStep => "adam_train_step",
            Self::EmbeddingForward => "embedding_forward",
            Self::LayerNormForward => "layernorm_forward",
            Self::Conv2dForward => "conv2d_forward",
            Self::AdamWTrainStep => "adamw_train_step",
            Self::StepLr => "steplr",
            Self::Tanh => "tanh",
            Self::Gelu => "gelu",
            Self::BatchNorm1dForward => "batchnorm1d_forward",
            Self::MaxPool2dForward => "max_pool2d_forward",
            Self::FlattenForward => "flatten_forward",
            Self::MultiStepLr => "multisteplr",
            Self::BatchNorm2dForward => "batchnorm2d_forward",
            Self::AvgPool2dForward => "avg_pool2d_forward",
            Self::CosineAnnealingLr => "cosineannealinglr",
            Self::DataLoaderEpoch => "dataloader_epoch",
            Self::LeakyRelu => "leaky_relu",
            Self::GruForward => "gru_forward",
            Self::StateDictRoundtrip => "state_dict_roundtrip",
            Self::LstmForward => "lstm_forward",
            Self::AdaptiveAvgPool2dForward => "adaptive_avg_pool2d_forward",
            Self::AdamStateDictRoundtrip => "adam_state_dict",
            Self::Silu => "silu",
            Self::MhaForward => "mha_forward",
            Self::TransformerEncoderLayerForward => "transformer_encoder_layer_forward",
            Self::TransformerEncoderForward => "transformer_encoder_forward",
            Self::SdpaCausal => "sdpa_causal",
            Self::TransformerDecoderLayerForward => "transformer_decoder_layer_forward",
            Self::AddInplace => "add_",
            Self::ReluInplace => "relu_",
            Self::Narrow => "narrow",
            Self::MhaKeyPadding => "mha_key_padding",
            Self::DataLoaderSequential => "dataloader_sequential",
            Self::CreateGraphSecond => "create_graph_second",
            Self::DefaultCollate => "default_collate",
            Self::GradcheckSquare => "gradcheck_square",
            Self::CreateGraphPow => "create_graph_pow",
            Self::DeviceCpu => "device_cpu",
            Self::DtypeFloat32 => "dtype_float32",
            Self::DtypeFloat64 => "dtype_float64",
            Self::DtypeInt64 => "dtype_int64",
            Self::DtypeBool => "dtype_bool",
            Self::ViewTransposeReshape => "view_transpose_reshape",
            Self::CustomSquare => "custom_square",
            Self::CreateGraphExp => "create_graph_exp",
            Self::CreateGraphSigmoid => "create_graph_sigmoid",
            Self::CreateGraphSilu => "create_graph_silu",
            Self::CreateGraphGelu => "create_graph_gelu",
            Self::CreateGraphClamp => "create_graph_clamp",
            Self::CreateGraphLinear => "create_graph_linear",
            Self::CreateGraphCat => "create_graph_cat",
            Self::CreateGraphStack => "create_graph_stack",
            Self::CreateGraphBmm => "create_graph_bmm",
            Self::CreateGraphPermute => "create_graph_permute",
            Self::CreateGraphDropout => "create_graph_dropout",
            Self::CreateGraphIndexSelect => "create_graph_index_select",
            Self::CreateGraphLayernorm => "create_graph_layernorm",
            Self::NumpyRoundtrip => "numpy_roundtrip",
            Self::PandasRoundtrip => "pandas_roundtrip",
            Self::CreateGraphConv2d => "create_graph_conv2d",
            Self::CreateGraphBatchNorm1d => "create_graph_batchnorm1d",
            Self::CreateGraphBatchNorm2d => "create_graph_batchnorm2d",
            Self::CreateGraphMaxPool2d => "create_graph_max_pool2d",
            Self::CreateGraphAvgPool2d => "create_graph_avg_pool2d",
            Self::CreateGraphAdaptiveAvgPool2d => "create_graph_adaptive_avg_pool2d",
            Self::CreateGraphCrossEntropy => "create_graph_cross_entropy",
            Self::CreateGraphNarrow => "create_graph_narrow",
            Self::NestedToPadded => "nested_to_padded",
            Self::DeviceCuda => "device_cuda",
            Self::FusedLinearRelu => "fused_linear_relu",
            Self::GradScalerStep => "grad_scaler_step",
        }
    }
}

struct Args {
    op: Op,
    size: usize,
    iters: usize,
    warmup: usize,
    seed: u64,
}

fn usage() -> ! {
    eprintln!(
        "Usage: torch_parity_runner --op <name> [--size N] [--iters N] [--warmup N] [--seed N]"
    );
    process::exit(2);
}

fn parse_args() -> Result<Args, String> {
    let mut op = None;
    let mut size = 64usize;
    let mut iters = 20usize;
    let mut warmup = 3usize;
    let mut seed = 42u64;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--op" => {
                let v = args.next().ok_or("missing --op value")?;
                op = Some(Op::parse(&v)?);
            }
            "--size" => {
                size = args
                    .next()
                    .ok_or("missing --size")?
                    .parse()
                    .map_err(|_| "bad --size")?;
            }
            "--iters" => {
                iters = args
                    .next()
                    .ok_or("missing --iters")?
                    .parse()
                    .map_err(|_| "bad --iters")?;
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .ok_or("missing --warmup")?
                    .parse()
                    .map_err(|_| "bad --warmup")?;
            }
            "--seed" => {
                seed = args
                    .next()
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|_| "bad --seed")?;
            }
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Args {
        op: op.ok_or("missing --op")?,
        size,
        iters,
        warmup,
        seed,
    })
}

fn median_u64(xs: &[u64]) -> u64 {
    let mut v = xs.to_vec();
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2
    }
}

fn make_linear(in_f: usize, out_f: usize, seed: u64) -> Linear {
    let w = seeded_uniform(&[out_f, in_f], seed, -0.5, 0.5);
    let b = seeded_uniform(&[out_f], seed + 1, -0.1, 0.1);
    Linear::from_params(w, Some(b))
}

fn train_once(n: usize, seed: u64, steps: usize) -> f64 {
    let in_f = 4usize;
    let hidden = 8usize;
    let x = seeded_uniform(&[n, in_f], seed, -1.0, 1.0);
    let y = seeded_uniform(&[n, 1], seed + 1, -1.0, 1.0);
    let l1 = make_linear(in_f, hidden, seed + 10);
    let l2 = make_linear(hidden, 1, seed + 20);
    let relu_m = ReLU;
    let loss_fn = MSELoss;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let opt = SGD::new(params, 0.05);
    let mut last = 0.0f64;
    for _ in 0..steps {
        opt.zero_grad();
        let h = relu_m.forward(&l1.forward(&x));
        let pred = l2.forward(&h);
        let loss = loss_fn.forward(&pred, &y);
        loss.backward();
        opt.step();
        last = loss.item() as f64;
    }
    last
}

fn make_indices(n: usize, seed: u64) -> Vec<usize> {
    let k = (n / 2).max(1);
    let mut state = seed;
    let mut out = Vec::with_capacity(k);
    for _ in 0..k {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push(((state >> 8) as usize) % n);
    }
    out
}

fn make_class_targets(n: usize, classes: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push(((state >> 8) as usize) % classes);
    }
    out
}

fn adam_train_once(n: usize, seed: u64, steps: usize) -> f64 {
    let in_f = 4usize;
    let hidden = 8usize;
    let classes = 3usize;
    let x = seeded_uniform(&[n, in_f], seed, -1.0, 1.0);
    let target = make_class_targets(n, classes, seed + 1);
    let l1 = make_linear(in_f, hidden, seed + 10);
    let l2 = make_linear(hidden, classes, seed + 20);
    let relu_m = ReLU;
    let loss_fn = CrossEntropyLoss;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut opt = Adam::new(params, 0.05);
    let mut last = 0.0f64;
    for _ in 0..steps {
        opt.zero_grad();
        let h = relu_m.forward(&l1.forward(&x));
        let logits = l2.forward(&h);
        let loss = loss_fn.forward(&logits, &target);
        loss.backward();
        opt.step();
        last = loss.item() as f64;
    }
    last
}

fn adamw_train_once(n: usize, seed: u64, steps: usize) -> f64 {
    let in_f = 4usize;
    let hidden = 8usize;
    let classes = 3usize;
    let x = seeded_uniform(&[n, in_f], seed, -1.0, 1.0);
    let target = make_class_targets(n, classes, seed + 1);
    let l1 = make_linear(in_f, hidden, seed + 10);
    let l2 = make_linear(hidden, classes, seed + 20);
    let relu_m = ReLU;
    let loss_fn = CrossEntropyLoss;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut opt = AdamW::new(params, 0.05, 0.01);
    let mut last = 0.0f64;
    for _ in 0..steps {
        opt.zero_grad();
        let h = relu_m.forward(&l1.forward(&x));
        let logits = l2.forward(&h);
        let loss = loss_fn.forward(&logits, &target);
        loss.backward();
        opt.step();
        last = loss.item() as f64;
    }
    last
}

fn steplr_once(steps: usize) -> f64 {
    let mut lr = 0.1f32;
    let mut sched = StepLR::new(&mut lr, 2, 0.5);
    for _ in 0..steps {
        sched.step();
    }
    lr as f64
}

fn multisteplr_once(steps: usize) -> f64 {
    let mut lr = 0.1f32;
    let mut sched = MultiStepLR::new(&mut lr, vec![2, 4], 0.5);
    for _ in 0..steps {
        sched.step();
    }
    lr as f64
}

fn cosine_once(steps: usize) -> f64 {
    let mut lr = 0.1f32;
    let mut sched = CosineAnnealingLR::new(&mut lr, 10, 0.0);
    for _ in 0..steps {
        sched.step();
    }
    lr as f64
}

fn dataloader_once(n: usize, seed: u64) -> f64 {
    let features = seeded_uniform(&[n, 4], seed, -1.0, 1.0);
    let labels = seeded_uniform(&[n, 1], seed + 1, -1.0, 1.0);
    let ds = TensorDataset::new(features, labels);
    let mut loader = DataLoader::new(&ds, 8, true, seed + 9);
    loader.epoch_checksum()
}

fn dataloader_sequential_once(n: usize, seed: u64) -> f64 {
    let features = seeded_uniform(&[n, 4], seed, -1.0, 1.0);
    let labels = seeded_uniform(&[n, 1], seed + 1, -1.0, 1.0);
    let ds = TensorDataset::new(features, labels);
    let mut loader = DataLoader::from_sequential(&ds, 8);
    loader.epoch_checksum()
}

fn create_graph_second_once(n: usize, seed: u64) -> f64 {
    let x = seeded_uniform(&[n.min(8).max(4)], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let y = sum(&mul(&x, &x));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn default_collate_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(8).max(2);
    let mut samples = Vec::with_capacity(batch);
    for i in 0..batch {
        let x = seeded_uniform(&[4], seed + i as u64, -1.0, 1.0);
        let y = seeded_uniform(&[1], seed + 100 + i as u64, -1.0, 1.0);
        samples.push((x, y));
    }
    let (xb, yb) = default_collate(&samples);
    xb.checksum() + yb.checksum()
}

fn gradcheck_square_once(n: usize, seed: u64) -> f64 {
    let m = n.min(6).max(3);
    let x = seeded_uniform(&[m], seed, -0.8, 0.8);
    gradcheck_max_error(|t| sum(&mul(t, t)), &x, 1e-3) as f64
}

fn create_graph_pow_once(n: usize, seed: u64) -> f64 {
    let m = n.min(6).max(3);
    let x = seeded_uniform(&[m], seed, 0.5, 1.5);
    x.set_requires_grad(true);
    let three = full(&x.shape(), 3.0, false);
    let y = sum(&pow(&x, &three));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn device_cpu_once(n: usize, seed: u64) -> f64 {
    let x = seeded_uniform(&[n, n], seed, -1.0, 1.0);
    let y = x.to(Device::Cpu).cpu();
    assert_eq!(y.device(), Device::Cpu);
    assert!(!y.device().is_cuda());
    assert_eq!(y.device().type_str(), "cpu");
    y.checksum()
}

fn dtype_float32_once(n: usize, seed: u64) -> f64 {
    let x = seeded_uniform(&[n, n], seed, -1.0, 1.0);
    let y = x.to_dtype(Dtype::Float32).float();
    assert_eq!(y.dtype(), Dtype::Float32);
    assert!(y.dtype().is_floating_point());
    assert_eq!(y.dtype().type_str(), "float32");
    y.checksum()
}

fn dtype_float64_once(n: usize, seed: u64) -> f64 {
    let x = seeded_uniform(&[n, n], seed, -1.0, 1.0);
    let y = x.double().float();
    assert_eq!(x.double().dtype(), Dtype::Float64);
    assert!(x.double().dtype().is_floating_point());
    assert_eq!(x.double().dtype().type_str(), "float64");
    assert_eq!(y.dtype(), Dtype::Float32);
    y.checksum()
}

fn view_transpose_reshape_once(n: usize, seed: u64) -> f64 {
    let x = seeded_uniform(&[n, n], seed, -1.0, 1.0);
    let t = transpose(&x);
    let y = reshape(&t, &[n * n]);
    y.checksum()
}

fn dtype_int64_once(n: usize, seed: u64) -> f64 {
    let x = seeded_uniform(&[n, n], seed, -2.0, 2.0);
    let i = x.long();
    assert_eq!(i.dtype(), Dtype::Int64);
    assert_eq!(i.dtype().type_str(), "int64");
    i.float().checksum()
}

fn dtype_bool_once(n: usize, seed: u64) -> f64 {
    let x = seeded_uniform(&[n, n], seed, -1.0, 1.0);
    let b = x.bool_();
    assert_eq!(b.dtype(), Dtype::Bool);
    assert_eq!(b.dtype().type_str(), "bool");
    b.float().checksum()
}

fn numpy_roundtrip_once(n: usize, seed: u64) -> f64 {
    let x = seeded_uniform(&[n, n], seed, -1.0, 1.0);
    let a = to_numpy(&x);
    let y = from_numpy(&a);
    y.checksum()
}

fn pandas_roundtrip_once(n: usize, seed: u64) -> f64 {
    let cols = 3usize;
    let x = seeded_uniform(&[n, cols], seed, -1.0, 1.0);
    let df = to_dataframe(&x, &["c0", "c1", "c2"]);
    let y = from_dataframe(&df);
    y.checksum()
}

fn custom_square_once(n: usize, seed: u64) -> f64 {
    let m = n.min(8).max(3);
    let x = seeded_uniform(&[m], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let y = sum(&square_function(&x));
    y.backward();
    let g = x.grad().unwrap_or_else(|| vec![0.0; x.numel()]);
    Tensor::from_vec(g, &x.shape(), false).checksum()
}

fn create_graph_exp_once(n: usize, seed: u64) -> f64 {
    let m = n.min(8).max(3);
    let x = seeded_uniform(&[m], seed, -0.5, 0.5);
    x.set_requires_grad(true);
    let y = sum(&exp(&x));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn create_graph_sigmoid_once(n: usize, seed: u64) -> f64 {
    let m = n.min(8).max(3);
    let x = seeded_uniform(&[m], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let y = sum(&sigmoid(&x));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn create_graph_silu_once(n: usize, seed: u64) -> f64 {
    let m = n.min(8).max(3);
    let x = seeded_uniform(&[m], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let y = sum(&silu(&x));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn create_graph_gelu_once(n: usize, seed: u64) -> f64 {
    let m = n.min(8).max(3);
    let x = seeded_uniform(&[m], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let y = sum(&gelu(&x));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn create_graph_clamp_once(n: usize, seed: u64) -> f64 {
    let m = n.min(8).max(3);
    let x = seeded_uniform(&[m], seed, -1.5, 1.5);
    x.set_requires_grad(true);
    let c = clamp(&x, -0.5, 0.5);
    let y = sum(&mul(&c, &c));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn create_graph_linear_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(6).max(2);
    let x = seeded_uniform(&[batch, 4], seed, -1.0, 1.0);
    let w = seeded_uniform(&[3, 4], seed + 1, -0.5, 0.5);
    let b = seeded_uniform(&[3], seed + 2, -0.1, 0.1);
    x.set_requires_grad(true);
    w.set_requires_grad(true);
    b.set_requires_grad(true);
    let out = linear(&x, &w, Some(&b));
    let y = sum(&mul(&out, &out));
    let gs = grad(&y, &[&x, &w, &b], true);
    let gsum = add(&add(&sum(&gs[0]), &sum(&gs[1])), &sum(&gs[2]));
    let g2 = grad(&gsum, &[&x, &w], false);
    g2[0].checksum() + g2[1].checksum()
}

fn create_graph_cat_once(n: usize, seed: u64) -> f64 {
    let rows = n.min(4).max(2);
    let a = seeded_uniform(&[rows, 3], seed, -1.0, 1.0);
    let b = seeded_uniform(&[rows, 3], seed + 1, -1.0, 1.0);
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    let c = cat(&[&a, &b], 0);
    let y = sum(&mul(&c, &c));
    let gs = grad(&y, &[&a, &b], true);
    let gsum = add(&sum(&gs[0]), &sum(&gs[1]));
    let g2 = grad(&gsum, &[&a, &b], false);
    g2[0].checksum() + g2[1].checksum()
}

fn create_graph_stack_once(n: usize, seed: u64) -> f64 {
    let m = n.min(3).max(2);
    let a = seeded_uniform(&[m, m], seed, -1.0, 1.0);
    let b = seeded_uniform(&[m, m], seed + 1, -1.0, 1.0);
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    let c = stack(&[&a, &b], 0);
    let y = sum(&mul(&c, &c));
    let gs = grad(&y, &[&a, &b], true);
    let gsum = add(&sum(&gs[0]), &sum(&gs[1]));
    let g2 = grad(&gsum, &[&a, &b], false);
    g2[0].checksum() + g2[1].checksum()
}

fn create_graph_bmm_once(n: usize, seed: u64) -> f64 {
    let bsz = n.min(3).max(2);
    let a = seeded_uniform(&[bsz, 3, 4], seed, -1.0, 1.0);
    let b = seeded_uniform(&[bsz, 4, 3], seed + 1, -1.0, 1.0);
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    let c = bmm(&a, &b);
    let y = sum(&mul(&c, &c));
    let gs = grad(&y, &[&a, &b], true);
    let gsum = add(&sum(&gs[0]), &sum(&gs[1]));
    let g2 = grad(&gsum, &[&a, &b], false);
    g2[0].checksum() + g2[1].checksum()
}

fn create_graph_permute_once(n: usize, seed: u64) -> f64 {
    let d = n.min(3).max(2);
    let x = seeded_uniform(&[d, d + 1, d + 2], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let p = permute(&x, &[2, 0, 1]);
    let y = sum(&mul(&p, &p));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn create_graph_dropout_once(n: usize, seed: u64) -> f64 {
    let m = n.min(8).max(3);
    let x = seeded_uniform(&[m], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let d = dropout(&x, 0.25, true, seed + 9);
    let y = sum(&mul(&d, &d));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn create_graph_index_select_once(n: usize, seed: u64) -> f64 {
    let rows = n.min(6).max(4);
    let x = seeded_uniform(&[rows, 3], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let idx = vec![0usize, rows / 2, rows - 1];
    let s = index_select(&x, 0, &idx);
    let y = sum(&mul(&s, &s));
    let g = grad(&y, &[&x], true);
    let g2 = grad(&sum(&g[0]), &[&x], false);
    g2[0].checksum()
}

fn create_graph_layernorm_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(4).max(2);
    let c = 4usize;
    let x = seeded_uniform(&[batch, c], seed, -1.0, 1.0);
    let w = seeded_uniform(&[c], seed + 1, 0.5, 1.5);
    let b = seeded_uniform(&[c], seed + 2, -0.1, 0.1);
    let ln = LayerNorm::from_params(w, b, 1e-5);
    x.set_requires_grad(true);
    let out = ln.forward(&x);
    let y = sum(&mul(&out, &out));
    let gs = grad(&y, &[&x, &ln.weight, &ln.bias], true);
    let s = add(&add(&sum(&gs[0]), &sum(&gs[1])), &sum(&gs[2]));
    let g2 = grad(&s, &[&x, &ln.weight, &ln.bias], false);
    g2[0].checksum() + g2[1].checksum() + g2[2].checksum()
}

fn create_graph_conv2d_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(3).max(2);
    let spatial = n.min(6).max(4);
    let cin = 2usize;
    let cout = 3usize;
    let k = 3usize;
    let x = seeded_uniform(&[batch, cin, spatial, spatial], seed, -1.0, 1.0);
    let w = seeded_uniform(&[cout, cin, k, k], seed + 1, -0.2, 0.2);
    let b = seeded_uniform(&[cout], seed + 2, -0.1, 0.1);
    let conv = Conv2d::from_params(w, Some(b));
    x.set_requires_grad(true);
    let out = conv.forward(&x);
    let y = sum(&mul(&out, &out));
    let gs = grad(&y, &[&x, &conv.weight, conv.bias.as_ref().unwrap()], true);
    gs[0].checksum() + gs[1].checksum() + gs[2].checksum()
}

fn create_graph_batchnorm1d_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(8).max(4);
    let c = 4usize;
    let x = seeded_uniform(&[batch, c], seed, -1.0, 1.0);
    let w = seeded_uniform(&[c], seed + 1, 0.5, 1.5);
    let b = seeded_uniform(&[c], seed + 2, -0.1, 0.1);
    let bn = BatchNorm1d::from_params(w, b, 1e-5, 0.1);
    x.set_requires_grad(true);
    let out = bn.forward(&x);
    let y = sum(&mul(&out, &out));
    let gs = grad(&y, &[&x, &bn.weight, &bn.bias], true);
    gs[0].checksum() + gs[1].checksum() + gs[2].checksum()
}

fn create_graph_batchnorm2d_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(3).max(2);
    let spatial = n.min(5).max(3);
    let c = 3usize;
    let x = seeded_uniform(&[batch, c, spatial, spatial], seed, -1.0, 1.0);
    let w = seeded_uniform(&[c], seed + 1, 0.5, 1.5);
    let b = seeded_uniform(&[c], seed + 2, -0.1, 0.1);
    let bn = BatchNorm2d::from_params(w, b, 1e-5, 0.1);
    x.set_requires_grad(true);
    let out = bn.forward(&x);
    let y = sum(&mul(&out, &out));
    let gs = grad(&y, &[&x, &bn.weight, &bn.bias], true);
    gs[0].checksum() + gs[1].checksum() + gs[2].checksum()
}

fn create_graph_max_pool2d_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(3).max(2);
    let spatial = n.min(6).max(4);
    let x = seeded_uniform(&[batch, 2, spatial, spatial], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let out = max_pool2d(&x, 2, 2);
    let y = sum(&mul(&out, &out));
    let g = grad(&y, &[&x], true);
    g[0].checksum()
}

fn create_graph_avg_pool2d_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(3).max(2);
    let spatial = n.min(6).max(4);
    let x = seeded_uniform(&[batch, 2, spatial, spatial], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let out = avg_pool2d(&x, 2, 2);
    let y = sum(&mul(&out, &out));
    let g = grad(&y, &[&x], true);
    g[0].checksum()
}

fn create_graph_adaptive_avg_pool2d_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(3).max(2);
    let spatial = n.min(6).max(4);
    let x = seeded_uniform(&[batch, 2, spatial, spatial], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let out = adaptive_avg_pool2d(&x, 2, 2);
    let y = sum(&mul(&out, &out));
    let g = grad(&y, &[&x], true);
    g[0].checksum()
}

fn create_graph_cross_entropy_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(8).max(4);
    let classes = 5usize;
    let logits = seeded_uniform(&[batch, classes], seed, -1.0, 1.0);
    let target = make_class_targets(batch, classes, seed + 3);
    logits.set_requires_grad(true);
    let y = cross_entropy(&logits, &target);
    let g = grad(&y, &[&logits], true);
    g[0].checksum()
}

fn create_graph_narrow_once(n: usize, seed: u64) -> f64 {
    let rows = n.min(8).max(4);
    let cols = 6usize;
    let x = seeded_uniform(&[rows, cols], seed, -1.0, 1.0);
    x.set_requires_grad(true);
    let start = 1usize;
    let length = 3usize;
    let out = narrow(&x, 1, start, length);
    let y = sum(&mul(&out, &out));
    let g = grad(&y, &[&x], true);
    g[0].checksum()
}

fn nested_to_padded_once(n: usize, seed: u64) -> f64 {
    let n = n.min(6).max(3);
    let mut parts = Vec::with_capacity(n);
    for i in 0..n {
        let len = (i % 3) + 1;
        parts.push(seeded_uniform(&[len], seed + i as u64, -1.0, 1.0));
    }
    let nt = nested_tensor(parts);
    nt.to_padded_tensor(0.0).checksum()
}

fn device_cuda_once() -> f64 {
    assert!(Device::Cuda.is_cuda());
    assert!(!Device::Cpu.is_cuda());
    assert_eq!(Device::Cuda.type_str(), "cuda");
    1.0
}

fn fused_linear_relu_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(16).max(4);
    let in_f = 8usize;
    let out_f = 6usize;
    let x = seeded_uniform(&[batch, in_f], seed, -1.0, 1.0);
    let w = seeded_uniform(&[out_f, in_f], seed + 1, -0.2, 0.2);
    let b = seeded_uniform(&[out_f], seed + 2, -0.1, 0.1);
    fused_linear_relu(&x, &w, Some(&b)).checksum()
}

fn grad_scaler_step_once(n: usize, seed: u64) -> f64 {
    let batch = n.min(16).max(4);
    let in_f = 4usize;
    let out_f = 3usize;
    let x = seeded_uniform(&[batch, in_f], seed, -1.0, 1.0);
    let target = seeded_uniform(&[batch, out_f], seed + 1, -0.5, 0.5);
    let layer = make_linear(in_f, out_f, seed + 10);
    let opt = SGD::new(layer.parameters(), 0.05);
    let mut scaler = GradScaler::new();
    scaler.scale = 8.0;
    for _ in 0..3 {
        opt.zero_grad();
        let y = layer.forward(&x);
        let loss = mean(&mul(&sub(&y, &target), &sub(&y, &target)));
        let scaled = scaler.scale_loss(&loss);
        scaled.backward();
        scaler.step_sgd(&opt);
    }
    layer.forward(&x).checksum() + scaler.get_scale() as f64
}

fn adam_state_dict_once(seed: u64) -> f64 {
    let x = seeded_uniform(&[8, 4], seed, -1.0, 1.0);
    let target = make_class_targets(8, 3, seed + 1);
    let l1 = make_linear(4, 8, seed + 10);
    let l2 = make_linear(8, 3, seed + 20);
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut opt = Adam::new(params, 0.05);
    let loss_fn = CrossEntropyLoss;
    for _ in 0..3 {
        opt.zero_grad();
        let h = ReLU.forward(&l1.forward(&x));
        let logits = l2.forward(&h);
        let loss = loss_fn.forward(&logits, &target);
        loss.backward();
        opt.step();
    }
    let sd = opt.state_dict();
    let mut opt2 = Adam::new(opt.params.clone(), 0.01);
    opt2.load_state_dict(&sd);
    opt2.state_dict().checksum()
}

fn make_emb_indices(n: usize, vocab: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push(((state >> 8) as usize) % vocab);
    }
    out
}

fn make_transformer_layer(d_model: usize, nhead: usize, dim_ff: usize, seed: u64) -> TransformerEncoderLayer {
    let ipw = seeded_uniform(&[3 * d_model, d_model], seed + 1, -0.2, 0.2);
    let ipb = seeded_uniform(&[3 * d_model], seed + 2, -0.1, 0.1);
    let ow = seeded_uniform(&[d_model, d_model], seed + 3, -0.2, 0.2);
    let ob = seeded_uniform(&[d_model], seed + 4, -0.1, 0.1);
    let mha = MultiheadAttention::from_params(ipw, ipb, ow, ob, nhead);
    let l1 = Linear::from_params(
        seeded_uniform(&[dim_ff, d_model], seed + 5, -0.2, 0.2),
        Some(seeded_uniform(&[dim_ff], seed + 6, -0.1, 0.1)),
    );
    let l2 = Linear::from_params(
        seeded_uniform(&[d_model, dim_ff], seed + 7, -0.2, 0.2),
        Some(seeded_uniform(&[d_model], seed + 8, -0.1, 0.1)),
    );
    let n1 = LayerNorm::from_params(
        seeded_uniform(&[d_model], seed + 9, 0.5, 1.5),
        seeded_uniform(&[d_model], seed + 10, -0.1, 0.1),
        1e-5,
    );
    let n2 = LayerNorm::from_params(
        seeded_uniform(&[d_model], seed + 11, 0.5, 1.5),
        seeded_uniform(&[d_model], seed + 12, -0.1, 0.1),
        1e-5,
    );
    TransformerEncoderLayer::from_parts(mha, l1, l2, n1, n2, TransformerActivation::Relu)
}

fn make_mha(d_model: usize, nhead: usize, seed: u64) -> MultiheadAttention {
    MultiheadAttention::from_params(
        seeded_uniform(&[3 * d_model, d_model], seed + 1, -0.2, 0.2),
        seeded_uniform(&[3 * d_model], seed + 2, -0.1, 0.1),
        seeded_uniform(&[d_model, d_model], seed + 3, -0.2, 0.2),
        seeded_uniform(&[d_model], seed + 4, -0.1, 0.1),
        nhead,
    )
}

fn make_decoder_layer(d_model: usize, nhead: usize, dim_ff: usize, seed: u64) -> TransformerDecoderLayer {
    let self_attn = make_mha(d_model, nhead, seed);
    let cross_attn = make_mha(d_model, nhead, seed + 100);
    let l1 = Linear::from_params(
        seeded_uniform(&[dim_ff, d_model], seed + 5, -0.2, 0.2),
        Some(seeded_uniform(&[dim_ff], seed + 6, -0.1, 0.1)),
    );
    let l2 = Linear::from_params(
        seeded_uniform(&[d_model, dim_ff], seed + 7, -0.2, 0.2),
        Some(seeded_uniform(&[d_model], seed + 8, -0.1, 0.1)),
    );
    let n1 = LayerNorm::from_params(
        seeded_uniform(&[d_model], seed + 9, 0.5, 1.5),
        seeded_uniform(&[d_model], seed + 10, -0.1, 0.1),
        1e-5,
    );
    let n2 = LayerNorm::from_params(
        seeded_uniform(&[d_model], seed + 11, 0.5, 1.5),
        seeded_uniform(&[d_model], seed + 12, -0.1, 0.1),
        1e-5,
    );
    let n3 = LayerNorm::from_params(
        seeded_uniform(&[d_model], seed + 13, 0.5, 1.5),
        seeded_uniform(&[d_model], seed + 14, -0.1, 0.1),
        1e-5,
    );
    TransformerDecoderLayer::from_parts(
        self_attn,
        cross_attn,
        l1,
        l2,
        n1,
        n2,
        n3,
        TransformerActivation::Relu,
    )
}

fn run_op(op: &Op, n: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    match op {
        Op::Zeros => {
            let t = zeros(&[n, n], false);
            let checksum = t.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(zeros(&[n, n], false));
                }),
            )
        }
        Op::Add => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = add(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(add(&a, &b));
                }),
            )
        }
        Op::Mul => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = mul(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mul(&a, &b));
                }),
            )
        }
        Op::Matmul => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = matmul(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(matmul(&a, &b));
                }),
            )
        }
        Op::Sum => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = sum(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sum(&a));
                }),
            )
        }
        Op::Mean => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = mean(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mean(&a));
                }),
            )
        }
        Op::Relu => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = relu(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(relu(&a));
                }),
            )
        }
        Op::Sigmoid => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = sigmoid(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sigmoid(&a));
                }),
            )
        }
        Op::Transpose => {
            let a = seeded_uniform(&[n, (n / 2).max(1)], seed, -1.0, 1.0);
            let checksum = transpose(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(transpose(&a));
                }),
            )
        }
        Op::Reshape => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = reshape(&a, &[n * n]).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(reshape(&a, &[n * n]));
                }),
            )
        }
        Op::LinearForward => {
            let batch = n.min(32).max(4);
            let x = seeded_uniform(&[batch, 8], seed, -1.0, 1.0);
            let layer = make_linear(8, 4, seed + 3);
            let checksum = layer.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(layer.forward(&x));
                }),
            )
        }
        Op::MseLoss => {
            let a = seeded_uniform(&[n, 4], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, 4], seed + 1, -1.0, 1.0);
            let checksum = MSELoss.forward(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(MSELoss.forward(&a, &b));
                }),
            )
        }
        Op::TrainStep => {
            let batch = n.min(32).max(8);
            let checksum = train_once(batch, seed, 5);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(train_once(batch, seed, 5));
                }),
            )
        }
        Op::Exp => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = exp(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(exp(&a));
                }),
            )
        }
        Op::Log => {
            let a = seeded_uniform(&[n, n], seed, 0.1, 2.0);
            let checksum = log(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(log(&a));
                }),
            )
        }
        Op::Pow => {
            let a = seeded_uniform(&[n, n], seed, 0.1, 2.0);
            let b = seeded_uniform(&[n, n], seed + 1, 0.5, 2.0);
            let checksum = pow(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(pow(&a, &b));
                }),
            )
        }
        Op::Clamp => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = clamp(&a, -0.5, 0.5).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(clamp(&a, -0.5, 0.5));
                }),
            )
        }
        Op::BroadcastAdd => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let checksum = add(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(add(&a, &b));
                }),
            )
        }
        Op::Cat => {
            let w = (n / 2).max(1);
            let a = seeded_uniform(&[n, w], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, w], seed + 1, -1.0, 1.0);
            let checksum = cat(&[&a, &b], 1).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cat(&[&a, &b], 1));
                }),
            )
        }
        Op::Stack => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = stack(&[&a, &b], 0).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(stack(&[&a, &b], 0));
                }),
            )
        }
        Op::IndexSelect => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let idx = make_indices(n, seed + 7);
            let checksum = index_select(&a, 1, &idx).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(index_select(&a, 1, &idx));
                }),
            )
        }
        Op::Softmax => {
            let classes = n.min(16).max(4);
            let a = seeded_uniform(&[n, classes], seed, -1.0, 1.0);
            let checksum = softmax(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(softmax(&a));
                }),
            )
        }
        Op::CrossEntropy => {
            let batch = n.min(32).max(8);
            let classes = 4usize;
            let a = seeded_uniform(&[batch, classes], seed, -1.0, 1.0);
            let target = make_class_targets(batch, classes, seed + 3);
            let checksum = cross_entropy(&a, &target).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cross_entropy(&a, &target));
                }),
            )
        }
        Op::Dropout => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = dropout(&a, 0.25, true, seed + 9).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dropout(&a, 0.25, true, seed + 9));
                }),
            )
        }
        Op::SequentialForward => {
            let batch = n.min(32).max(4);
            let x = seeded_uniform(&[batch, 8], seed, -1.0, 1.0);
            let l1 = make_linear(8, 16, seed + 3);
            let l2 = make_linear(16, 4, seed + 5);
            let model = Sequential::new(vec![
                Box::new(l1),
                Box::new(ReLU),
                Box::new(l2),
            ]);
            let checksum = model.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(model.forward(&x));
                }),
            )
        }
        Op::AdamTrainStep => {
            let batch = n.min(32).max(8);
            let checksum = adam_train_once(batch, seed, 5);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(adam_train_once(batch, seed, 5));
                }),
            )
        }
        Op::EmbeddingForward => {
            let vocab = n.min(32).max(8);
            let dim = 8usize;
            let n_idx = n.min(16).max(4);
            let weight = seeded_uniform(&[vocab, dim], seed, -0.5, 0.5);
            let emb = Embedding::from_params(weight);
            let idx = make_emb_indices(n_idx, vocab, seed + 7);
            let checksum = emb.forward_indices(&idx).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(emb.forward_indices(&idx));
                }),
            )
        }
        Op::LayerNormForward => {
            let batch = n.min(32).max(4);
            let c = 8usize;
            let x = seeded_uniform(&[batch, c], seed, -1.0, 1.0);
            let w = seeded_uniform(&[c], seed + 1, 0.5, 1.5);
            let b = seeded_uniform(&[c], seed + 2, -0.1, 0.1);
            let ln = LayerNorm::from_params(w, b, 1e-5);
            let checksum = ln.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(ln.forward(&x));
                }),
            )
        }
        Op::Conv2dForward => {
            let batch = n.min(4).max(2);
            let spatial = n.min(8).max(4);
            let cin = 2usize;
            let cout = 3usize;
            let k = 3usize;
            let x = seeded_uniform(&[batch, cin, spatial, spatial], seed, -1.0, 1.0);
            let w = seeded_uniform(&[cout, cin, k, k], seed + 1, -0.2, 0.2);
            let b = seeded_uniform(&[cout], seed + 2, -0.1, 0.1);
            let conv = Conv2d::from_params(w, Some(b));
            let checksum = conv.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(conv.forward(&x));
                }),
            )
        }
        Op::AdamWTrainStep => {
            let batch = n.min(32).max(8);
            let checksum = adamw_train_once(batch, seed, 5);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(adamw_train_once(batch, seed, 5));
                }),
            )
        }
        Op::StepLr => {
            let checksum = steplr_once(5);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(steplr_once(5));
                }),
            )
        }
        Op::Tanh => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = tanh(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(tanh(&a));
                }),
            )
        }
        Op::Gelu => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = gelu(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(gelu(&a));
                }),
            )
        }
        Op::BatchNorm1dForward => {
            let batch = n.min(32).max(4);
            let c = 8usize;
            let x = seeded_uniform(&[batch, c], seed, -1.0, 1.0);
            let w = seeded_uniform(&[c], seed + 1, 0.5, 1.5);
            let b = seeded_uniform(&[c], seed + 2, -0.1, 0.1);
            let bn = BatchNorm1d::from_params(w, b, 1e-5, 0.1);
            let checksum = bn.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(bn.forward(&x));
                }),
            )
        }
        Op::MaxPool2dForward => {
            let batch = n.min(4).max(2);
            let spatial = n.min(8).max(4);
            let x = seeded_uniform(&[batch, 2, spatial, spatial], seed, -1.0, 1.0);
            let checksum = max_pool2d(&x, 2, 2).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(max_pool2d(&x, 2, 2));
                }),
            )
        }
        Op::FlattenForward => {
            let batch = n.min(4).max(2);
            let spatial = n.min(8).max(4);
            let x = seeded_uniform(&[batch, 2, spatial, spatial], seed, -1.0, 1.0);
            let flat = Flatten::default();
            let checksum = flat.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(flat.forward(&x));
                }),
            )
        }
        Op::MultiStepLr => {
            let checksum = multisteplr_once(6);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(multisteplr_once(6));
                }),
            )
        }
        Op::BatchNorm2dForward => {
            let batch = n.min(4).max(2);
            let spatial = n.min(8).max(4);
            let c = 3usize;
            let x = seeded_uniform(&[batch, c, spatial, spatial], seed, -1.0, 1.0);
            let w = seeded_uniform(&[c], seed + 1, 0.5, 1.5);
            let b = seeded_uniform(&[c], seed + 2, -0.1, 0.1);
            let bn = BatchNorm2d::from_params(w, b, 1e-5, 0.1);
            let checksum = bn.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(bn.forward(&x));
                }),
            )
        }
        Op::AvgPool2dForward => {
            let batch = n.min(4).max(2);
            let spatial = n.min(8).max(4);
            let x = seeded_uniform(&[batch, 2, spatial, spatial], seed, -1.0, 1.0);
            let checksum = avg_pool2d(&x, 2, 2).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(avg_pool2d(&x, 2, 2));
                }),
            )
        }
        Op::CosineAnnealingLr => {
            let checksum = cosine_once(7);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cosine_once(7));
                }),
            )
        }
        Op::DataLoaderEpoch => {
            let samples = n.min(32).max(16);
            let checksum = dataloader_once(samples, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dataloader_once(samples, seed));
                }),
            )
        }
        Op::LeakyRelu => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = leaky_relu(&a, 0.01).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(leaky_relu(&a, 0.01));
                }),
            )
        }
        Op::GruForward => {
            let batch = n.min(4).max(2);
            let seq = n.min(6).max(3);
            let input_size = 4usize;
            let hidden = 5usize;
            let x = seeded_uniform(&[batch, seq, input_size], seed, -1.0, 1.0);
            let w_ih = seeded_uniform(&[3 * hidden, input_size], seed + 1, -0.2, 0.2);
            let w_hh = seeded_uniform(&[3 * hidden, hidden], seed + 2, -0.2, 0.2);
            let b_ih = seeded_uniform(&[3 * hidden], seed + 3, -0.1, 0.1);
            let b_hh = seeded_uniform(&[3 * hidden], seed + 4, -0.1, 0.1);
            let gru = GRU::from_params(w_ih, w_hh, b_ih, b_hh);
            let (out, _hn) = gru.forward_seq(&x, None);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(gru.forward_seq(&x, None).0);
                }),
            )
        }
        Op::StateDictRoundtrip => {
            let w = seeded_uniform(&[4, 3], seed, -0.5, 0.5);
            let b = seeded_uniform(&[4], seed + 1, -0.1, 0.1);
            let layer = Linear::from_params(w, Some(b));
            let named = [
                ("weight", &layer.weight),
                ("bias", layer.bias.as_ref().unwrap()),
            ];
            let sd = state_dict(&named);
            let w2 = zeros(&[4, 3], true);
            let b2 = zeros(&[4], true);
            let layer2 = Linear::from_params(w2, Some(b2));
            {
                let named2 = [
                    ("weight", &layer2.weight),
                    ("bias", layer2.bias.as_ref().unwrap()),
                ];
                load_state_dict(&named2, &sd);
            }
            let x = seeded_uniform(&[8, 3], seed + 2, -1.0, 1.0);
            let checksum = layer2.forward(&x).checksum() + state_dict_checksum(&sd);
            (
                checksum,
                Box::new(move || {
                    let named2 = [
                        ("weight", &layer2.weight),
                        ("bias", layer2.bias.as_ref().unwrap()),
                    ];
                    let sd2 = state_dict(&named2);
                    std::hint::black_box(state_dict_checksum(&sd2));
                }),
            )
        }
        Op::LstmForward => {
            let batch = n.min(4).max(2);
            let seq = n.min(6).max(3);
            let input_size = 4usize;
            let hidden = 5usize;
            let x = seeded_uniform(&[batch, seq, input_size], seed, -1.0, 1.0);
            let w_ih = seeded_uniform(&[4 * hidden, input_size], seed + 1, -0.2, 0.2);
            let w_hh = seeded_uniform(&[4 * hidden, hidden], seed + 2, -0.2, 0.2);
            let b_ih = seeded_uniform(&[4 * hidden], seed + 3, -0.1, 0.1);
            let b_hh = seeded_uniform(&[4 * hidden], seed + 4, -0.1, 0.1);
            let lstm = LSTM::from_params(w_ih, w_hh, b_ih, b_hh);
            let (out, _) = lstm.forward_seq(&x, None);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(lstm.forward_seq(&x, None).0);
                }),
            )
        }
        Op::AdaptiveAvgPool2dForward => {
            let batch = n.min(4).max(2);
            let spatial = n.min(8).max(4);
            let x = seeded_uniform(&[batch, 3, spatial, spatial], seed, -1.0, 1.0);
            let checksum = adaptive_avg_pool2d(&x, 2, 2).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(adaptive_avg_pool2d(&x, 2, 2));
                }),
            )
        }
        Op::AdamStateDictRoundtrip => {
            let checksum = adam_state_dict_once(seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(adam_state_dict_once(seed));
                }),
            )
        }
        Op::Silu => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = silu(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(silu(&a));
                }),
            )
        }
        Op::MhaForward => {
            let batch = n.min(2).max(1);
            let seq = n.min(4).max(2);
            let embed = 8usize;
            let heads = 2usize;
            let x = seeded_uniform(&[batch, seq, embed], seed, -1.0, 1.0);
            let ipw = seeded_uniform(&[3 * embed, embed], seed + 1, -0.2, 0.2);
            let ipb = seeded_uniform(&[3 * embed], seed + 2, -0.1, 0.1);
            let ow = seeded_uniform(&[embed, embed], seed + 3, -0.2, 0.2);
            let ob = seeded_uniform(&[embed], seed + 4, -0.1, 0.1);
            let mha = MultiheadAttention::from_params(ipw, ipb, ow, ob, heads);
            let checksum = mha.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mha.forward(&x));
                }),
            )
        }
        Op::TransformerEncoderLayerForward => {
            let batch = n.min(2).max(1);
            let seq = n.min(4).max(2);
            let d_model = 8usize;
            let nhead = 2usize;
            let dim_ff = 16usize;
            let x = seeded_uniform(&[batch, seq, d_model], seed, -1.0, 1.0);
            let layer = make_transformer_layer(d_model, nhead, dim_ff, seed);
            let checksum = layer.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(layer.forward(&x));
                }),
            )
        }
        Op::TransformerEncoderForward => {
            let batch = n.min(2).max(1);
            let seq = n.min(4).max(2);
            let d_model = 8usize;
            let nhead = 2usize;
            let dim_ff = 16usize;
            let x = seeded_uniform(&[batch, seq, d_model], seed, -1.0, 1.0);
            let enc = TransformerEncoder::from_layers(vec![
                make_transformer_layer(d_model, nhead, dim_ff, seed),
                make_transformer_layer(d_model, nhead, dim_ff, seed + 100),
            ]);
            let checksum = enc.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(enc.forward(&x));
                }),
            )
        }
        Op::SdpaCausal => {
            let batch = n.min(2).max(1);
            let seq = n.min(4).max(2);
            let d = 8usize;
            let q = seeded_uniform(&[batch, seq, d], seed, -1.0, 1.0);
            let k = seeded_uniform(&[batch, seq, d], seed + 1, -1.0, 1.0);
            let v = seeded_uniform(&[batch, seq, d], seed + 2, -1.0, 1.0);
            let mask = generate_square_subsequent_mask(seq);
            let checksum = scaled_dot_product_attention_masked(&q, &k, &v, Some(&mask)).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(scaled_dot_product_attention_masked(
                        &q,
                        &k,
                        &v,
                        Some(&mask),
                    ));
                }),
            )
        }
        Op::TransformerDecoderLayerForward => {
            let batch = n.min(2).max(1);
            let tgt_len = n.min(3).max(2);
            let mem_len = n.min(4).max(2);
            let d_model = 8usize;
            let nhead = 2usize;
            let dim_ff = 16usize;
            let tgt = seeded_uniform(&[batch, tgt_len, d_model], seed, -1.0, 1.0);
            let mem = seeded_uniform(&[batch, mem_len, d_model], seed + 1, -1.0, 1.0);
            let mask = generate_square_subsequent_mask(tgt_len);
            let layer = make_decoder_layer(d_model, nhead, dim_ff, seed);
            let checksum = layer.forward(&tgt, &mem, Some(&mask)).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(layer.forward(&tgt, &mem, Some(&mask)));
                }),
            )
        }
        Op::AddInplace => {
            let a0 = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let a0_data = a0.data();
            let work = Tensor::from_vec(a0_data.clone(), &a0.shape(), false);
            let checksum = {
                add_(&work, &b);
                work.checksum()
            };
            (
                checksum,
                Box::new(move || {
                    work.copy_from_slice(&a0_data);
                    add_(&work, &b);
                    std::hint::black_box(());
                }),
            )
        }
        Op::ReluInplace => {
            let a0 = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let a0_data = a0.data();
            let work = Tensor::from_vec(a0_data.clone(), &a0.shape(), false);
            let checksum = {
                relu_(&work);
                work.checksum()
            };
            (
                checksum,
                Box::new(move || {
                    work.copy_from_slice(&a0_data);
                    relu_(&work);
                    std::hint::black_box(());
                }),
            )
        }
        Op::Narrow => {
            let rows = n.min(16).max(4);
            let cols = n.min(12).max(4);
            let a = seeded_uniform(&[rows, cols], seed, -1.0, 1.0);
            let start = 1usize;
            let length = (cols / 2).max(1);
            let checksum = narrow(&a, 1, start, length).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(narrow(&a, 1, start, length));
                }),
            )
        }
        Op::MhaKeyPadding => {
            let batch = n.min(2).max(1);
            let seq = n.min(4).max(2);
            let embed = 8usize;
            let heads = 2usize;
            let x = seeded_uniform(&[batch, seq, embed], seed, -1.0, 1.0);
            let ipw = seeded_uniform(&[3 * embed, embed], seed + 1, -0.2, 0.2);
            let ipb = seeded_uniform(&[3 * embed], seed + 2, -0.1, 0.1);
            let ow = seeded_uniform(&[embed, embed], seed + 3, -0.2, 0.2);
            let ob = seeded_uniform(&[embed], seed + 4, -0.1, 0.1);
            let mha = MultiheadAttention::from_params(ipw, ipb, ow, ob, heads);
            // Pad the last key position for every batch row (f32 >0.5 ⇒ pad).
            let mut kpm_data = vec![0.0f32; batch * seq];
            for b in 0..batch {
                kpm_data[b * seq + (seq - 1)] = 1.0;
            }
            let kpm = Tensor::from_vec(kpm_data, &[batch, seq], false);
            let checksum = mha
                .forward_qkv_masked(&x, &x, &x, None, Some(&kpm))
                .0
                .checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mha.forward_qkv_masked(&x, &x, &x, None, Some(&kpm)).0);
                }),
            )
        }
        Op::DataLoaderSequential => {
            let samples = n.min(32).max(16);
            let checksum = dataloader_sequential_once(samples, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dataloader_sequential_once(samples, seed));
                }),
            )
        }
        Op::CreateGraphSecond => {
            let checksum = create_graph_second_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_second_once(n, seed));
                }),
            )
        }
        Op::DefaultCollate => {
            let checksum = default_collate_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(default_collate_once(n, seed));
                }),
            )
        }
        Op::GradcheckSquare => {
            let checksum = gradcheck_square_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(gradcheck_square_once(n, seed));
                }),
            )
        }
        Op::CreateGraphPow => {
            let checksum = create_graph_pow_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_pow_once(n, seed));
                }),
            )
        }
        Op::DeviceCpu => {
            let checksum = device_cpu_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(device_cpu_once(n, seed));
                }),
            )
        }
        Op::DtypeFloat32 => {
            let checksum = dtype_float32_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dtype_float32_once(n, seed));
                }),
            )
        }
        Op::DtypeFloat64 => {
            let checksum = dtype_float64_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dtype_float64_once(n, seed));
                }),
            )
        }
        Op::ViewTransposeReshape => {
            let checksum = view_transpose_reshape_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(view_transpose_reshape_once(n, seed));
                }),
            )
        }
        Op::DtypeInt64 => {
            let checksum = dtype_int64_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dtype_int64_once(n, seed));
                }),
            )
        }
        Op::DtypeBool => {
            let checksum = dtype_bool_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dtype_bool_once(n, seed));
                }),
            )
        }
        Op::CustomSquare => {
            let checksum = custom_square_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(custom_square_once(n, seed));
                }),
            )
        }
        Op::CreateGraphExp => {
            let checksum = create_graph_exp_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_exp_once(n, seed));
                }),
            )
        }
        Op::CreateGraphSigmoid => {
            let checksum = create_graph_sigmoid_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_sigmoid_once(n, seed));
                }),
            )
        }
        Op::CreateGraphSilu => {
            let checksum = create_graph_silu_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_silu_once(n, seed));
                }),
            )
        }
        Op::CreateGraphGelu => {
            let checksum = create_graph_gelu_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_gelu_once(n, seed));
                }),
            )
        }
        Op::CreateGraphClamp => {
            let checksum = create_graph_clamp_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_clamp_once(n, seed));
                }),
            )
        }
        Op::CreateGraphLinear => {
            let checksum = create_graph_linear_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_linear_once(n, seed));
                }),
            )
        }
        Op::CreateGraphCat => {
            let checksum = create_graph_cat_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_cat_once(n, seed));
                }),
            )
        }
        Op::CreateGraphStack => {
            let checksum = create_graph_stack_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_stack_once(n, seed));
                }),
            )
        }
        Op::CreateGraphBmm => {
            let checksum = create_graph_bmm_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_bmm_once(n, seed));
                }),
            )
        }
        Op::CreateGraphPermute => {
            let checksum = create_graph_permute_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_permute_once(n, seed));
                }),
            )
        }
        Op::CreateGraphDropout => {
            let checksum = create_graph_dropout_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_dropout_once(n, seed));
                }),
            )
        }
        Op::CreateGraphIndexSelect => {
            let checksum = create_graph_index_select_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_index_select_once(n, seed));
                }),
            )
        }
        Op::CreateGraphLayernorm => {
            let checksum = create_graph_layernorm_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_layernorm_once(n, seed));
                }),
            )
        }
        Op::NumpyRoundtrip => {
            let checksum = numpy_roundtrip_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(numpy_roundtrip_once(n, seed));
                }),
            )
        }
        Op::PandasRoundtrip => {
            let checksum = pandas_roundtrip_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(pandas_roundtrip_once(n, seed));
                }),
            )
        }
        Op::CreateGraphConv2d => {
            let checksum = create_graph_conv2d_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_conv2d_once(n, seed));
                }),
            )
        }
        Op::CreateGraphBatchNorm1d => {
            let checksum = create_graph_batchnorm1d_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_batchnorm1d_once(n, seed));
                }),
            )
        }
        Op::CreateGraphBatchNorm2d => {
            let checksum = create_graph_batchnorm2d_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_batchnorm2d_once(n, seed));
                }),
            )
        }
        Op::CreateGraphMaxPool2d => {
            let checksum = create_graph_max_pool2d_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_max_pool2d_once(n, seed));
                }),
            )
        }
        Op::CreateGraphAvgPool2d => {
            let checksum = create_graph_avg_pool2d_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_avg_pool2d_once(n, seed));
                }),
            )
        }
        Op::CreateGraphAdaptiveAvgPool2d => {
            let checksum = create_graph_adaptive_avg_pool2d_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_adaptive_avg_pool2d_once(n, seed));
                }),
            )
        }
        Op::CreateGraphCrossEntropy => {
            let checksum = create_graph_cross_entropy_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_cross_entropy_once(n, seed));
                }),
            )
        }
        Op::CreateGraphNarrow => {
            let checksum = create_graph_narrow_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(create_graph_narrow_once(n, seed));
                }),
            )
        }
        Op::NestedToPadded => {
            let checksum = nested_to_padded_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(nested_to_padded_once(n, seed));
                }),
            )
        }
        Op::DeviceCuda => {
            let checksum = device_cuda_once();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(device_cuda_once());
                }),
            )
        }
        Op::FusedLinearRelu => {
            let checksum = fused_linear_relu_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(fused_linear_relu_once(n, seed));
                }),
            )
        }
        Op::GradScalerStep => {
            let checksum = grad_scaler_step_once(n, seed);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(grad_scaler_step_once(n, seed));
                }),
            )
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            usage();
        }
    };
    if args.iters == 0 {
        eprintln!("error: iters must be > 0");
        process::exit(1);
    }

    let (checksum, mut thunk) = run_op(&args.op, args.size, args.seed);
    for _ in 0..args.warmup {
        thunk();
    }
    let mut samples = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t0 = Instant::now();
        thunk();
        samples.push(t0.elapsed().as_nanos() as u64);
    }
    let mean_ns = samples.iter().map(|&x| x as f64).sum::<f64>() / samples.len() as f64;
    println!(
        "{{\n  \"language\": \"rust\",\n  \"op\": \"{}\",\n  \"size\": {},\n  \"iters\": {},\n  \"warmup\": {},\n  \"seed\": {},\n  \"median_ns\": {},\n  \"mean_ns\": {:.6},\n  \"min_ns\": {},\n  \"max_ns\": {},\n  \"checksum\": {:.17e}\n}}",
        args.op.as_str(),
        args.size,
        args.iters,
        args.warmup,
        args.seed,
        median_u64(&samples),
        mean_ns,
        samples.iter().min().unwrap(),
        samples.iter().max().unwrap(),
        checksum
    );
}
