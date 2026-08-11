//! Reference MLP: Linear → ReLU → Linear + CrossEntropy + Adam (few steps).
//!
//! ```text
//! cargo run -p rustorch --example reference_mlp --release
//! ```

use rustorch::{
    cross_entropy, fused_linear_relu, linear, no_grad, seeded_uniform, Adam, Linear, Module, ReLU,
};

fn make_targets(n: usize, classes: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push(((state >> 8) as usize) % classes);
    }
    out
}

fn main() {
    let batch = 16usize;
    let in_f = 8usize;
    let hidden = 16usize;
    let classes = 4usize;
    let seed = 42u64;

    let x = seeded_uniform(&[batch, in_f], seed, -1.0, 1.0);
    let target = make_targets(batch, classes, seed + 1);
    let l1 = Linear::from_params(
        seeded_uniform(&[hidden, in_f], seed + 10, -0.2, 0.2),
        Some(seeded_uniform(&[hidden], seed + 11, -0.1, 0.1)),
    );
    let l2 = Linear::from_params(
        seeded_uniform(&[classes, hidden], seed + 20, -0.2, 0.2),
        Some(seeded_uniform(&[classes], seed + 21, -0.1, 0.1)),
    );

    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut opt = Adam::new(params, 0.05);

    let mut last_loss = 0.0f64;
    for step in 0..5 {
        opt.zero_grad();
        let h = ReLU.forward(&l1.forward(&x));
        let logits = l2.forward(&h);
        let loss = cross_entropy(&logits, &target);
        loss.backward();
        opt.step();
        last_loss = loss.checksum();
        println!("step={step} loss_checksum={last_loss}");
    }

    let fused_cs = no_grad(|| {
        let h = fused_linear_relu(&x, &l1.weight, l1.bias.as_ref());
        linear(&h, &l2.weight, l2.bias.as_ref()).checksum()
    });
    println!("final fused_logits_checksum={fused_cs} last_loss={last_loss}");
}
