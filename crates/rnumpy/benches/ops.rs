use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rnumpy::{add, matmul, mean, seeded_uniform, sum};

fn bench_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnumpy");

    for &n in &[64usize, 256, 1024] {
        let a = seeded_uniform(&[n, n], 42, -1.0, 1.0);
        let b = seeded_uniform(&[n, n], 43, -1.0, 1.0);

        group.bench_with_input(BenchmarkId::new("add", n), &n, |bencher, _| {
            bencher.iter(|| add(black_box(&a), black_box(&b)));
        });

        group.bench_with_input(BenchmarkId::new("matmul", n), &n, |bencher, _| {
            bencher.iter(|| matmul(black_box(&a), black_box(&b)));
        });

        group.bench_with_input(BenchmarkId::new("sum", n), &n, |bencher, _| {
            bencher.iter(|| sum(black_box(&a)));
        });

        group.bench_with_input(BenchmarkId::new("mean", n), &n, |bencher, _| {
            bencher.iter(|| mean(black_box(&a)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ops);
criterion_main!(benches);
