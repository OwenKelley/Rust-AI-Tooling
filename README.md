# Rust AI Tooling

Building Rust equivalents of the Python AI/ML stack, with parity and speed tests.

## Start here

- Inventory: [`python-ai-ml-tooling.md`](python-ai-ml-tooling.md)
- NumPy slice: [`core_numerical/README.md`](core_numerical/README.md)
- SciPy slice: [`core_numerical/SCIPY.md`](core_numerical/SCIPY.md)
- Next roadmap: [`core_numerical/ROADMAP.md`](core_numerical/ROADMAP.md)
- Speed comparisons: [`core_numerical/SPEED.md`](core_numerical/SPEED.md)
- End-to-end examples: [`example comparisons/`](example%20comparisons/) (MNIST MLP wall-clock)

## Quick compare (NumPy)

```powershell
# Ensure cargo is visible (needed if the terminal was open before rustup install)
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

cargo build -p parity_runner --release
cd python
pip install -e .
python -m core_numerical.numpy_parity.compare
```

## Quick compare (SciPy)

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

cargo build -p parity_runner --bin scipy_parity_runner --release
cd python
pip install -e .
python -m core_numerical.scipy_parity.compare
```
