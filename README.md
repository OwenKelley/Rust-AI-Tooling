# Rust AI Tooling

Building Rust equivalents of the Python AI/ML stack, with parity and speed tests.

## Start here

- Inventory: [`python-ai-ml-tooling.md`](python-ai-ml-tooling.md)
- First translation slice (NumPy): [`core_numerical/README.md`](core_numerical/README.md)

## Quick compare (NumPy)

```powershell
# Ensure cargo is visible (needed if the terminal was open before rustup install)
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

cargo build -p parity_runner --release
cd python
pip install -e .
python -m core_numerical.numpy_parity.compare
```
