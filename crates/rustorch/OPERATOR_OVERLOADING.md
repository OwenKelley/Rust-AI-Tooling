# RusTorch operator overloading candidates

Pick which rows you want implemented. Free functions stay; overloads would call them so behavior and autograd stay identical.

**Status today:** low-risk arithmetic and methods are in [`tensor_ops.rs`](src/tensor_ops.rs) (`+ - * /`, `-`, `+= -= *=`, `matmul`/`t`/`sum`/…). Remaining rows below are still candidates.

**Rust constraint:** only traits that exist in `std::ops` / `std::cmp` / indexing can be overloaded. This list excludes syntax Rust cannot express (no `@`, no `**`, no elementwise `<` returning a tensor).

### Implemented (do not re-request)

| IDs | What |
|-----|------|
| A1–A5, O1–O3 | `+ - * /` and unary `-` for `Tensor` / `&Tensor` combos |
| B1–B3 | `+= -= *=` |
| C3, H1–H7, D2 | `matmul`/`bmm`/`t`/`reshape`/`view`/`sum`/`mean`/`pow`/`abs`/`exp`/`log` methods |

---

## How to choose

For each row, reply with the **ID** (e.g. `A1`, `B2`) you want. Suggested defaults are marked ★.

When an op needs several ownership shapes (`Tensor`, `&Tensor`, `f32`), treat those as one feature unless noted.

---

## A. Arithmetic (elementwise) — ★ highest UX impact

Existing: `add`, `sub`, `mul`, `div`, `neg`.

| ID | Syntax (desired) | Rust trait(s) | Calls today | Notes |
|----|------------------|---------------|-------------|--------|
| A1 ★ | `a + b` | `Add` | `add` | Also `&a + &b`, `a + &b`, `&a + b` if you want PyTorch-like flexibility |
| A2 ★ | `a - b` | `Sub` | `sub` | Same ref variants |
| A3 ★ | `a * b` | `Mul` | `mul` | **Elementwise only** (PyTorch `*`). Do **not** also use `*` for matmul |
| A4 ★ | `a / b` | `Div` | `div` | |
| A5 ★ | `-a` | `Neg` | `neg` | |
| A6 | `a % b` | `Rem` | *(none)* | Remainder / fmod; needs a new kernel first |
| A7 | `a + 1.0`, `1.0 + a`, … | `Add<f32>` / `Add<Tensor> for f32` | broadcast scalar | Needs scalar–tensor broadcast helpers (partially via broadcast paths) |

---

## B. In-place arithmetic — ★ high (matches `add_`, etc.)

Existing: `add_`, `sub_`, `mul_` (no `div_` today).

| ID | Syntax | Rust trait(s) | Calls today | Notes |
|----|--------|---------------|-------------|--------|
| B1 ★ | `a += b` | `AddAssign` | `add_` | Typically `impl AddAssign<&Tensor> for Tensor` |
| B2 ★ | `a -= b` | `SubAssign` | `sub_` | |
| B3 ★ | `a *= b` | `MulAssign` | `mul_` | Elementwise |
| B4 | `a /= b` | `DivAssign` | *(none)* | Add `div_` first, then overload |
| B5 | `a += 1.0` etc. | `AddAssign<f32>` … | *(partial)* | Scalar in-place |

**Autograd caution:** in-place on tensors that require grad / share storage is already delicate in PyTorch; same rules would apply.

---

## C. Matmul

Rust has no `@` operator. Prefer a method (or keep the free function).

| ID | Syntax | Rust trait(s) | Calls today | Notes |
|----|--------|---------------|-------------|--------|
| C2 | `a % b` as matmul | `Rem` | `matmul` | Possible but non-idiomatic; clashes with remainder if A6 exists |
| C3 ★ | `matmul(&a, &b)` / `a.matmul(&b)` | method | `matmul` | Recommended |
| C4 | `a * b` as matmul | `Mul` | `matmul` | Possible only if you **skip A3**; conflicts with elementwise `*` |

---

## D. Power / other unary math

| ID | Syntax | Rust trait(s) | Calls today | Notes |
|----|--------|---------------|-------------|--------|
| D2 | `a.pow(b)` method | method | `pow` | Preferred (Rust has no `**`) |
| D3 | `!a` | `Not` | *(none)* | Possible for Bool tensors; poor fit for float |

Keep as functions (no overload): `abs`, `exp`, `log`, `clamp`, `relu`, `sum`, `mean`, …

---

## E. Indexing / slicing — large UX win, more design work

Existing: `index_select`, `gather_rows`, `narrow`, `select`, `chunk`.

| ID | Syntax | Rust trait(s) | Calls today | Notes |
|----|--------|---------------|-------------|--------|
| E1 | `t[i]` (1D or first dim) | `Index<usize>` | `select` / row gather | Return type: owned `Tensor` vs view is a big design choice |
| E2 | `t[i] = …` | `IndexMut<usize>` | copy into storage | Hard with shared `Rc` storage + autograd |
| E3 | `t[start..end]` | `Index<Range<usize>>` | `narrow` | Dim 0 only unless a richer index type exists |
| E4 | `t[idxs]` with `idxs: &[usize]` or `Vec<usize>` | `Index<&[usize]>` etc. | `index_select` / `gather_rows` | Valid Rust; not Python’s `t[[…]]` |
| E5 | `t[(r, c)]` | `Index<(usize, usize)>` | gather element / narrow | 2D convenience |

**Recommendation:** treat E\* as a separate design pass from A/B.

---

## F. Comparison — no ops in crate yet

Rust’s `<` / `<=` / `>` / `>=` always yield `bool` for the whole value, so they cannot mean PyTorch-style elementwise compare → bool tensor. Use methods instead.

| ID | Syntax | Rust trait(s) | Calls today | Notes |
|----|--------|---------------|-------------|--------|
| F1 | `a == b` | `PartialEq` | *(none)* | Whole-tensor equality (`bool`), not elementwise; awkward for floats (prefer `allclose`) |
| F3 | `a.lt(&b)`, `le`, `gt`, `ge`, `eq` | methods | *(new)* | Elementwise → bool/`Tensor`; PyTorch-like |

---

## G. Bitwise — only after bool/int kernels exist

| ID | Syntax | Trait | Notes |
|----|--------|-------|--------|
| G1 | `a & b`, `a \| b`, `a ^ b`, `!a` | `BitAnd`, `BitOr`, `BitXor`, `Not` | Possible for Bool/Int tensors once elementwise kernels exist |

---

## H. Convenience methods (not operators, but often grouped with this UX work)

| ID | API | Wraps | Notes |
|----|-----|-------|--------|
| H1 ★ | `a.matmul(&b)` | `matmul` | |
| H2 | `a.bmm(&b)` | `bmm` | |
| H3 ★ | `a.t()` / `a.transpose()` | `transpose` | |
| H4 | `a.reshape([…])` / `a.view([…])` | `reshape` | |
| H5 | `a.sum()` / `a.mean()` | `sum` / `mean` | Method form of free fns |
| H6 | `a.pow(&b)` | `pow` | Same as D2 |
| H7 | `a.abs()` / `a.exp()` / `a.log()` | matching fns | |

These avoid trait coherence / ownership issues and still improve PyTorch familiarity.

---

## Ownership matrix (for any chosen arithmetic op)

If you enable A1–A5, pick which LHS/RHS forms you want:

| Variant ID | Forms | Typical impl |
|------------|-------|--------------|
| O1 ★ | `&Tensor op &Tensor` → `Tensor` | Most common; no move |
| O2 | `Tensor op Tensor` → `Tensor` | Consumes both (or clones internally) |
| O3 | `Tensor op &Tensor`, `&Tensor op Tensor` | Mixed |
| O4 | `&Tensor op f32`, `f32 op &Tensor` | Scalar broadcast |
| O5 | `Tensor op f32`, `f32 op Tensor` | Scalar broadcast consuming |

PyTorch allows almost all of these. Implementing **O1 only** is enough for good UX with lowest complexity.

---

## Suggested bundles

| Bundle | IDs | Intent |
|--------|-----|--------|
| **Minimal ★** | A1–A5, O1, optionally B1–B3 | `+ - * /` and unary `-` (+ in-place) |
| **PyTorch-ish arithmetic** | Minimal + O4 + B1–B3 + H1 + H3 + H5 | Scalars + methods for matmul/t/sum |
| **Full sugar** | PyTorch-ish + E1 + E3 + H\* | Indexing; more design/review |
| **Avoid** | C4 together with A3; C2 if A6 also exists; F1 as exact float equality | Misleading / conflicting |

---

## Implementation sketch (for later)

```text
crates/rustorch/src/ops.rs          # keep add/mul/… as source of truth
crates/rustorch/src/tensor_ops.rs   # new: impl Add/Sub/… for Tensor (thin wrappers)
crates/rustorch/src/lib.rs          # mod tensor_ops;
crates/rustorch/TRANSLATING.md      # document a + b ↔ add
```

Autograd: wrappers must call existing `add`/`mul`/… so `GradFn` wiring stays unchanged.

---

## Your checklist

Copy and mark:

```text
[ ] A1 +
[ ] A2 -
[ ] A3 * (elementwise)
[ ] A4 /
[ ] A5 unary -
[ ] A6 %
[ ] A7 scalar ±*/ 

[ ] B1 +=
[ ] B2 -=
[ ] B3 *=
[ ] B4 /=
[ ] B5 scalar assign

[ ] C2 % as matmul (discouraged)
[ ] C3 matmul method
[ ] C4 * as matmul (only without A3)

[ ] D2 pow method
[ ] D3 ! for bool

[ ] E1 t[i]
[ ] E2 t[i] =
[ ] E3 t[range]
[ ] E4 t[idxs] (&[usize] / Vec)
[ ] E5 t[(r,c)]

[ ] F1 PartialEq (whole tensor)
[ ] F3 comparison methods (elementwise)

[ ] G1 bitwise

[ ] H1 matmul method
[ ] H2 bmm method
[ ] H3 t()/transpose
[ ] H4 reshape/view
[ ] H5 sum/mean methods
[ ] H6 pow method
[ ] H7 abs/exp/log methods

Ownership: O1 / O2 / O3 / O4 / O5
```
