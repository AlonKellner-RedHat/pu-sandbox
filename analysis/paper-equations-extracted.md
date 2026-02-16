# Paper Equations Extracted from TeX Source

**Source**: analysis/paper-source/1703.00593/pu_nnre.tex
**Extracted**: 2026-02-16
**Purpose**: Reference for mathematical discrepancy analysis (User Story 1)

---

## PN Risk Estimator

**Location**: Line 141, Label `eq:risk-pn-hat`

**Equation**:
```
\hRpn(g) = \pip\hRp^+(g) + \pin\hRn^-(g)
```

**Components**:
- `\pip` = Prior probability of positive class (π_p)
- `\pin` = Prior probability of negative class (π_n = 1 - π_p)
- `\hRp^+(g)` = Empirical positive risk: `(1/N_p) * Σ ℓ(g(x_p_i), +1)`
- `\hRn^-(g)` = Empirical negative risk: `(1/N_n) * Σ ℓ(g(x_n_i), -1)`

**Description** (from paper):
> In PN learning, thanks to the availability of $\Xp$ and $\Xn$, $R(g)$ can be approximated directly

---

## uPU Risk Estimator

**Location**: Line 146, Label `eq:risk-pu-hat`

**Equation**:
```
\hRpu(g) = \pip\hRp^+(g) - \pip\hRp^-(g) + \hRu^-(g)
```

**Components**:
- `\pip` = Prior probability of positive class (π_p)
- `\hRp^+(g)` = Empirical positive risk with positive label: `(1/N_p) * Σ ℓ(g(x_p_i), +1)`
- `\hRp^-(g)` = Empirical positive risk with negative label: `(1/N_p) * Σ ℓ(g(x_p_i), -1)`
- `\hRu^-(g)` = Empirical unlabeled risk: `(1/N_u) * Σ ℓ(g(x_u_i), -1)`

**Description** (from paper):
> As $\pin\prn(x)=p(x)-\pip\prp(x)$, we can obtain that $\pin\Rn^-(g)=\Ru^-(g)-\pip\Rp^-(g)$

**Mathematical Derivation**:
```
R(g) = π_p * R_p^+(g) + π_n * R_n^-(g)
     = π_p * R_p^+(g) + [R_u^-(g) - π_p * R_p^-(g)]
     = π_p * R_p^+(g) - π_p * R_p^-(g) + R_u^-(g)
```

**Key Property**: Can go negative during training (may cause overfitting)

---

## nnPU Risk Estimator (Non-Negative)

**Location**: Line 223, Label `eq:risk-pu-tilde`

**Equation**:
```
\tRpu(g) = \pip\hRp^+(g) + max{0, \hRu^-(g) - \pip\hRp^-(g)}
```

**Components**:
- `\pip` = Prior probability of positive class (π_p)
- `\hRp^+(g)` = Empirical positive risk with positive label: `(1/N_p) * Σ ℓ(g(x_p_i), +1)`
- `\hRp^-(g)` = Empirical positive risk with negative label: `(1/N_p) * Σ ℓ(g(x_p_i), -1)`
- `\hRu^-(g)` = Empirical unlabeled risk: `(1/N_u) * Σ ℓ(g(x_u_i), -1)`
- `max{0, ...}` = Non-negative correction

**Description** (from paper):
> Based on this key observation, we propose a \emph{non-negative risk estimator} for PU learning

**Key Property**: Guaranteed to stay non-negative (prevents overfitting with flexible models)

**Relationship to uPU**:
```
\tRpu(g) = \pip\hRp^+(g) + max{0, \hRu^-(g) - \pip\hRp^-(g)}
         = max{\pip\hRp^+(g), \hRpu(g)}
```

When `\hRu^-(g) - \pip\hRp^-(g) < 0` (negative risk term), nnPU clamps it to 0.

---

## Implementation Notes

### For Comparison with Current Implementation

1. **Sign Check**: Verify all terms have correct signs (especially the `-\pip\hRp^-(g)` term in uPU)
2. **Max Operation**: nnPU must use `max{0, ...}` on the negative risk component
3. **Prior Usage**: All three methods use π_p (class prior)
4. **Expectation Approximation**: Verify empirical averages are computed correctly

### Expected Behavior

- **PN**: Straightforward weighted sum of positive and negative risks
- **uPU**: Can become negative during training (subtracts positive-as-negative risk)
- **nnPU**: Never negative (clamps to 0), more robust against overfitting

### Gradient Considerations

From paper discussion (need to verify in implementation):
- **nnPU gradient flow**: When risk goes negative, gradient should not flow through the max operation
- This typically requires `.detach()` in PyTorch on the comparison term

---

## Reference for Analysis Tasks

- **T019**: Compare `src/pu_learning/losses/pn.py` against PN equation above
- **T020**: Compare `src/pu_learning/losses/upu.py` against uPU equation above
- **T021**: Compare `src/pu_learning/losses/nnpu.py` against nnPU equation above

**Next Steps**: Review reference implementations (kiryor, cimeister) to see how they implement these equations, then compare against current implementation.
