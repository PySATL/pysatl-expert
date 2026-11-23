# Statistics and procedure results

Catalog statistics are raw statistic values. A raw value is not a p-value or
decision. In model-free analysis, the caller selects distribution names and
statistic IDs explicitly. The RF path selects statistics from its validated
model schema plus any explicitly requested `additional_statistics`; this
special path does not calculate the entire registered catalog.

Additional procedures declare their IDs and required fields. A procedure can
return a raw statistic and, only when explicitly requested and permitted,
calibrated evidence such as a p-value or critical values. Fitted-parameter
raw statistics cannot use Criterion's simple-hypothesis calibrator. The
provider is responsible for the method's applicability; the pipeline validates
that the result matches the request but does not establish scientific validity.
Catalog procedures preserve their full `ProcedureResult` in
`AnalysisResult.procedure_results`, keyed by IDs such as `normal__ad`.

The current dependency is `pysatl-criterion` `main` (`v1.0.0`, commit
`3685d7e746fd7c175e18dc9b113f1db0d4fe3455`). API names and available statistics
are discovered dynamically from this installed revision.

`CriterionAdapter` reads parameter values, supported hypotheses, alternatives,
and versioned statistic configuration. Its `parameter_mode` distinguishes a
family hypothesis, explicitly known parameters, and parameters fitted from the
analyzed sample. For a family hypothesis, catalog evaluation uses the original
sample; where the statistic only accepts fixed parameters, the catalog supplies
fitted canonical coordinates for the raw statistic. Catalog results retain that
identity, including source, parameterization, configuration, alternative, and
parameter mode. Fitted catalog results also record `preprocessing`: the
preparation method and fitted parameter snapshot. An empty mapping means
original observations, as used by a direct `CriterionAdapter`. A failed fit does
not block statistics with a supported empty family hypothesis; fitted statistics
and legacy scalar criteria without an explicit `requires_fit = False` remain
unavailable. Distribution support checks still apply before calculation.
`GenericCriterion.calculate` remains a scalar wrapper over `evaluate`.

## Why catalog entries are skipped

These exclusions are already applied when the shared runtime catalog is built.
They therefore apply to model-free analysis as well as RF analysis; naming an
excluded ID explicitly does not enable it. `CriterionSelector` validates
requested IDs against `CRITERIA_SPECS`, so an excluded ID remains unavailable.
The public `analyze_sample` API has no override to re-enable excluded IDs;
explicit selection is limited to the active catalog.

The recorded reasons fall into distinct groups:

| Recorded reason | Statistic IDs |
|---|---|
| Exact duplicates in the completed raw dataset | `beta__lillie` = `beta__ks`; `lognormal__rj` = `lognormal__lg`; `normal__rj` = `normal__lg`; `student__lillie` = `student__ks`; `uniform__censored_stein_u` = `uniform__stein_u` for uncensored generated samples; `uniform__lillie` = `uniform__ks`; `weibull__lillie` = `weibull__ks` |
| Recorded numerical or domain risks | `beta__chi2_pearson` (non-finite when expected bin frequencies are zero); `beta__mode` (formula invalid if either fitted shape is at most one); `exponential__ww` (unstable ratio after fitted-location canonicalization); `gamma__mt` (non-finite or above float32 range in the balanced pilot); `lognormal__glb`, `lognormal__zwa`, `lognormal__zwc` (non-finite in distribution tails); `lognormal__sh` (overflow for larger samples); `normal__glb`, `normal__zwa`, `normal__zwc` (non-finite in distribution tails); `normal__sh` (overflow for larger samples); `weibull__lt2`, `weibull__lt3` (numerically explosive exponential transform); `weibull__ls` (above float32 range in the balanced pilot); `weibull__mt` (non-finite or above float32 range in the balanced pilot); `weibull__chi2_pearson` (explodes when equal-width tail bins have near-zero expected probability) |

The code-level legacy blacklist is `bhs`, `kl_sup`, `cq*`, `rs`, `ahs`, `hp`,
`independencenumber`, `cliquenumber`, `avgdegree`, `edgesnumber`, `maxdegree`,
and `connectedcomponents`. The catalog records these as `blacklisted`, and `cq*`
is separately marked `pending_validation`. To inspect the complete runtime list
for the installed Criterion revision, read `SKIPPED_CRITERIA`:

```python
from pysatl_expert.criteria.catalog import SKIPPED_CRITERIA

for entry in SKIPPED_CRITERIA:
    print(entry.full_code, entry.distribution, entry.short_code, entry.reason)
```

## Catalog and selectors

::: pysatl_expert.criteria.catalog
::: pysatl_expert.criteria.selectors.selector

## Configured Criterion adapter

::: pysatl_expert.criteria.criterion_adapter

## SciPy procedure adapter

::: pysatl_expert.criteria.scipy_adapter
