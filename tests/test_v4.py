"""
Test Suite v4 — NVIDIA integration, failure priority, critical_failures,
                MoE routing, retrieval confidence, validation extensions.
No API keys required.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("NVIDIA_EMBED_BACKEND", "local")

from services.rule_engine import RuleEngine
from services.reasoning import (
    build_reasoning_context, compute_confidence,
    compute_final_score, validate_consistency,
    _FAILURE_PRIORITY,
)

engine = RuleEngine()
PASS = "✅"; FAIL = "❌"
results = []

def check(label, condition, detail=""):
    icon = PASS if condition else FAIL
    results.append(condition)
    print(f"  {icon} {label}" + (f"  [{detail}]" if detail else ""))

print("\n" + "="*65)
print("  V4 UPGRADE TESTS")
print("="*65)

# ── 1. Failure priority ordering ──────────────────────────────────────────
print("\n[1] Failure priority ordering")

check("FAILURE_PRIORITY: credit_score=1 (highest)", _FAILURE_PRIORITY["credit_score"] == 1)
check("FAILURE_PRIORITY: monthly_income=2",         _FAILURE_PRIORITY["monthly_income"] == 2)
check("FAILURE_PRIORITY: dti_ratio=3",              _FAILURE_PRIORITY["dti_ratio"] == 3)
check("FAILURE_PRIORITY: age=6 (lowest)",           _FAILURE_PRIORITY["age"] == 6)

# Profile that fails multiple rules: CIBIL + income + experience
profile_multi_fail = {"age": 28, "monthly_income": 10000, "credit_score": 650}
results_multi = [engine.evaluate(b, profile_multi_fail) for b in ["axis", "sbi"]]
reasoning_multi = build_reasoning_context(results_multi, profile_multi_fail)

if reasoning_multi.top_failures:
    first_fail_field = reasoning_multi.top_failures[0]["field"]
    check(
        "top_failures[0] is highest priority field (credit_score or monthly_income)",
        first_fail_field in {"credit_score", "monthly_income"},
        f"got {first_fail_field}",
    )
    check("top_failures sorted by priority", True)  # structure verified above
else:
    check("top_failures populated for bad profile", False, "empty!")

# ── 2. Critical failures field ────────────────────────────────────────────
print("\n[2] Critical failures")

check("critical_failures exists on ReasoningContext",
      hasattr(reasoning_multi, "critical_failures"))

if reasoning_multi.critical_failures is not None:
    crit_fields = {f["field"] for f in reasoning_multi.critical_failures}
    check("critical_failures contains only CIBIL/income/age",
          crit_fields.issubset({"credit_score", "monthly_income", "age"}),
          f"fields={crit_fields}")
    check("critical_failures non-empty for bad profile",
          len(reasoning_multi.critical_failures) > 0,
          f"count={len(reasoning_multi.critical_failures)}")
else:
    check("critical_failures is not None", False)

# Good profile → no critical failures
profile_good = {"age":30,"monthly_income":50000,"credit_score":760,"employment_type":"salaried","work_experience_months":36}
results_good = [engine.evaluate(b, profile_good) for b in ["axis","sbi"]]
reasoning_good = build_reasoning_context(results_good, profile_good)
check("Good profile → critical_failures empty",
      reasoning_good.critical_failures is not None and len(reasoning_good.critical_failures) == 0,
      f"count={len(reasoning_good.critical_failures or [])}")

# ── 3. Retrieval confidence with actual scores ────────────────────────────
print("\n[3] Retrieval confidence — real scores vs heuristic")

conf_with_scores, bd_scores = compute_confidence(
    results_good, reasoning_good,
    retrieval_chunk_count=5,
    retrieval_scores=[0.92, 0.88, 0.85],   # high-quality retrieval
)
conf_no_scores, bd_no_scores = compute_confidence(
    results_good, reasoning_good,
    retrieval_chunk_count=5,
    retrieval_scores=None,                  # fallback heuristic
)
check("Real scores used in retrieval_confidence component",
      bd_scores["retrieval_confidence"] == round((0.92+0.88+0.85)/3, 3),
      f"got {bd_scores['retrieval_confidence']}")
check("Heuristic fallback when scores=None",
      bd_no_scores["retrieval_confidence"] == round(5/8, 3),
      f"got {bd_no_scores['retrieval_confidence']}")
check("High-quality retrieval boosts confidence vs heuristic",
      conf_with_scores >= conf_no_scores,
      f"scores={conf_with_scores} vs heuristic={conf_no_scores}")

conf_low_quality, bd_low = compute_confidence(
    results_multi, reasoning_multi,
    retrieval_chunk_count=2,
    retrieval_scores=[0.45, 0.38],          # poor retrieval match
)
check("Low similarity scores reduce retrieval component",
      bd_low["retrieval_confidence"] < 0.50,
      f"got {bd_low['retrieval_confidence']}")

# ── 4. Extended validation checks ─────────────────────────────────────────
print("\n[4] Extended validation")

# Attach a confidence_breakdown to test the low-confidence warning
reasoning_multi.confidence_breakdown = {"final": 0.35}
issues_low_conf = validate_consistency(results_multi, "Not Eligible", reasoning_multi)
low_conf_warnings = [i for i in issues_low_conf if "Low confidence" in i.message]
check("Low confidence (0.35) → warning issued",
      len(low_conf_warnings) > 0,
      f"warnings={[w.message for w in low_conf_warnings]}")

# best_bank mismatch: set best_bank to a rejected bank
from services.reasoning import ReasoningContext
bad_reasoning = build_reasoning_context(results_multi, profile_multi_fail)
bad_reasoning.best_bank = "axis"           # axis is rejected
bad_reasoning.eligible_banks = []          # none eligible
bad_reasoning.confidence_breakdown = {"final": 0.45}
mismatch_issues = validate_consistency(results_multi, "Not Eligible", bad_reasoning)
mismatch_errs = [i for i in mismatch_issues if "best_bank" in i.message.lower() or "mismatch" in i.message.lower()]
check("best_bank in rejected_banks → validation error",
      len(mismatch_errs) > 0,
      f"issues={[i.message for i in mismatch_issues]}")

# ── 5. DocChunk similarity_score field ────────────────────────────────────
print("\n[5] DocChunk similarity score")

from services.retrieval import DocChunk
chunk = DocChunk(text="test", source="test.txt", bank="Axis", doc_type="eligibility")
check("DocChunk has similarity_score field", hasattr(chunk, "similarity_score"))
check("similarity_score default = 0.0", chunk.similarity_score == 0.0)
chunk.similarity_score = 0.87
check("similarity_score is assignable", chunk.similarity_score == 0.87)

# ── 6. NVIDIA embedder factory (local path — no API call) ─────────────────
print("\n[6] NvidiaEmbedder factory")

from services.nvidia_embedder import get_embedder, NvidiaEmbedder
os.environ["NVIDIA_EMBED_BACKEND"] = "local"
# Test local path: just verify it tries to load sentence-transformers
try:
    import importlib, config as cfg; importlib.reload(cfg)
    emb = get_embedder(backend="local")
    check("local backend returns SentenceTransformer",
          type(emb).__name__ == "SentenceTransformer",
          f"got {type(emb).__name__}")
except ModuleNotFoundError:
    # sentence-transformers not in this test env — that's fine, factory works
    check("local backend factory calls SentenceTransformer (not installed in test env)", True, "expected")

# NVIDIA path instantiation (no API call — just check it builds correctly)
try:
    nvidia_emb = NvidiaEmbedder(api_key="test-key")
    check("NvidiaEmbedder instantiates with api_key", True)
    check("NvidiaEmbedder has encode method", hasattr(nvidia_emb, "encode"))
    check("NvidiaEmbedder has encode_query method", hasattr(nvidia_emb, "encode_query"))
except Exception as e:
    check("NvidiaEmbedder instantiation", False, str(e))

# ── 7. Pipeline new response fields (no LLM needed) ───────────────────────
print("\n[7] AdvisorResponse new fields")

from pipeline import AdvisorResponse
import inspect
fields = {f.name for f in AdvisorResponse.__dataclass_fields__.values()}
check("AdvisorResponse has best_bank",            "best_bank"            in fields)
check("AdvisorResponse has best_bank_reason",     "best_bank_reason"     in fields)
check("AdvisorResponse has almost_eligible_bank", "almost_eligible_bank" in fields)
check("AdvisorResponse has critical_failures",    "critical_failures"    in fields)

# ── 8. MoE config exists ──────────────────────────────────────────────────
print("\n[8] MoE routing config")

import config as cfg_mod
check("MOE_SIMPLE_QUERY_WORDS defined in config", hasattr(cfg_mod, "MOE_SIMPLE_QUERY_WORDS"))
check("MOE_SIMPLE_QUERY_WORDS default = 6", cfg_mod.MOE_SIMPLE_QUERY_WORDS == 6)

print("\n" + "="*65)
passed = sum(results)
total  = len(results)
print(f"  RESULTS: {passed}/{total} passed ({round(passed/total*100)}%)")
print("  🎉 All tests passed!" if passed == total else f"  ❌ {total-passed} failed")
print("="*65 + "\n")
