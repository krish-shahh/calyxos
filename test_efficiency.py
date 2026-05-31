"""
calyxos efficiency test: proves incremental recomputation saves work
in a simulated Claude Code subagent pipeline.

Test 1 (linear):  codebase -> parse -> analysis -> plan -> output
Test 2 (fan-out): 4 independent module branches merging to one report

Each derived node sleeps 200ms to simulate expensive work.
"""

import time
import traceback

SLEEP_SEC = 0.2  # 200ms per node
REPORT_LINES: list[str] = []


def log(msg: str = "") -> None:
    REPORT_LINES.append(msg)
    print(msg)


def expensive(val, _label: str = ""):
    """Simulate expensive work (200ms sleep)."""
    time.sleep(SLEEP_SEC)
    return val


# =====================================================================
# calyxos pipelines
# =====================================================================

from calyxos import node, NodeFlag, set_value


# ── Linear pipeline ──────────────────────────────────────────────────

class LinearPipeline:
    def __init__(self):
        self.compute_log: list[str] = []

    @node(NodeFlag.CAN_SET)
    def codebase(self) -> str:
        return "repo-v1"

    @node()
    def parse(self) -> str:
        self.compute_log.append("parse")
        return expensive(f"parsed({self.codebase()})")

    @node()
    def analysis(self) -> str:
        self.compute_log.append("analysis")
        return expensive(f"analyzed({self.parse()})")

    @node()
    def plan(self) -> str:
        self.compute_log.append("plan")
        return expensive(f"planned({self.analysis()})")

    @node()
    def output(self) -> str:
        self.compute_log.append("output")
        return expensive(f"output({self.plan()})")


# ── Fan-out pipeline (4 branches x 3 transforms + 1 merge) ──────────

class FanOutPipeline:
    """
    4 independent module inputs, each feeding a 3-node branch,
    all converging to a single report node.

        api ─── parse_api ─── analyze_api ─── plan_api ───┐
       auth ─── parse_auth ── analyze_auth ── plan_auth ──┤
         db ─── parse_db ──── analyze_db ──── plan_db ────┤── report
         ui ─── parse_ui ──── analyze_ui ──── plan_ui ────┘

    13 derived nodes total. Changing one input recomputes only
    its 3-node branch + the merge = 4 nodes. The other 9 are cached.
    """

    def __init__(self):
        self.compute_log: list[str] = []

    # ── Inputs (settable, no compute cost) ───────────────────────
    @node(NodeFlag.CAN_SET)
    def api(self) -> str: return "api-v1"
    @node(NodeFlag.CAN_SET)
    def auth(self) -> str: return "auth-v1"
    @node(NodeFlag.CAN_SET)
    def db(self) -> str: return "db-v1"
    @node(NodeFlag.CAN_SET)
    def ui(self) -> str: return "ui-v1"

    # ── API branch ───────────────────────────────────────────────
    @node()
    def parse_api(self) -> str:
        self.compute_log.append("parse_api")
        return expensive(f"parsed({self.api()})")

    @node()
    def analyze_api(self) -> str:
        self.compute_log.append("analyze_api")
        return expensive(f"analyzed({self.parse_api()})")

    @node()
    def plan_api(self) -> str:
        self.compute_log.append("plan_api")
        return expensive(f"planned({self.analyze_api()})")

    # ── Auth branch ──────────────────────────────────────────────
    @node()
    def parse_auth(self) -> str:
        self.compute_log.append("parse_auth")
        return expensive(f"parsed({self.auth()})")

    @node()
    def analyze_auth(self) -> str:
        self.compute_log.append("analyze_auth")
        return expensive(f"analyzed({self.parse_auth()})")

    @node()
    def plan_auth(self) -> str:
        self.compute_log.append("plan_auth")
        return expensive(f"planned({self.analyze_auth()})")

    # ── DB branch ────────────────────────────────────────────────
    @node()
    def parse_db(self) -> str:
        self.compute_log.append("parse_db")
        return expensive(f"parsed({self.db()})")

    @node()
    def analyze_db(self) -> str:
        self.compute_log.append("analyze_db")
        return expensive(f"analyzed({self.parse_db()})")

    @node()
    def plan_db(self) -> str:
        self.compute_log.append("plan_db")
        return expensive(f"planned({self.analyze_db()})")

    # ── UI branch ────────────────────────────────────────────────
    @node()
    def parse_ui(self) -> str:
        self.compute_log.append("parse_ui")
        return expensive(f"parsed({self.ui()})")

    @node()
    def analyze_ui(self) -> str:
        self.compute_log.append("analyze_ui")
        return expensive(f"analyzed({self.parse_ui()})")

    @node()
    def plan_ui(self) -> str:
        self.compute_log.append("plan_ui")
        return expensive(f"planned({self.analyze_ui()})")

    # ── Merge ────────────────────────────────────────────────────
    @node()
    def report(self) -> str:
        self.compute_log.append("report")
        parts = [self.plan_api(), self.plan_auth(),
                 self.plan_db(), self.plan_ui()]
        return expensive(f"report({'+'.join(parts)})")


# =====================================================================
# Naive pipelines (no caching -- recompute everything every time)
# =====================================================================

class NaiveLinear:
    def __init__(self, codebase: str = "repo-v1"):
        self._codebase = codebase
        self.compute_log: list[str] = []

    def run(self) -> str:
        self.compute_log = []
        cb = self._codebase  # no sleep for input

        self.compute_log.append("parse")
        parsed = expensive(f"parsed({cb})")

        self.compute_log.append("analysis")
        analyzed = expensive(f"analyzed({parsed})")

        self.compute_log.append("plan")
        planned = expensive(f"planned({analyzed})")

        self.compute_log.append("output")
        return expensive(f"output({planned})")


class NaiveFanOut:
    def __init__(self, api="api-v1", auth="auth-v1", db="db-v1", ui="ui-v1"):
        self.inputs = {"api": api, "auth": auth, "db": db, "ui": ui}
        self.compute_log: list[str] = []

    def run(self) -> str:
        self.compute_log = []
        plans = []
        for name in ["api", "auth", "db", "ui"]:
            inp = self.inputs[name]

            self.compute_log.append(f"parse_{name}")
            parsed = expensive(f"parsed({inp})")

            self.compute_log.append(f"analyze_{name}")
            analyzed = expensive(f"analyzed({parsed})")

            self.compute_log.append(f"plan_{name}")
            planned = expensive(f"planned({analyzed})")

            plans.append(planned)

        self.compute_log.append("report")
        return expensive(f"report({'+'.join(plans)})")


# =====================================================================
# Test runner
# =====================================================================

def run_test(title, cx_pipeline, cx_run, cx_mutate, cx_run2,
             nv_cold, nv_mutated, derived_count):
    """Run cold-start, mutation, and cache-hit tests for one pipeline shape."""
    errors: list[str] = []

    log(f"{'=' * 70}")
    log(title)
    log(f"{'=' * 70}")
    log()

    # ── Phase 1: Cold start ──────────────────────────────────────
    log(f"{'— Phase 1: Cold start ':—<70}")
    log()

    cx_pipeline.compute_log = []
    t0 = time.perf_counter()
    try:
        r_cx_1 = cx_run()
    except Exception as e:
        errors.append(f"calyxos cold: {e}\n{traceback.format_exc()}")
        r_cx_1 = f"ERROR: {e}"
    t_cx_cold = time.perf_counter() - t0
    cx_cold_log = list(cx_pipeline.compute_log)

    t0 = time.perf_counter()
    try:
        r_nv_1 = nv_cold.run()
    except Exception as e:
        errors.append(f"naive cold: {e}\n{traceback.format_exc()}")
        r_nv_1 = f"ERROR: {e}"
    t_nv_cold = time.perf_counter() - t0
    nv_cold_log = list(nv_cold.compute_log)

    log(f"  calyxos: {len(cx_cold_log):>2} nodes computed  {t_cx_cold*1000:>7.0f}ms")
    log(f"  naive:   {len(nv_cold_log):>2} nodes computed  {t_nv_cold*1000:>7.0f}ms")
    log()

    # ── Phase 2: Mutation ────────────────────────────────────────
    log(f"{'— Phase 2: Mutate one input, rerun ':—<70}")
    log()

    cx_pipeline.compute_log = []
    t0 = time.perf_counter()
    try:
        cx_mutate()
        r_cx_2 = cx_run2()
    except Exception as e:
        errors.append(f"calyxos mutation: {e}\n{traceback.format_exc()}")
        r_cx_2 = f"ERROR: {e}"
    t_cx_mut = time.perf_counter() - t0
    cx_mut_log = list(cx_pipeline.compute_log)
    cx_mut_skipped = derived_count - len(cx_mut_log)

    t0 = time.perf_counter()
    try:
        r_nv_2 = nv_mutated.run()
    except Exception as e:
        errors.append(f"naive mutation: {e}\n{traceback.format_exc()}")
        r_nv_2 = f"ERROR: {e}"
    t_nv_mut = time.perf_counter() - t0
    nv_mut_log = list(nv_mutated.compute_log)

    log(f"  calyxos: {len(cx_mut_log):>2} nodes computed, {cx_mut_skipped:>2} skipped  {t_cx_mut*1000:>7.0f}ms")
    log(f"  naive:   {len(nv_mut_log):>2} nodes computed,  0 skipped  {t_nv_mut*1000:>7.0f}ms")
    log()

    # ── Phase 3: No-change rerun ─────────────────────────────────
    log(f"{'— Phase 3: No change, rerun (pure cache hit) ':—<70}")
    log()

    cx_pipeline.compute_log = []
    t0 = time.perf_counter()
    try:
        r_cx_3 = cx_run2()
    except Exception as e:
        errors.append(f"calyxos cache: {e}\n{traceback.format_exc()}")
        r_cx_3 = f"ERROR: {e}"
    t_cx_cache = time.perf_counter() - t0
    cx_cache_log = list(cx_pipeline.compute_log)

    nv_nochange = nv_mutated.__class__(**nv_mutated.inputs) if hasattr(nv_mutated, 'inputs') else NaiveLinear(nv_mutated._codebase)
    t0 = time.perf_counter()
    try:
        r_nv_3 = nv_nochange.run()
    except Exception as e:
        errors.append(f"naive cache: {e}\n{traceback.format_exc()}")
        r_nv_3 = f"ERROR: {e}"
    t_nv_cache = time.perf_counter() - t0
    nv_cache_log = list(nv_nochange.compute_log)

    log(f"  calyxos: {len(cx_cache_log):>2} nodes computed, {derived_count:>2} skipped  {t_cx_cache*1000:>7.1f}ms")
    log(f"  naive:   {len(nv_cache_log):>2} nodes computed,  0 skipped  {t_nv_cache*1000:>7.0f}ms")
    log()

    # ── Speedups ─────────────────────────────────────────────────
    sp_mut = t_nv_mut / t_cx_mut if t_cx_mut > 0 else float("inf")
    sp_cache = t_nv_cache / t_cx_cache if t_cx_cache > 0 else float("inf")

    return {
        "derived_count": derived_count,
        "cold": (len(cx_cold_log), t_cx_cold, len(nv_cold_log), t_nv_cold),
        "mutation": (len(cx_mut_log), cx_mut_skipped, t_cx_mut,
                     len(nv_mut_log), t_nv_mut, sp_mut),
        "cache": (len(cx_cache_log), derived_count, t_cx_cache,
                  len(nv_cache_log), t_nv_cache, sp_cache),
        "results": (r_cx_1, r_cx_2, r_cx_3, r_nv_1, r_nv_2, r_nv_3),
        "logs": (cx_cold_log, cx_mut_log, cx_cache_log),
        "errors": errors,
    }


def main() -> None:
    all_errors: list[str] = []

    log(f"{'#' * 70}")
    log("calyxos efficiency benchmark")
    log(f"{'#' * 70}")
    log(f"node cost: {int(SLEEP_SEC * 1000)}ms sleep per derived node")
    log()

    # ── Test 1: Linear pipeline (5 nodes, all in one chain) ──────

    lin = LinearPipeline()
    r1 = run_test(
        title="TEST 1: Linear pipeline  (codebase -> parse -> analysis -> plan -> output)",
        cx_pipeline=lin,
        cx_run=lin.output,
        cx_mutate=lambda: set_value(lin, "codebase", "repo-v2"),
        cx_run2=lin.output,
        nv_cold=NaiveLinear("repo-v1"),
        nv_mutated=NaiveLinear("repo-v2"),
        derived_count=4,
    )
    all_errors.extend(r1["errors"])

    # ── Test 2: Fan-out pipeline (4 branches x 3 + 1 merge) ─────

    fan = FanOutPipeline()
    r2 = run_test(
        title="TEST 2: Fan-out pipeline  (4 branches x 3 transforms + 1 merge = 13 derived nodes)",
        cx_pipeline=fan,
        cx_run=fan.report,
        cx_mutate=lambda: set_value(fan, "api", "api-v2"),
        cx_run2=fan.report,
        nv_cold=NaiveFanOut(),
        nv_mutated=NaiveFanOut(api="api-v2"),
        derived_count=13,
    )
    all_errors.extend(r2["errors"])

    # ── Combined comparison table ────────────────────────────────

    log()
    log(f"{'=' * 78}")
    log("COMBINED RESULTS")
    log(f"{'=' * 78}")
    log()

    hdr = f"{'scenario':<36} {'engine':<10} {'computed':>8} {'skipped':>8} {'time':>9} {'speedup':>8}"
    sep = "-" * len(hdr)
    log(hdr)
    log(sep)

    def row(label, engine, computed, skipped, t, speedup=""):
        log(f"{label:<36} {engine:<10} {computed:>8} {skipped:>8} {t:>9} {speedup:>8}")

    # Linear cold
    c = r1["cold"]
    row("linear: cold start", "calyxos", c[0], 0, f"{c[1]*1000:.0f}ms")
    row("", "naive", c[2], 0, f"{c[3]*1000:.0f}ms")
    log(sep)

    # Linear mutation
    m = r1["mutation"]
    row("linear: mutate codebase", "calyxos", m[0], m[1], f"{m[2]*1000:.0f}ms", f"{m[5]:.1f}x")
    row("", "naive", m[3], 0, f"{m[4]*1000:.0f}ms")
    log(sep)

    # Linear cache
    c = r1["cache"]
    row("linear: no-change rerun", "calyxos", c[0], c[1], f"{c[2]*1000:.1f}ms", f"{c[5]:.0f}x")
    row("", "naive", c[3], 0, f"{c[4]*1000:.0f}ms")
    log(sep)

    # Fan-out cold
    c = r2["cold"]
    row("fan-out: cold start", "calyxos", c[0], 0, f"{c[1]*1000:.0f}ms")
    row("", "naive", c[2], 0, f"{c[3]*1000:.0f}ms")
    log(sep)

    # Fan-out mutation
    m = r2["mutation"]
    row("fan-out: mutate 1 of 4 inputs", "calyxos", m[0], m[1], f"{m[2]*1000:.0f}ms", f"{m[5]:.1f}x")
    row("", "naive", m[3], 0, f"{m[4]*1000:.0f}ms")
    log(sep)

    # Fan-out cache
    c = r2["cache"]
    row("fan-out: no-change rerun", "calyxos", c[0], c[1], f"{c[2]*1000:.1f}ms", f"{c[5]:.0f}x")
    row("", "naive", c[3], 0, f"{c[4]*1000:.0f}ms")
    log(sep)

    log()

    # ── Correctness checks ───────────────────────────────────────

    log("CORRECTNESS CHECKS")

    checks_passed = 0
    checks_total = 0

    def check(name, cond, detail=""):
        nonlocal checks_passed, checks_total
        checks_total += 1
        status = "PASS" if cond else "FAIL"
        if cond:
            checks_passed += 1
        msg = f"  [{status}] {name}"
        if detail and not cond:
            msg += f"  ({detail})"
        log(msg)
        if not cond:
            all_errors.append(f"check failed: {name} -- {detail}")

    # Linear checks
    l1, l2, l3 = r1["logs"]
    check("linear cold: all 4 derived nodes compute", len(l1) == 4,
          f"got {len(l1)}: {l1}")
    check("linear mutation: all 4 recompute (root changed)", len(l2) == 4,
          f"got {len(l2)}: {l2}")
    check("linear no-change: 0 recomputes", len(l3) == 0,
          f"got {len(l3)}: {l3}")

    r = r1["results"]
    check("linear results correct",
          r[0] == "output(planned(analyzed(parsed(repo-v1))))" and
          r[1] == "output(planned(analyzed(parsed(repo-v2))))" and
          r[2] == r[1],
          f"got: {r[0]}, {r[1]}, {r[2]}")

    # Fan-out checks
    f1, f2, f3 = r2["logs"]
    check("fan-out cold: all 13 derived nodes compute", len(f1) == 13,
          f"got {len(f1)}: {f1}")
    check("fan-out mutation: only 4 recompute (1 branch + merge)", len(f2) == 4,
          f"got {len(f2)}: {f2}")
    check("fan-out mutation: recomputed nodes are api branch + report",
          set(f2) == {"parse_api", "analyze_api", "plan_api", "report"},
          f"got: {f2}")
    check("fan-out no-change: 0 recomputes", len(f3) == 0,
          f"got {len(f3)}: {f3}")

    # Fan-out timing: mutation should be ~3x faster
    m = r2["mutation"]
    check("fan-out mutation speedup > 2x", m[5] > 2.0,
          f"got {m[5]:.1f}x")

    # Cache hit should be >100x faster
    c = r2["cache"]
    check("fan-out cache hit speedup > 100x", c[5] > 100,
          f"got {c[5]:.0f}x")

    log()
    log(f"checks: {checks_passed}/{checks_total} passed")
    log()

    # ── Errors ───────────────────────────────────────────────────

    if all_errors:
        log(f"{'=' * 70}")
        log("ERRORS / BUGS")
        log(f"{'=' * 70}")
        for i, err in enumerate(all_errors, 1):
            log(f"  {i}. {err}")
        log()
    else:
        log("no errors detected.")
        log()

    # ── Write report ─────────────────────────────────────────────

    report = "\n".join(REPORT_LINES) + "\n"
    with open("calyxos_test_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
