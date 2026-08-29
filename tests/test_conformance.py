from vsl_core.conformance.suite import run_conformance_suite
from vsl_maf import MAFAdapter


def test_conformance_suite_passes():
    failures = run_conformance_suite(MAFAdapter())
    assert failures == [], "\n".join(failures)
