"""Authenticate an actual-main normalization promotion using its frozen predecessor."""
from pathlib import Path
from attention_normalize_stage_gate_v1 import (R, ROOT, DREF, STAGE, STAGE_DATA,
    PRIMITIVE_PATH, authenticate_stage, assert_active_stage,
    verify_dispatch_substitution, js, sha, package)

SNAPSHOT = R / 'experimental/revisions/before-main-attention-normalize-v1'


def historical_path(path):
    path = Path(path)
    if path.parent == ROOT / 'backend/nr_backend':
        return SNAPSHOT / 'nr_backend' / path.name
    if path == R / 'backend-checkpoint.json':
        return SNAPSHOT / 'backend-checkpoint.json'
    return path


def check_historical_files(files):
    for name, digest in files.items():
        assert sha(historical_path(name)) == digest, name


def authenticate_promotion():
    """Historical-capable identity check; does not require active main equality."""
    gates = authenticate_stage(SNAPSHOT / 'nr_backend')
    manifest_path = STAGE_DATA / 'manifest.json'
    manifest = js(manifest_path)
    promotion_path = DREF / 'attention-normalize-backend-promotion-v1.json'
    promotion = js(promotion_path)
    assert promotion['promoted'] and promotion['copied_byte_identically_from_validated_stage']
    assert promotion['baseline_sources'] == manifest['baseline_sources']
    assert promotion['current_sources'] == manifest['staged_sources'] == package(STAGE / 'nr_backend')
    assert promotion['changed_files'] == ['attention.py'] and promotion['added_files'] == ['triton_attention_normalize.py']
    assert promotion['promotion_script_sha256'] == sha(R / 'promote_attention_normalize_backend_v1.py')
    assert promotion['stage_manifest_sha256'] == sha(manifest_path)
    assert promotion['primitive_sha256'] == sha(PRIMITIVE_PATH)
    assert promotion['stage_cpu_sha256'] == sha(STAGE_DATA / 'cpu-validation-v1.json')
    assert promotion['previous_promotion_sha256'] == sha(R / 'attention-exponential-backend-promotion-v1.json')
    assert Path(promotion['baseline_snapshot']) == SNAPSHOT
    saved_path = SNAPSHOT / 'manifest.json'
    saved = js(saved_path)
    assert sha(saved_path) == promotion['baseline_snapshot_manifest_sha256']
    assert saved['sources'] == manifest['baseline_sources'] == package(SNAPSHOT / 'nr_backend')
    assert all(sha(SNAPSHOT / n) == h for n, h in saved['saved_files'].items())
    suite_path = STAGE_DATA / 'full-regression-suite-v1.json'
    audit_path = STAGE_DATA / 'full-regression-saved-audit-v1.json'
    suite, audit = js(suite_path), js(audit_path)
    assert promotion['full_stage_audit_sha256'] == sha(audit_path)
    assert promotion['full_stage_suite_sha256'] == audit['suite_sha256'] == sha(suite_path)
    assert suite['all_validators_passed'] and suite['comparisons'] == 55
    assert audit['all_byte_equal'] and audit['persisted_rgb_and_private_reread_equal'] and audit['full_frame_comparisons'] == 55
    assert audit['staged_sources'] == manifest['staged_sources'] and audit['baseline_sources'] == manifest['baseline_sources']
    assert audit['audit_script_sha256'] == sha(R / 'audit_attention_normalize_full_regressions_v1.py')
    check_historical_files(suite['frozen_artifacts'])
    check_historical_files(audit['gate_artifacts'])
    for path in (Path(__file__), promotion_path, saved_path, suite_path, audit_path):
        gates[str(path)] = sha(path)
    return gates


def authenticate_main():
    gates = authenticate_promotion()
    assert package(ROOT / 'backend/nr_backend') == js(STAGE_DATA / 'manifest.json')['staged_sources']
    return gates
