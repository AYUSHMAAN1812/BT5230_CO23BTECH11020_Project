from ocsr_project.metrics import bootstrap_mean_interval, evaluate_pairs


def test_metrics_keep_invalid_predictions_in_denominator():
    summary, rows = evaluate_pairs(
        ["CCO", "c1ccccc1"], ["OCC", "invalid"], ["one", "two"]
    )
    assert summary["count"] == 2
    assert summary["valid_smiles_rate"] == 0.5
    assert summary["exact_isomeric_match_rate"] == 0.5
    assert rows[1]["tanimoto"] == 0.0


def test_bootstrap_interval_contains_constant_mean():
    low, high = bootstrap_mean_interval([0.5, 0.5, 0.5], seed=3, resamples=20)
    assert low == high == 0.5


def test_optional_invalid_reference_handling_is_conservative():
    summary, rows = evaluate_pairs(
        ["C#[N+][Kr]F", "CCO"],
        ["C#[N+][Kr]F", "OCC"],
        allow_invalid_references=True,
    )
    assert summary["count"] == 2
    assert summary["reference_parseable_rate"] == 0.5
    assert rows[0]["reference_parseable"] is False
    assert rows[0]["exact_isomeric_match"] is False
    assert rows[0]["tanimoto"] == 0.0
