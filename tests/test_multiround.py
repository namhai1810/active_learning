from src.active_learning.multiround import (
    ActiveLearningCallbacks,
    ActiveLearningConfig,
    partition_ids_by_round,
    run_active_learning_loop,
)


def _verification(round_index):
    return [{"hypothesis_id": f"h{round_index}", "certified": True}]


def _gain(round_index, delta):
    return [
        {
            "hypothesis_id": f"h{round_index}",
            "m": 2,
            "delta_hat": delta,
            "radius": 0.1,
            "status": "estimated",
        }
    ]


def test_multiround_uses_updated_checkpoint_then_stops_on_nonpositive_u():
    seen = []

    def discover(round_index, checkpoint):
        seen.append((round_index, checkpoint["adapter_path"]))
        return [{"id": f"h{round_index}"}]

    callbacks = ActiveLearningCallbacks(
        discover=discover,
        verify=lambda r, checkpoint, hypotheses: _verification(r),
        generate=lambda r, checkpoint, hypotheses, verification: {"round": r},
        estimate_gains=lambda r, checkpoint, hypotheses, generated: _gain(
            r, 0.2 if r == 1 else -0.2
        ),
        estimate_costs=lambda r, checkpoint, hypotheses, verification, generated: (
            {f"h{r}": 1.0},
            {},
        ),
        update_student=lambda r, checkpoint, decision, generated: {
            "name_or_path": "student",
            "adapter_path": f"adapter-{r}",
        },
    )
    result = run_active_learning_loop(
        config=ActiveLearningConfig(max_rounds=4, max_budget=10, lambda_cost=0.05),
        initial_checkpoint={"name_or_path": "student", "adapter_path": None},
        callbacks=callbacks,
    )
    assert seen == [(1, None), (2, "adapter-1")]
    assert [row.decision.action for row in result.rounds] == ["continue", "stop"]
    assert result.stop_reason == "no_positive_optimistic_net_value"
    assert result.final_checkpoint["adapter_path"] == "adapter-1"
    assert result.total_cost == 1.0


def test_multiround_budget_stops_before_student_update():
    updates = []
    callbacks = ActiveLearningCallbacks(
        discover=lambda r, checkpoint: [{"id": "h1"}],
        verify=lambda r, checkpoint, hypotheses: _verification(1),
        generate=lambda r, checkpoint, hypotheses, verification: [],
        estimate_gains=lambda r, checkpoint, hypotheses, generated: _gain(1, 0.5),
        estimate_costs=lambda r, checkpoint, hypotheses, verification, generated: (
            {"h1": 4.0},
            {},
        ),
        update_student=lambda *args: updates.append(args) or {"adapter": "new"},
    )
    result = run_active_learning_loop(
        config=ActiveLearningConfig(max_rounds=3, max_budget=3, lambda_cost=0.0),
        initial_checkpoint={"adapter": "old"},
        callbacks=callbacks,
    )
    assert updates == []
    assert result.stop_reason == "budget_exhausted"
    assert result.total_cost == 0


def test_adaptive_eval_partitions_are_deterministic_disjoint_and_exhaustive():
    ids = [f"g{index}" for index in range(23)]
    first = partition_ids_by_round(ids, 5, seed=42, namespace="gain")
    second = partition_ids_by_round(ids, 5, seed=42, namespace="gain")
    assert first == second
    assert set().union(*(set(shard) for shard in first)) == set(ids)
    assert all(
        set(first[left]).isdisjoint(first[right])
        for left in range(5)
        for right in range(left + 1, 5)
    )
