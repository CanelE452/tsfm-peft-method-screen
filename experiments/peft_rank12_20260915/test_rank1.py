def test_r1_fp64_semantics():
    from run_rank1 import semantic_cpu
    assert semantic_cpu()["group_isolation"]
