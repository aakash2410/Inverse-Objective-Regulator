from ior.gym.agents import CostMinimiserAgent
from ior.inference.structural import StructuralFeaturiser, STRUCTURAL_DIMS


def test_basis_is_fixed_and_purpose_independent():
    f = StructuralFeaturiser()
    a = f.decompose("minimise cost")
    b = f.decompose("answer faithfully")
    assert a.sub_goals == b.sub_goals == STRUCTURAL_DIMS


def test_deterministic():
    traj = CostMinimiserAgent().run(seed=3)
    f = StructuralFeaturiser()
    s1 = f.score_trajectory(traj.steps, f.decompose(traj.declared_purpose))
    s2 = f.score_trajectory(traj.steps, f.decompose(traj.declared_purpose))
    assert [s.scores for s in s1] == [s.scores for s in s2]


def test_scores_in_unit_interval():
    traj = CostMinimiserAgent().run(seed=1)
    f = StructuralFeaturiser()
    scored = f.score_trajectory(traj.steps, f.decompose(traj.declared_purpose))
    for s in scored:
        assert len(s.scores) == len(STRUCTURAL_DIMS)
        assert all(0.0 <= x <= 1.0 for x in s.scores)


def test_first_step_is_novel():
    traj = CostMinimiserAgent().run(seed=1)
    f = StructuralFeaturiser()
    scored = f.score_trajectory(traj.steps, f.decompose(traj.declared_purpose))
    # action_novelty (dim 0) is 1.0 on the first occurrence of an action.
    assert scored[0].scores[0] == 1.0
