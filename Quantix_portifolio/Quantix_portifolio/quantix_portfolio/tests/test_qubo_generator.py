import numpy as np

from agents.qubo_generator import QUBOGenerator


def test_qubo_generator_shapes_and_terms() -> None:
    generator = QUBOGenerator()
    expected_returns = np.array([0.1, 0.2, 0.05])
    covariance = np.array(
        [
            [0.02, 0.01, 0.0],
            [0.01, 0.03, 0.01],
            [0.0, 0.01, 0.015],
        ]
    )
    signs = np.array([1.0, 1.0, -1.0])

    result = generator.generate(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_lambda=1.2,
        cardinality_limit=2,
        alpha_signs=signs,
    )

    assert result.q_matrix.shape == (3, 3)
    assert len(result.qubo_dict) > 0
    assert result.cardinality_target == 2
