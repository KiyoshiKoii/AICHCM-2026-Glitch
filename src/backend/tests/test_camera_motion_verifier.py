from backend.services.camera_motion_verifier import CameraMotionVerifier


def test_camera_motion_verifier_scores_matching_upward_transition_higher() -> None:
    constraints = CameraMotionVerifier._constraints(
        "Máy quay chéo lên rồi chuyển cảnh sang nguyên liệu khác"
    )
    matching = {
        "up_direction": 0.9,
        "down_direction": -0.9,
        "diagonal_fraction": 0.9,
        "static_fraction": 0.0,
        "transition_score": 1.0,
    }
    opposite = {
        "up_direction": -0.9,
        "down_direction": 0.9,
        "diagonal_fraction": 0.1,
        "static_fraction": 0.0,
        "transition_score": 0.0,
    }

    assert CameraMotionVerifier._score_metrics(matching, constraints) > (
        CameraMotionVerifier._score_metrics(opposite, constraints)
    )


def test_camera_motion_verifier_ignores_events_without_camera_constraint() -> None:
    assert not CameraMotionVerifier.has_explicit_camera_constraint(
        "Người đầu bếp đặt miếng măng tây vào chảo dầu"
    )
