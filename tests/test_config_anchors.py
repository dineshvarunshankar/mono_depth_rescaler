from proto.config import load


def test_anchors_deployment_defaults():
    a = load().anchors
    assert a.use_tof is True
    assert a.tof_pipe == "tof"
    assert a.tof_max_points == 500
    assert 0 <= a.tof_confidence_min <= 255
    assert a.balance_mode == "off"
    assert 0 <= a.balance_vio_fraction <= 1
    assert a.balance_vio_sufficient > 0
    assert a.balance_tof_fallback > 0


def test_rescale_args_all_numeric():
    # PyYAML parses exponent literals without a sign (1.0e8) as STRINGS; guard it.
    for k, v in load().rescale.args.items():
        assert isinstance(v, (int, float)), f"rescale.args.{k}={v!r} is {type(v).__name__}, not numeric"


def test_deployment_profiles_resolve_atomically():
    openvins = load(profile="openvins", fov="crop")
    assert openvins.vio.pipe == "ov_extended"
    assert openvins.anchors.tof_max_points == 500
    assert openvins.anchors.projection == "world_pose"

    qvio = load(profile="qvio", fov="stretch")
    assert qvio.vio.pipe == "qvio_extended"
    assert qvio.anchors.tof_max_points == 500
    assert qvio.inference.fov == "stretch"
