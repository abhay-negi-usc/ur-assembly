"""compute.device: the ONE yaml knob for where torch models run.

Three hardcoded-cuda crashes surfaced one at a time when this codebase first ran on a CPU-only
laptop -- each hundreds of layers deep in the first inference. The knob exists so switching a
machine between GPU and CPU (or sharing configs across hosts) is a config edit, not a debugging
session. These tests pin the resolution rules without loading the model.
"""

import pytest

from urlab import config as C
from urlab.perception.sam3_backend import resolve_device


def _cuda_available():
    import torch
    return torch.cuda.is_available()


def test_auto_resolves_to_a_real_device():
    dev = resolve_device('auto')
    assert dev == ('cuda' if _cuda_available() else 'cpu')
    assert resolve_device(None) == dev, 'unset must behave as auto'
    assert resolve_device(' AUTO ') == dev, 'whitespace/case must not matter'


def test_cpu_always_wins():
    """Force CPU even on a GPU host -- the GPU may be owned by other software."""
    assert resolve_device('cpu') == 'cpu'


def test_cuda_is_a_requirement_not_a_wish():
    """Asking for cuda on a host without it must fail AT CONSTRUCTION with a message that names
    the knob -- not crash deep inside the first inference."""
    if _cuda_available():
        assert resolve_device('cuda') == 'cuda'
    else:
        with pytest.raises(RuntimeError, match='compute.device'):
            resolve_device('cuda')


def test_typos_are_rejected():
    with pytest.raises(ValueError, match="'auto', 'cpu' or 'cuda'"):
        resolve_device('gpu')


def test_the_knob_lives_in_common_yaml():
    """Machine facts live in _common.yaml; every config must inherit compute.device."""
    for name in ('bnc_assembly', 'cable_pick_place', 'cartesian'):
        cfg = C.load(name)
        assert cfg.get_path('compute.device') == 'auto', name
    cfg = C.load('bnc_assembly', ['compute.device=cpu'])
    assert cfg.get_path('compute.device') == 'cpu', '--set must reach it'
