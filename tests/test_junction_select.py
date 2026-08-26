"""Slope-based junction selection: pick the TRANSITION, not the longest flat run.

The failure these pin is specific. `_longest_run_index` finds the cable and infers the connector
from whichever end of it borders a rise, so anything else thick enough to break the run -- a
strain-relief boot, a moulded sleeve, a wrapped marker -- splits the cable into unequal halves and
the junction lands on that intermediate feature instead of the connector.

Profiles here are synthetic and stated in cable diameters, so retuning a real cable cannot break
them; each test names the geometry it stands for.
"""

import numpy as np
import pytest

from urlab.perception import junction as J


def _profile(n, base, features):
    """A flat cable of diameter `base` with (start, stop, diameter) features laid on it."""
    d = np.full(n, float(base))
    for lo, hi, val in features:
        d[lo:hi] = float(val)
    return d


def test_an_intermediate_feature_does_not_win_over_the_connector():
    """The bug. A boot/sleeve between the connector and the cable splits the constant run; the
    longest half then borders the SLEEVE, so the old selector put the junction there."""
    # connector 43 near the start, a 27 sleeve after it, then a long plain cable
    dia = _profile(600, 12.0, [(20, 60, 43.0), (150, 260, 27.0)])
    k, info = J.find_junction_index(dia, select='slope')
    assert info['thick_diameter'] == pytest.approx(43.0, abs=1.0), 'must lock onto the 43 feature'
    assert k < 150, f'junction should sit against the connector, not the sleeve (got {k})'


def test_a_connector_at_the_very_end_of_the_path_is_found():
    """The strand is traced tip to tip, so a plug at the end peaks at sample 0 or n-1 with only
    one neighbour. An interior-only peak scan drops the commonest geometry there is."""
    for feat, expect_right in (((0, 40, 40.0), False), ((560, 600, 40.0), True)):
        dia = _profile(600, 12.0, [feat])
        cands = J.connector_candidates(dia)
        assert cands, f'no candidate for a connector at the path end: {feat}'
        assert cands[0]['connector_on_right'] is expect_right


def test_a_cable_with_a_plug_at_each_end_reports_two_junctions():
    """What multi-junction detection is for: both ends, each pointing into its own connector."""
    dia = _profile(800, 12.0, [(0, 45, 38.0), (755, 800, 30.0)])
    cands = J.connector_candidates(dia)
    assert len(cands) == 2, [c['peak_idx'] for c in cands]
    left, right = cands
    assert left['connector_on_right'] is False and right['connector_on_right'] is True
    # each junction sits on the CABLE side of its own feature, i.e. between the two
    assert left['index'] < right['index']
    assert 0 < left['index'] < 200 and 600 < right['index'] < 800


def test_the_junction_goes_on_the_cable_side_flank():
    """A connector has a slope on BOTH sides. Taking the steeper one alone puts the junction on
    the far side of the connector from the cable, which inverts connector_on_right and therefore
    cable_side_mask -- it reads as a good result while pointing the wrong way."""
    dia = _profile(600, 12.0, [(40, 90, 40.0)])          # connector near the start
    cands = J.connector_candidates(dia)
    assert len(cands) == 1
    c = cands[0]
    assert c['index'] > c['peak_idx'], 'the cable is to the RIGHT, so the junction must be too'
    assert c['connector_on_right'] is False


def test_a_crossing_spike_is_not_mistaken_for_a_connector():
    """Two fused strands read wide because the ray runs through both. Untreated, that spike is
    indistinguishable from a connector -- and its apparent width grows as 1/sin of the angle."""
    dia = _profile(600, 12.0, [(300, 330, 45.0)])
    crossing = np.zeros(600, bool)
    crossing[295:335] = True
    assert not J.connector_candidates(dia, crossing), 'a mostly-crossing spike is an artefact'


def test_a_real_connector_survives_a_crossing_over_its_apex():
    """Rejecting by APEX alone loses real connectors: where a cable lies across its own plug the
    flagged samples can include the peak while the rest of the feature is honest."""
    dia = _profile(600, 12.0, [(20, 70, 40.0)])
    crossing = np.zeros(600, bool)
    crossing[42:48] = True                                # a few flagged samples over the apex
    cands = J.connector_candidates(dia, crossing)
    assert cands, 'the whole feature must not be discarded for a flagged apex'
    assert cands[0]['thick'] == pytest.approx(40.0, abs=1.5)


def test_a_flat_profile_falls_back_and_keeps_the_old_shape():
    """No connector-like feature anywhere (a floor seam, a bare cable). The caller still gets the
    same keys it always did rather than an empty result it would have to special-case."""
    dia = _profile(400, 12.0, [])
    assert J.connector_candidates(dia) == []
    k, info = J.find_junction_index(dia, select='slope')
    assert 0 <= k < 400
    for key in ('junction_k', 'cable_baseline', 'thin_diameter', 'thick_diameter', 'contrast',
                'connector_on_right'):
        assert key in info


def test_peak_min_decides_what_counts_as_a_connector():
    """The knob that also decides HOW MANY junctions a cable reports."""
    dia = _profile(600, 12.0, [(40, 90, 40.0), (300, 340, 20.0)])   # 3.3x and 1.7x the cable
    assert len(J.connector_candidates(dia, peak_min=1.5)) == 2
    assert len(J.connector_candidates(dia, peak_min=1.8)) == 1


def test_the_original_selector_is_still_reachable():
    """Kept so a past run can be reproduced, and as a fallback for a scene that defeats slope."""
    dia = _profile(600, 12.0, [(0, 40, 40.0)])
    k_slope, _ = J.find_junction_index(dia, select='slope')
    k_run, _ = J.find_junction_index(dia, select='longest_run')
    assert isinstance(k_run, int) and 0 <= k_run < 600
    assert isinstance(k_slope, int)
    with pytest.raises(ValueError, match='slope|longest_run'):
        J.find_junction_index(dia, select='nonsense')


def test_compute_junction_reports_every_junction_it_found():
    """End to end on a synthetic mask: the top-level keys still describe the thickest connector,
    and `junctions` carries all of them so a caller can choose."""
    cv2 = pytest.importorskip('cv2')
    img = np.zeros((300, 1000), np.uint8)
    cv2.line(img, (60, 150), (940, 150), 255, 13)
    cv2.circle(img, (70, 150), 30, 255, -1)               # the thicker connector
    cv2.circle(img, (930, 150), 22, 255, -1)
    res = J.compute_junction(img.astype(bool), work_dim=1024)
    assert res is not None and res['junctions'] and len(res['junctions']) == 2
    best = max(res['junctions'], key=lambda j: j['connector_diameter_px'])
    assert res['junction'] == pytest.approx(best['junction'], abs=1e-6), \
        'the top-level junction must be the thickest of the reported ones'
    assert res['junctions'][0]['junction'][0] < res['junctions'][1]['junction'][0]
    assert [j['connector_on_right'] for j in res['junctions']] == [False, True]
