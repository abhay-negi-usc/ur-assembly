"""TILE ASSEMBLY -- a copy of coupler_marker_assemble with its own config. What is worth pinning is
that it IS separate: it runs from configs/tile_assembly.yaml, names its own run folder and log, and
that config still resolves a calibrated assembly and a catalogued tile."""

import inspect
import os

from urlab import config as C
from urlab import tool_frames
from urlab.apps import tile_assembly as ta
from urlab.apps.coupler_pick_assemble import AssembleCycle


def test_it_runs_from_its_own_config():
    assert "'tile_assembly'" in inspect.getsource(ta.main)
    cfg = C.load('tile_assembly')
    assert os.path.basename(cfg.get('_config_path')) == 'tile_assembly.yaml'


def test_it_is_the_marker_assemble_cycle_under_its_own_name():
    assert issubclass(ta.TileAssemblyCycle, AssembleCycle)
    assert ta.TileAssemblyCycle.RUN_NAME == 'tile_assembly'
    assert ta.log.name.endswith('tile-asm')


def test_its_config_resolves_an_assembly_and_a_catalogued_tile():
    cfg = C.load('tile_assembly')
    name, entry = ta.resolve_marker_assembly(cfg)
    assert entry['held_object'] in tool_frames.load_objects(cfg)
    assert ta.parse_transitions(cfg.get('transition_joints_deg')) is not None
