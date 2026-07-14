"""Reusable robot behaviours, decoupled from any one demo.

These are the "several modules and skills which can be abstracted out into common functions":

    servo   -- align / centre / approach a marker; closed-loop PBVS
    scan    -- multi-view cable scan + connector-pose fusion
    pick    -- grasp geometry, the counts-based grasp check, recovery
    touch   -- probe the connector height by contact
    insert  -- stand-off, compliant chunked insertion, multi-step retract

A demo is a short script that composes these; they do not know about each other, so any two
combine (scan + insert, scan + touch + insert) without one being a base class of the other.
"""

from . import insert, pick, scan, servo, touch

__all__ = ['servo', 'scan', 'pick', 'touch', 'insert']
