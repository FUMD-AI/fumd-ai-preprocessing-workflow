"""
fumd_workflow
=============

Shared, importable logic for the FUMD-AI preprocessing workflow
(SUMO + OMNeT++ -> AI-ready cellular-handover dataset).

This package exists so that the core scientific logic used by the
pipeline notebooks (trajectory reconstruction and migration/handover
labeling) lives in one tested, versioned place instead of being
copy-pasted and re-edited inside notebook cells.

Authors: Cristina Bernad (ORCID: 0000-0001-9537-415X),
         Sonja Filiposka <sonja.filiposka@finki.ukim.mk> (ORCID: 0000-0003-0034-2855),
         Katja Gilly (ORCID: 0000-0002-8985-0639)
Copyright (c) 2026 Cristina Bernad, Sonja Filiposka, Katja Gilly
SPDX-License-Identifier: MIT
See LICENSE.txt (code, MIT) / LICENSE-CC-BY-4.0.txt (text/figures, CC BY 4.0)
in the repository root, and ro-crate-metadata.json for full FAIR metadata.
"""

from .labeling import build_trajectories, annotate_migrations

__all__ = ["build_trajectories", "annotate_migrations"]
