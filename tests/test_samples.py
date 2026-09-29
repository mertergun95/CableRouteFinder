"""Regressionstest mit den Beispielplänen in ``samples/`` (benötigt Tesseract)."""
import os
import shutil

import pytest

from cableroutefinder.klp import kuep_name_tokens, normalize_cable_id, normalize_element_token
from cableroutefinder.kuep import normalize_name

HERE = os.path.dirname(__file__)
KLP = os.path.join(HERE, "..", "samples", "10_15_SKL_04-500_KLP.pdf")
KUEP = os.path.join(HERE, "..", "samples", "10_16_SKL_A02_KUEP.pdf")

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract nicht installiert")


def test_normalize_cable_id():
    assert normalize_cable_id("$1307010") == "S1307010"
    assert normalize_cable_id("1307010") is None
    assert normalize_cable_id("1307010", allow_missing_prefix=True) == "S1307010"


def test_name_normalization():
    assert normalize_name("KS 1307000 600") == "KS 1307000"
    assert normalize_name("SK 13P 16") == "SK 13P16"
    assert normalize_name("1364.008/13G5008") == "13G4008/13G5008"
    assert normalize_element_token("63045") == "G3045"
    assert normalize_element_token("Ww25D") == "W25D"
    assert kuep_name_tokens("13W22/13G4004") == [frozenset({"W22", "G4004"})]
    assert kuep_name_tokens("13L3045Y") == [frozenset({"3045Y"})]
    assert kuep_name_tokens("KS 1307000") == [frozenset({"KS1307000"})]


@pytest.fixture(scope="module")
def result():
    from cableroutefinder.pipeline import analyze
    return analyze(KLP, KUEP, log=lambda *_: None)


@needs_tesseract
def test_kuep_cables(result):
    ids = {c.cable_id for c in result.kuep.cables}
    assert {"S1307010", "S1307505", "S1307800", "S1307910"} <= ids
    c = next(c for c in result.kuep.cables if c.cable_id == "S1307505")
    assert c.from_name == "KS 1307000"
    assert c.to_name == "13W22/13G4004"
    assert c.length_m == 110


@needs_tesseract
def test_klp_labels(result):
    assert len(result.labels) >= 55
    assert all(l.anchor is not None for l in result.labels)


@needs_tesseract
@pytest.mark.parametrize("cable_id,kuep_len", [("S1307502", 130), ("S1307505", 110), ("S1307514", 170),
                                              ("S1307800", 330)])
def test_route_lengths_plausible(result, cable_id, kuep_len):
    c = next(c for c in result.cables if c.cable_id == cable_id)
    assert c.route.polylines
    # Weglänge im Plan ist etwas kürzer als die KÜP-Länge (Reserven, Einführungen)
    assert 0.7 * kuep_len <= c.length_m <= 1.1 * kuep_len
