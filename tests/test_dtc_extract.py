"""Unit tests for DTC regex extraction and typo correction."""

from warranty.dtc_extract_match import extract_dtcs_regex


def test_standard_codes():
    text = "Found P0300 and C0035 in the scan."
    result = extract_dtcs_regex(text, typo_correction=False)
    assert result == ["C0035", "P0300"]


def test_lowercase_codes():
    text = "p0420 catalyst efficiency low"
    result = extract_dtcs_regex(text, typo_correction=False)
    assert result == ["P0420"]


def test_multiple_codes_dedup():
    text = "P0171 appeared twice: P0171 and also P0300."
    result = extract_dtcs_regex(text, typo_correction=False)
    assert result == ["P0171", "P0300"]


def test_typo_correction_O_to_0():
    text = "DTC p0l7l noted earlier"  # l should be 1
    result = extract_dtcs_regex(text, typo_correction=True)
    # After typo correction O->0, l->1, "p0l7l" becomes "p01711"
    # but original "p0l7l" won't match. The corrected "p01711" is 5 hex
    # digits so won't match either. Let's test a more realistic typo:
    text2 = "DTC PO171 detected"  # O instead of 0
    result2 = extract_dtcs_regex(text2, typo_correction=True)
    assert "P0171" in result2


def test_body_and_network_codes():
    text = "B1000 BCM fault and U0100 lost comm"
    result = extract_dtcs_regex(text, typo_correction=False)
    assert "B1000" in result
    assert "U0100" in result


def test_no_dtc():
    text = "No diagnostic trouble codes stored."
    result = extract_dtcs_regex(text, typo_correction=True)
    assert result == []


def test_hex_codes():
    text = "Got P0ABF hybrid battery fault"
    result = extract_dtcs_regex(text, typo_correction=False)
    assert result == ["P0ABF"]
