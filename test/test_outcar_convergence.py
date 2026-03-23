"""Tests for tools4vasp.outcar_convergence."""

import gzip
from unittest.mock import patch

import pytest

from tools4vasp.outcar_convergence import (
    _find_outcar,
    _parse_potcar_poscar_elements,
    check_ionic_convergence,
    check_outcar,
    check_potcar_poscar_alignment,
    check_scf_convergence_per_step,
    run,
)
from conftest import (
    OUTCAR_ALIGNED_CONVERGED,
    OUTCAR_MISMATCHED_CONVERGED,
    OUTCAR_MULTI_STEP_CONVERGED,
    OUTCAR_MULTI_STEP_PARTIAL,
    OUTCAR_ONE_STEP_CONVERGED,
    OUTCAR_ONE_STEP_CONVERGED_V6,
    OUTCAR_ONE_STEP_SCF_FAILED,
    _POTCAR_POSCAR_HEADER_ALIGNED,
    _POTCAR_POSCAR_HEADER_LEADING_SPACE,
    _POTCAR_POSCAR_HEADER_MISMATCHED,
    _POTCAR_POSCAR_HEADER_PAW_SUFFIX,
    _POTCAR_POSCAR_HEADER_SINGLE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_outcar(tmp_path, content, filename="OUTCAR"):
    p = tmp_path / filename
    if filename.endswith(".gz"):
        with gzip.open(p, "wt") as fh:
            fh.write(content)
    else:
        p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# _find_outcar
# ---------------------------------------------------------------------------

class TestFindOutcar:
    def test_direct_file_path(self, tmp_path):
        p = _write_outcar(tmp_path, "x")
        assert _find_outcar(str(p)) == str(p)

    def test_direct_gz_path(self, tmp_path):
        p = _write_outcar(tmp_path, "x", "OUTCAR.gz")
        assert _find_outcar(str(p)) == str(p)

    def test_directory_finds_outcar(self, tmp_path):
        p = _write_outcar(tmp_path, "x")
        assert _find_outcar(str(tmp_path)) == str(p)

    def test_directory_finds_gz_when_no_plain(self, tmp_path):
        p = _write_outcar(tmp_path, "x", "OUTCAR.gz")
        assert _find_outcar(str(tmp_path)) == str(p)

    def test_directory_prefers_plain_over_gz(self, tmp_path):
        plain = _write_outcar(tmp_path, "x", "OUTCAR")
        _write_outcar(tmp_path, "x", "OUTCAR.gz")
        assert _find_outcar(str(tmp_path)) == plain

    def test_missing_returns_none(self, tmp_path):
        assert _find_outcar(str(tmp_path)) is None

    def test_nonexistent_path_returns_none(self, tmp_path):
        assert _find_outcar(str(tmp_path / "nowhere")) is None


# ---------------------------------------------------------------------------
# check_scf_convergence_per_step
# ---------------------------------------------------------------------------

class TestCheckScfConvergencePerStep:
    def test_single_step_converged(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        result = check_scf_convergence_per_step(p)
        assert result == [True]

    def test_single_step_failed(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_SCF_FAILED)
        result = check_scf_convergence_per_step(p)
        assert result == [False]

    def test_multi_step_partial(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_MULTI_STEP_PARTIAL)
        result = check_scf_convergence_per_step(p)
        assert result == [True, False, True]

    def test_multi_step_all_converged(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_MULTI_STEP_CONVERGED)
        result = check_scf_convergence_per_step(p)
        assert result == [True, True]

    def test_empty_outcar_returns_empty_list(self, tmp_path):
        p = _write_outcar(tmp_path, "no iteration markers here\n")
        assert check_scf_convergence_per_step(p) == []

    def test_vasp6_scf_string_recognised(self, tmp_path):
        # VASP 6.x uses "reached required accuracy - stopping SCF-cycle"
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED_V6)
        result = check_scf_convergence_per_step(p)
        assert result == [True]

    def test_gz_file_is_read_transparently(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED, "OUTCAR.gz")
        result = check_scf_convergence_per_step(p)
        assert result == [True]

    def test_length_matches_number_of_steps(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_MULTI_STEP_PARTIAL)
        result = check_scf_convergence_per_step(p)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# check_ionic_convergence
# ---------------------------------------------------------------------------

class TestCheckIonicConvergence:
    def test_converged_single_step(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        assert check_ionic_convergence(p) is True

    def test_not_converged_scf_failed(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_SCF_FAILED)
        assert check_ionic_convergence(p) is False

    def test_not_converged_multi_step_partial(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_MULTI_STEP_PARTIAL)
        assert check_ionic_convergence(p) is False

    def test_converged_multi_step(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_MULTI_STEP_CONVERGED)
        assert check_ionic_convergence(p) is True

    def test_gz_file_is_read_transparently(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED, "OUTCAR.gz")
        assert check_ionic_convergence(p) is True


# ---------------------------------------------------------------------------
# check_outcar
# ---------------------------------------------------------------------------

class TestCheckOutcar:
    def test_returns_correct_keys(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ALIGNED_CONVERGED)
        result = check_outcar(p)
        assert set(result.keys()) == {
            'outcar_path', 'potcar_aligned', 'potcar_message',
            'poscar_elements', 'potcar_elements',
            'n_steps', 'scf_converged', 'n_scf_failed', 'ionic_converged',
        }

    def test_converged_single_step(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        result = check_outcar(p)
        assert result['n_steps'] == 1
        assert result['scf_converged'] == [True]
        assert result['n_scf_failed'] == 0
        assert result['ionic_converged'] is True

    def test_failed_scf(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_SCF_FAILED)
        result = check_outcar(p)
        assert result['n_steps'] == 1
        assert result['scf_converged'] == [False]
        assert result['n_scf_failed'] == 1
        assert result['ionic_converged'] is False

    def test_multi_step_partial(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_MULTI_STEP_PARTIAL)
        result = check_outcar(p)
        assert result['n_steps'] == 3
        assert result['n_scf_failed'] == 1
        assert result['ionic_converged'] is False

    def test_accepts_directory(self, tmp_path):
        _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        result = check_outcar(str(tmp_path))
        assert result['n_steps'] == 1

    def test_accepts_gz_file(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED, "OUTCAR.gz")
        result = check_outcar(p)
        assert result['n_steps'] == 1
        assert result['ionic_converged'] is True

    def test_accepts_directory_with_gz(self, tmp_path):
        _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED, "OUTCAR.gz")
        result = check_outcar(str(tmp_path))
        assert result['n_steps'] == 1

    def test_raises_on_missing_outcar(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            check_outcar(str(tmp_path))

    def test_outcar_path_in_result(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        result = check_outcar(p)
        assert result['outcar_path'] == p

    def test_n_scf_failed_count(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_MULTI_STEP_PARTIAL)
        result = check_outcar(p)
        # Step 2 failed, steps 1 and 3 converged
        assert result['n_scf_failed'] == 1


# ---------------------------------------------------------------------------
# _parse_potcar_poscar_elements  (unit tests on raw text)
# ---------------------------------------------------------------------------

class TestParsePotcarPoscarElements:
    def test_aligned_multi_element(self):
        poscar, potcar = _parse_potcar_poscar_elements(_POTCAR_POSCAR_HEADER_ALIGNED)
        assert poscar == ['Si', 'H', 'C']
        assert potcar == ['Si', 'H', 'C']

    def test_mismatched_returns_different_lists(self):
        poscar, potcar = _parse_potcar_poscar_elements(_POTCAR_POSCAR_HEADER_MISMATCHED)
        assert poscar == ['Si', 'H', 'C']
        assert potcar == ['Si', 'C', 'H']

    def test_single_element_no_doubling(self):
        poscar, potcar = _parse_potcar_poscar_elements(_POTCAR_POSCAR_HEADER_SINGLE)
        assert poscar == ['Si']
        assert potcar == ['Si']

    def test_paw_suffix_stripped(self):
        poscar, potcar = _parse_potcar_poscar_elements(_POTCAR_POSCAR_HEADER_PAW_SUFFIX)
        assert poscar == ['K', 'O']
        assert potcar == ['K', 'O']

    def test_leading_space_vasp5_format(self):
        # Real-world VASP 5.x OUTCARs indent POTCAR/POSCAR lines with a space
        poscar, potcar = _parse_potcar_poscar_elements(_POTCAR_POSCAR_HEADER_LEADING_SPACE)
        assert poscar == ['Si', 'H', 'O', 'C']
        assert potcar == ['Si', 'H', 'O', 'C']

    def test_missing_potcar_raises(self):
        with pytest.raises(ValueError, match="POTCAR"):
            _parse_potcar_poscar_elements("POSCAR: Si H\n")

    def test_missing_poscar_raises(self):
        with pytest.raises(ValueError, match="POSCAR"):
            _parse_potcar_poscar_elements("POTCAR: PAW_PBE Si 05Jan2001\n")


# ---------------------------------------------------------------------------
# check_potcar_poscar_alignment
# ---------------------------------------------------------------------------

class TestCheckPotcarPoscarAlignment:
    def test_aligned_returns_true(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ALIGNED_CONVERGED)
        result = check_potcar_poscar_alignment(p)
        assert result['aligned'] is True
        assert result['message'] == 'OK'
        assert result['poscar_elements'] == ['Si', 'H', 'C']
        assert result['potcar_elements'] == ['Si', 'H', 'C']

    def test_mismatched_returns_false(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_MISMATCHED_CONVERGED)
        result = check_potcar_poscar_alignment(p)
        assert result['aligned'] is False
        assert 'MISMATCH' in result['message']
        assert result['poscar_elements'] == ['Si', 'H', 'C']
        assert result['potcar_elements'] == ['Si', 'C', 'H']

    def test_gz_file_transparent(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ALIGNED_CONVERGED, 'OUTCAR.gz')
        result = check_potcar_poscar_alignment(p)
        assert result['aligned'] is True

    def test_returns_required_keys(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ALIGNED_CONVERGED)
        result = check_potcar_poscar_alignment(p)
        assert set(result.keys()) == {'aligned', 'poscar_elements', 'potcar_elements', 'message'}


# ---------------------------------------------------------------------------
# check_outcar — alignment keys now included
# ---------------------------------------------------------------------------

class TestCheckOutcarAlignment:
    def test_aligned_key_present(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ALIGNED_CONVERGED)
        result = check_outcar(p)
        assert 'potcar_aligned' in result
        assert 'potcar_message' in result
        assert 'poscar_elements' in result
        assert 'potcar_elements' in result

    def test_aligned_value_true(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_ALIGNED_CONVERGED)
        result = check_outcar(p)
        assert result['potcar_aligned'] is True
        assert result['potcar_message'] == 'OK'

    def test_mismatched_value_false(self, tmp_path):
        p = _write_outcar(tmp_path, OUTCAR_MISMATCHED_CONVERGED)
        result = check_outcar(p)
        assert result['potcar_aligned'] is False
        assert 'MISMATCH' in result['potcar_message']

    def test_no_header_gives_none(self, tmp_path):
        # Header-less OUTCAR (e.g. truncated file) → potcar_aligned is None
        p = _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        result = check_outcar(p)
        assert result['potcar_aligned'] is None


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestRun:
    def test_returns_dict(self, tmp_path):
        _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        result = run(str(tmp_path))
        assert isinstance(result, dict)
        assert result['ionic_converged'] is True

    def test_all_converged_output(self, tmp_path, capsys):
        _write_outcar(tmp_path, OUTCAR_MULTI_STEP_CONVERGED)
        run(str(tmp_path))
        out = capsys.readouterr().out
        assert "all steps converged" in out
        assert "CONVERGED" in out

    def test_failed_scf_output(self, tmp_path, capsys):
        _write_outcar(tmp_path, OUTCAR_MULTI_STEP_PARTIAL)
        run(str(tmp_path))
        out = capsys.readouterr().out
        assert "did NOT converge" in out
        assert "NOT CONVERGED" in out

    def test_default_path_is_cwd(self, tmp_path, monkeypatch):
        _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        monkeypatch.chdir(tmp_path)
        result = run()
        assert result['n_steps'] == 1


# ---------------------------------------------------------------------------
# main() — CLI entry point
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_with_path_arg(self, tmp_path):
        _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        with patch("sys.argv", ["vaspcheck-outcar", str(tmp_path)]):
            from tools4vasp.outcar_convergence import main
            main()  # should not raise

    def test_main_default_arg(self, tmp_path, monkeypatch):
        _write_outcar(tmp_path, OUTCAR_ONE_STEP_CONVERGED)
        monkeypatch.chdir(tmp_path)
        with patch("sys.argv", ["vaspcheck-outcar"]):
            from tools4vasp.outcar_convergence import main
            main()  # should not raise
