import importlib.util
from pathlib import Path
import sys
import pytest

from radar_platform.navigation.io import ROOT,digest,write_json


@pytest.fixture
def tmp_path(project_tmp_path):
    return project_tmp_path

sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('navigation_cache_priority',ROOT/'scripts/prioritize_navigation_terrain_turns.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_only_completed_identical_case_is_published(tmp_path,monkeypatch):
    base={'scene':'unchanged'};record={'id':'turn_00000'};extra=tmp_path/'extra';original=tmp_path/'original'
    def generate(job):
        extra.mkdir();value={'configuration_sha256':digest(dict(base=base,record=record))}
        write_json(extra/'complete.json',value);return value
    monkeypatch.setattr(module,'generate_case',generate)
    result=module.one((base,record,str(extra),str(original)))
    assert result['result']=='published_complete_cache'
    assert original.is_symlink() and original.resolve()==extra.resolve()
    assert (original/'complete.json').exists()


def test_active_original_case_is_never_overwritten(tmp_path,monkeypatch):
    original=tmp_path/'original';original.mkdir();(original/'request_0.json').write_text('owned')
    def forbidden(_):raise AssertionError('must not duplicate active original case')
    monkeypatch.setattr(module,'generate_case',forbidden)
    result=module.one(({}, {'id':'turn_00000'},str(tmp_path/'extra'),str(original)))
    assert result['result']=='owned_by_original_producer'
    assert (original/'request_0.json').read_text()=='owned' and not original.is_symlink()


def test_original_winning_publication_race_is_preserved(tmp_path,monkeypatch):
    base={};record={'id':'turn_00000'};extra=tmp_path/'extra';original=tmp_path/'original'
    def generate(job):
        extra.mkdir();original.mkdir();(original/'request_0.json').write_text('original')
        value={'configuration_sha256':digest(dict(base=base,record=record))};write_json(extra/'complete.json',value);return value
    monkeypatch.setattr(module,'generate_case',generate)
    result=module.one((base,record,str(extra),str(original)))
    assert result['result']=='original_producer_started_first_original_preserved'
    assert (original/'request_0.json').read_text()=='original' and not original.is_symlink()


def test_configuration_mismatch_cannot_enter_original_cache(tmp_path,monkeypatch):
    extra=tmp_path/'extra';original=tmp_path/'original'
    def generate(job):
        extra.mkdir();write_json(extra/'complete.json',{'configuration_sha256':'wrong'})
        return {'configuration_sha256':'wrong'}
    monkeypatch.setattr(module,'generate_case',generate)
    with pytest.raises(ValueError,match='configuration'):module.one(({}, {'id':'turn_00000'},str(extra),str(original)))
    assert not original.exists() and not original.is_symlink()
