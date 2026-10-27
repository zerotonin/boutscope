"""local_paths.json resolution."""

from __future__ import annotations

import json

import pytest

from boutscope import config


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for key in ("PATHS", "PROFILE", "VIDEO_ROOT", "PROJECT_ROOT"):
        monkeypatch.delenv(f"BOUTSCOPE_{key}", raising=False)
    monkeypatch.setattr(config, "_candidate_files", lambda: [tmp_path / "local_paths.json"])
    return tmp_path


def test_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("BOUTSCOPE_VIDEO_ROOT", str(tmp_path / "vids"))
    assert config.video_root() == tmp_path / "vids"


def test_profile_from_file(monkeypatch, tmp_path):
    (tmp_path / "local_paths.json").write_text(
        json.dumps(
            {
                "active_profile": "lab",
                "profiles": {
                    "lab": {"video_root": str(tmp_path / "lab")},
                    "local": {"video_root": "x"},
                },
            }
        ),
        encoding="utf-8",
    )
    assert config.video_root() == tmp_path / "lab"
    monkeypatch.setenv("BOUTSCOPE_PROFILE", "local")
    assert str(config.video_root()) == "x"


def test_missing_or_placeholder_fails_loudly(tmp_path):
    with pytest.raises(config.ConfigError, match="local_paths.template.json"):
        config.project_root()
    (tmp_path / "local_paths.json").write_text(
        json.dumps({"profiles": {"local": {"project_root": "<absolute path to your projects>"}}}),
        encoding="utf-8",
    )
    with pytest.raises(config.ConfigError, match="project_root"):
        config.project_root()


def test_unknown_key():
    with pytest.raises(KeyError):
        config.get_path("data_root")
