import os

from ordivyr_lite.context import build_system_prompt, project_guidance


def test_guidance_available_before_first_edit_and_custom_system_is_preserved(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Keep identifiers snake_case. Run the focused tests.")
    assert "Keep identifiers snake_case" in build_system_prompt(None, tmp_path, "workspace", ["edit_file"])
    assert "Keep identifiers" not in build_system_prompt("operator override", tmp_path, "workspace", ["edit_file"])
    assert build_system_prompt("", tmp_path, "workspace", ["edit_file"]) == ""
    assert "Keep identifiers" not in build_system_prompt(None, tmp_path, "workspace", [])


def test_guidance_never_reads_outside_workspace_or_special_files(tmp_path):
    child = tmp_path / "project"
    child.mkdir()
    outside = tmp_path / "AGENTS.md"
    outside.write_text("Outside instructions")
    assert project_guidance(child) == ""
    (child / "AGENTS.md").symlink_to(outside)
    assert project_guidance(child) == ""
    (child / "AGENTS.md").unlink()
    os.mkfifo(child / "AGENTS.md")
    assert project_guidance(child) == ""


def test_guidance_is_bounded_with_visible_truncation(tmp_path):
    (tmp_path / "AGENTS.md").write_text("a" * 17000)
    guidance = project_guidance(tmp_path)
    assert "truncated at 16 KiB" in guidance and len(guidance) < 16800
