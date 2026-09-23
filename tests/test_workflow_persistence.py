"""Regression tests for Telegram queue/state persistence.

The production workflow keeps main protected and stores mutable Telegram
runtime state on a dedicated telegram-state branch.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "telegram.yml"


def _git(repo, *args):
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _step_block(name):
    lines = WORKFLOW.read_text().splitlines()
    step_index = next(
        i for i, line in enumerate(lines)
        if line.strip() == f"- name: {name}"
    )
    run_index = next(
        i
        for i in range(step_index, len(lines))
        if lines[i].strip() == "run: |"
    )
    block = []
    for line in lines[run_index + 1:]:
        if line.strip() == "":
            block.append("")
            continue
        if not line.startswith(" " * 10):
            break
        block.append(line[10:])
    return "\n".join(block).rstrip() + "\n"


def _run_step(repo, name, runner_temp):
    runner_temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["RUNNER_TEMP"] = str(runner_temp)
    return subprocess.run(
        ["bash", "-e", "-c", _step_block(name)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def _write_runtime_files(work, marker="initial"):
    data = work / "data"
    data.mkdir(exist_ok=True)
    (data / "telegram_queue.json").write_text(
        '{"generated_at":"%s","count":0,"stories":[]}' % marker,
        encoding="utf-8",
    )
    (data / "telegram_state.json").write_text(
        '{"posted":[{"story_id":"%s"}],"scheduled":[],"failures":[]}' % marker,
        encoding="utf-8",
    )


@pytest.fixture
def git_sandbox(tmp_path):
    work = tmp_path / "work"
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", str(origin)],
        check=True,
    )
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(work)],
        check=True,
    )
    _git(work, "config", "user.email", "bot@test")
    _git(work, "config", "user.name", "bot")
    _git(work, "remote", "add", "origin", str(origin))
    shutil.copyfile(ROOT / ".gitignore", work / ".gitignore")
    _git(work, "add", ".gitignore")
    _git(work, "commit", "-q", "-m", "init")
    _git(work, "push", "-q", "-u", "origin", "main")
    return work


def _state_branch_head(work):
    out = _git(work, "ls-remote", "--heads", "origin", "telegram-state")
    assert out
    return out.split()[0]


def _fetch_state_branch(work):
    _git(work, "fetch", "-q", "origin", "telegram-state")
    return _git(work, "rev-parse", "FETCH_HEAD")


def test_persist_creates_state_branch_without_updating_main(
    git_sandbox, tmp_path
):
    work = git_sandbox
    main_before = _git(work, "rev-parse", "origin/main")
    _write_runtime_files(work)

    proc = _run_step(
        work,
        "Persist queue and state",
        tmp_path / "runner-1",
    )

    assert proc.returncode == 0, proc.stderr
    assert _git(work, "rev-parse", "origin/main") == main_before
    state_head = _state_branch_head(work)
    assert state_head != main_before

    _fetch_state_branch(work)
    files = _git(
        work,
        "show",
        "--pretty=",
        "--name-only",
        "FETCH_HEAD",
    ).splitlines()
    assert set(files) == {
        "data/telegram_queue.json",
        "data/telegram_state.json",
    }


def test_persist_updates_existing_state_branch(git_sandbox, tmp_path):
    work = git_sandbox
    main_before = _git(work, "rev-parse", "origin/main")

    _write_runtime_files(work, "first")
    first = _run_step(
        work,
        "Persist queue and state",
        tmp_path / "runner-1",
    )
    assert first.returncode == 0, first.stderr
    first_state_head = _state_branch_head(work)

    _write_runtime_files(work, "second")
    second = _run_step(
        work,
        "Persist queue and state",
        tmp_path / "runner-2",
    )
    assert second.returncode == 0, second.stderr
    second_state_head = _state_branch_head(work)

    assert second_state_head != first_state_head
    assert _git(work, "rev-parse", "origin/main") == main_before

    _fetch_state_branch(work)
    state = _git(
        work,
        "show",
        "FETCH_HEAD:data/telegram_state.json",
    )
    assert '"story_id":"second"' in state


def test_persist_excludes_other_runtime_files(git_sandbox, tmp_path):
    work = git_sandbox
    _write_runtime_files(work)
    (work / "data" / "news.db").write_bytes(b"db-bytes")
    (work / "data" / "source_health.json").write_text(
        "{}",
        encoding="utf-8",
    )

    proc = _run_step(
        work,
        "Persist queue and state",
        tmp_path / "runner-1",
    )
    assert proc.returncode == 0, proc.stderr

    _fetch_state_branch(work)
    files = _git(
        work,
        "show",
        "--pretty=",
        "--name-only",
        "FETCH_HEAD",
    ).splitlines()
    assert set(files) == {
        "data/telegram_queue.json",
        "data/telegram_state.json",
    }


def test_restore_loads_latest_state_branch(git_sandbox, tmp_path):
    work = git_sandbox
    _write_runtime_files(work, "saved")

    persist = _run_step(
        work,
        "Persist queue and state",
        tmp_path / "runner-1",
    )
    assert persist.returncode == 0, persist.stderr

    _write_runtime_files(work, "local")
    restore = _run_step(
        work,
        "Restore Telegram queue and state",
        tmp_path / "runner-2",
    )
    assert restore.returncode == 0, restore.stderr
    assert '"story_id":"saved"' in (
        work / "data" / "telegram_state.json"
    ).read_text(encoding="utf-8")


def test_workflow_never_pushes_runtime_state_to_main():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "HEAD:refs/heads/telegram-state" in text
    assert "git push origin main" not in text
    assert "git pull --rebase origin main" not in text


def test_restore_happens_before_collection():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert text.index("- name: Restore Telegram queue and state") < text.index(
        "- name: Collect fresh news"
    )
