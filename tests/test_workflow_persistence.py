"""Regression tests for Telegram queue/state git persistence.

The production ``telegram.yml`` workflow must persist
``data/telegram_queue.json`` and ``data/telegram_state.json``
across GitHub Actions runs.  These files are git-ignored with
explicit un-ignore exceptions, which means they are UNTRACKED
until their first commit; a plain ``git diff --quiet`` can never
see them.  The tests below execute the real ``Commit queue and
state`` run block from the workflow inside throwaway git repos and
assert that:

- missing queue/state files are detected as "no state changes"
- newly created (untracked) queue/state files are staged,
  committed and pushed
- modified tracked queue/state files are staged again
- ONLY the two intended files ever land in a commit
- every other data/ file (news.db, source_health.json, ...)
  stays excluded
"""
import subprocess
import shutil
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


def _commit_step_block():
    """Extract the literal ``run`` content of the workflow's
    ``Commit queue and state`` step.
    """
    lines = WORKFLOW.read_text().splitlines()
    step_index = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == "- name: Commit queue and state"
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


def _run_commit_step(repo):
    block = _commit_step_block()
    proc = subprocess.run(
        ["bash", "-e", "-c", block],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return proc


def _last_commit_files(repo, ref="origin/main"):
    out = _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        ref,
    )
    return out.splitlines()


@pytest.fixture
def git_sandbox(tmp_path):
    work = tmp_path / "work"
    origin = tmp_path / "origin.git"
    origin.mkdir()
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


def _write_runtime_files(work):
    data = work / "data"
    data.mkdir()
    (data / "telegram_queue.json").write_text(
        '{"generated_at": "2026-08-13T00:00:00Z", "count": 0, '
        '"stories": []}',
        encoding="utf-8",
    )
    (data / "telegram_state.json").write_text(
        '{"posted": [], "scheduled": [], "failures": [], '
        '"last_posted_at": null}',
        encoding="utf-8",
    )


def test_detects_missing_files_as_no_changes(git_sandbox):
    work = git_sandbox
    proc = _run_commit_step(work)
    assert proc.returncode == 0
    assert "no state changes" in proc.stdout
    assert "state changes detected" not in proc.stdout
    assert len(_last_commit_files(work)) == 0


def test_newly_created_queue_state_staged_and_committed(git_sandbox):
    work = git_sandbox
    _write_runtime_files(work)
    (work / "data" / "news.db").write_bytes(b"db-bytes")
    (work / "data" / "source_health.json").write_text(
        "{}", encoding="utf-8"
    )
    (work / "data" / "junk.bin").write_bytes(b"junk")

    proc = _run_commit_step(work)
    assert proc.returncode == 0, proc.stderr
    assert "state changes detected" in proc.stdout
    assert "no state changes" not in proc.stdout

    files = _last_commit_files(work)
    assert set(files) == {
        "data/telegram_queue.json",
        "data/telegram_state.json",
    }
    # pushed origin/main equals local head after the bot commit
    assert _git(work, "rev-parse", "origin/main") == _git(
        work, "rev-parse", "HEAD"
    )


def test_modified_tracked_state_detected_again(git_sandbox):
    work = git_sandbox
    _write_runtime_files(work)
    assert _run_commit_step(work).returncode == 0

    state_file = work / "data" / "telegram_state.json"
    state_file.write_text(
        '{"posted": [{"story_id": "s1"}], "scheduled": [], '
        '"failures": [], "last_posted_at": null}',
        encoding="utf-8",
    )

    proc = _run_commit_step(work)
    assert proc.returncode == 0, proc.stderr
    # the second commit must contain ONLY the modified state file
    files = _last_commit_files(work)
    assert files == ["data/telegram_state.json"]


def test_only_two_intended_files_can_be_staged(git_sandbox):
    work = git_sandbox
    _write_runtime_files(work)
    (work / "data" / "news.db").write_bytes(b"db-bytes")
    (work / "data" / "source_health.json").write_text(
        "{}", encoding="utf-8"
    )
    (work / "data" / "junk.bin").write_bytes(b"junk")

    # stage exactly as the workflow's git add line does
    _git(
        work,
        "add",
        "--",
        "data/telegram_queue.json",
        "data/telegram_state.json",
    )
    staged = _git(work, "diff", "--cached", "--name-only")
    assert staged.splitlines() == [
        "data/telegram_queue.json",
        "data/telegram_state.json",
    ]


def test_other_data_files_remain_excluded_across_commits(
    git_sandbox,
):
    work = git_sandbox
    _write_runtime_files(work)
    (work / "data" / "news.db").write_bytes(b"db-bytes")
    (work / "data" / "source_health.json").write_text(
        "{}", encoding="utf-8"
    )

    for turn in range(2):
        state_file = work / "data" / "telegram_state.json"
        state_file.write_text(
            '{"posted": [{"story_id": "s%d"}], "scheduled": [], '
            '"failures": [], "last_posted_at": null}' % turn,
            encoding="utf-8",
        )
        proc = _run_commit_step(work)
        assert proc.returncode == 0, proc.stderr

    commit_files = set()
    for ref in _git(
        work, "rev-list", "--first-parent", "origin/main"
    ).splitlines():
        out = _git(
            work,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            ref,
        )
        for path in out.splitlines():
            commit_files.add(path)

    assert commit_files == {
        "data/telegram_queue.json",
        "data/telegram_state.json",
    }
    for forbidden in (
        "data/news.db",
        "data/source_health.json",
        "data/.gitkeep",
    ):
        assert forbidden not in commit_files
    # runtime noise stays ignored and un-tracked
    status = _git(
        work,
        "status",
        "--porcelain",
        "--ignored",
        "data/",
    )
    assert "!! data/news.db" in status
    assert "!! data/source_health.json" in status


def test_workflow_uses_untracked_aware_detection():
    block = _commit_step_block()
    assert "git status --porcelain" in block
    assert "--untracked-files=all" in block
    assert (
        "-- data/telegram_queue.json data/telegram_state.json"
        in block
    )
    # the bug it guards against: plain git diff cannot see
    # freshly-created (untracked) queue/state files
    assert "git diff --quiet" not in block
    assert "git add data/telegram_queue.json data/telegram_state.json" in block
    assert "git pull --rebase origin main" in block
    assert "git push origin main" in block