"""DETERMINISTIC EVAL — the browser gateway's shared agent survives a swap.

The dashboard is the one gateway that is both multi-threaded and long-lived, so
it keeps a single Waku behind `waku.ops.browser_agent`. Two callers mutate it:
dashboard.py builds it on the first chat, settings_api rebuilds it when you
change provider or model. This pins the part that is easy to get wrong.

The bug this was written for: changing ANY setting rebuilt the agent, and a
fresh Waku starts on the eternal 'default' session. So one visit to the Settings
tab silently moved you into a different conversation — your chat was still in
the database, just no longer the thread the dock was showing. It looked like
data loss and was reported as one.
"""

from __future__ import annotations

import pytest

from waku.ops import browser_agent


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A throwaway home + a clean module global, so these tests never see (or
    leave behind) the real dashboard's agent."""
    monkeypatch.setenv("WAKU_HOME", str(tmp_path))
    monkeypatch.setenv("WAKU_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    monkeypatch.setattr(browser_agent, "_agent", None)
    monkeypatch.setattr(browser_agent, "_dashboard_session", None)
    return tmp_path


def test_rebuild_keeps_the_conversation(isolated):
    """A settings change swaps the BRAIN, not the thread."""
    first = browser_agent.get_agent()
    session = first.session.session_id
    assert session != "default", "a dashboard run must never chat into 'default'"

    assert browser_agent.rebuild() is None          # None == success
    after = browser_agent.current()

    assert after is not first, "rebuild must actually build a new agent"
    assert after.session.session_id == session, (
        "rebuild dropped the chat thread — the dock would show an empty "
        "conversation and the user's messages would look lost"
    )


def test_rebuild_without_a_prior_agent_still_gets_a_dated_session(isolated):
    """Hitting Settings before ever chatting is a legitimate first action. It
    must not leave the agent parked on 'default' either."""
    assert browser_agent.current() is None
    assert browser_agent.rebuild() is None
    assert browser_agent.current().session.session_id.startswith("dashboard-")


def test_a_failed_rebuild_keeps_the_working_agent(isolated, monkeypatch):
    """A typo'd key should cost you the switch, not the agent you already had.
    Losing both is the failure mode that makes a local tool feel broken."""
    first = browser_agent.get_agent()

    def explode(*a, **k):
        raise SystemExit("no API key found")     # what get_client actually raises

    monkeypatch.setattr("waku.app.Waku", explode)

    error = browser_agent.rebuild()
    assert error and "no API key" in error
    assert browser_agent.current() is first, "a failed swap must not orphan the user"
    assert first.session.session_id, "and the surviving agent keeps its thread"


def test_rebuild_restores_the_conversation_history(isolated):
    """Issue #247: rebuild swapped the BRAIN but not the MEMORY. session_id was
    carried over, yet Session.history stayed empty, so after a Settings save
    the model started with an empty context while the dock still showed the
    conversation ("I need the event details first" about an event it had just
    created). A settings change must reload the thread's history, exactly like
    switching back to a past conversation does."""
    first = browser_agent.get_agent()
    sid = first.session.session_id
    assert sid != "default"

    first.session.add_exchange(
        "add a food festival on Oct 1 from 2 to 3 PM",
        "Saved to .waku/calendar.ics (not synced to Apple Calendar)",
        source="dashboard",
    )
    assert len(first.session.history) == 2

    assert browser_agent.rebuild() is None          # None == success
    after = browser_agent.current()

    assert after is not first
    assert after.session.session_id == sid, "rebuild dropped the chat thread"
    assert [m["content"] for m in after.session.history] == [
        "add a food festival on Oct 1 from 2 to 3 PM",
        "Saved to .waku/calendar.ics (not synced to Apple Calendar)",
    ], (
        "rebuild left Session.history empty — the model would answer with no "
        "memory of the conversation even though the dock still shows it (#247)"
    )


def test_get_agent_restores_history_on_restart(isolated):
    """Issue #247 (restart leg): a fresh process resumes the same dashboard
    thread, so Session.history must be reloaded from chat_log — not left empty
    on the brand-new Session."""
    first = browser_agent.get_agent()
    sid = first.session.session_id
    first.session.add_exchange("remember my wifi password is velociraptor",
                               "Got it — saved a note.", source="dashboard")

    # simulate a server restart: brand-new process globals
    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(browser_agent, "_agent", None)
    monkeypatch.setattr(browser_agent, "_dashboard_session", None)

    restarted = browser_agent.get_agent()
    assert restarted.session.session_id == sid
    assert [m["content"] for m in restarted.session.history] == [
        "remember my wifi password is velociraptor",
        "Got it — saved a note.",
    ]
