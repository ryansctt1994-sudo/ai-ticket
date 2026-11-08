"""Offline tests for ticket matching in the PyGithub backend (no network calls)."""
import json

import ai_ticket.backends.pygithub as backend


def _payload(agent_name):
    return json.dumps(
        {"messages": [{"role": "system", "content": f"You are {agent_name}, an AI designed to help."}]},
        indent=2,
    )


class FakeIssue:
    def __init__(self, body):
        self.body = body


class FakeRepo:
    def __init__(self, issues):
        self._issues = issues

    def get_issues(self):
        return iter(self._issues)


def _with_issues(monkeypatch, issues):
    monkeypatch.setattr(backend, "repo", FakeRepo(issues))


def test_returns_issue_for_matching_agent(monkeypatch):
    other = FakeIssue("```" + _payload("Other-GPT") + "```")
    mine = FakeIssue("```" + _payload("Entrepreneur-GPT") + "```")
    _with_issues(monkeypatch, [other, mine])

    assert backend.get_existing_ticket({"content": _payload("Entrepreneur-GPT")}) is mine


def test_returns_none_when_no_issue_matches_agent(monkeypatch):
    _with_issues(monkeypatch, [FakeIssue("```" + _payload("Other-GPT") + "```"), FakeIssue(None)])

    assert backend.get_existing_ticket({"content": _payload("Entrepreneur-GPT")}) is None


def test_returns_none_when_event_has_no_agent_name(monkeypatch):
    _with_issues(monkeypatch, [FakeIssue("```" + _payload("Other-GPT") + "```")])

    assert backend.get_existing_ticket({"content": "not an agent prompt"}) is None
