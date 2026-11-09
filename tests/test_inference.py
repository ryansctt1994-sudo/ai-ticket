"""Offline tests for the on_event flow (no network calls)."""
import importlib
import json

import ai_ticket.backends.pygithub as backend


def _payload(agent_name):
    return json.dumps(
        {"messages": [{"role": "system", "content": f"You are {agent_name}, an AI designed to help."}]},
        indent=2,
    )


class FakeComment:
    def __init__(self, url):
        self.url = url


class FakeIssue:
    def __init__(self, number, body):
        self.number = number
        self.body = body
        self.comments = []

    def create_comment(self, body):
        self.comments.append(body)
        return FakeComment(f"issue/{self.number}#comment-{len(self.comments)}")


class FakeRepo:
    def __init__(self, issues=()):
        self.issues = list(issues)

    def get_issues(self):
        return iter(list(self.issues))

    def create_issue(self, title, body):
        issue = FakeIssue(len(self.issues) + 1, body)
        self.issues.append(issue)
        return issue


def _load_inference(monkeypatch, repo):
    # inference calls load_env() at import time, which would contact GitHub.
    monkeypatch.setattr(backend, "load_env", lambda: None)
    monkeypatch.setattr(backend, "repo", repo)
    return importlib.import_module("ai_ticket.events.inference")


def test_event_is_commented_on_its_agents_existing_ticket(monkeypatch):
    other = FakeIssue(1, "```" + _payload("Other-GPT") + "```")
    mine = FakeIssue(2, "```" + _payload("Entrepreneur-GPT") + "```")
    repo = FakeRepo([other, mine])
    inference = _load_inference(monkeypatch, repo)

    url = inference.on_event({"content": _payload("Entrepreneur-GPT")})

    assert url == "issue/2#comment-1"
    assert len(repo.issues) == 2
    assert other.comments == []


def test_event_for_new_agent_creates_a_ticket_then_comments(monkeypatch):
    other = FakeIssue(1, "```" + _payload("Other-GPT") + "```")
    repo = FakeRepo([other])
    inference = _load_inference(monkeypatch, repo)

    url = inference.on_event({"content": _payload("Entrepreneur-GPT")})

    assert len(repo.issues) == 2
    created = repo.issues[1]
    assert backend.find_name(created.body) == "Entrepreneur-GPT"
    assert url == "issue/2#comment-1"
    assert other.comments == []
