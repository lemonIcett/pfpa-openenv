"""
PFPA OpenEnv — Task Scenarios
Pre-defined signal scenarios for each difficulty level.
Each scenario defines the initial state and the ground-truth optimal actions.

Randomness: signals and predictions are shuffled on every reset() call using
a random seed, so the agent cannot memorise a fixed action sequence.
"""
from __future__ import annotations
import random
from typing import Dict, List
from environment.models import (
    ContextSignal, Prediction, CalendarEvent,
    EnvironmentState, SignalType, SignalUrgency, PredictionStatus,
    ConfidenceLevel, ActionType, UserContext, TaskInfo, TaskDifficulty
)
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")

def _now(offset_hours=0) -> str:
    t = datetime.now(IST) + timedelta(hours=offset_hours)
    return t.isoformat()

def _shuffle(items: list, rng: random.Random) -> list:
    """Return a shuffled copy without mutating the original."""
    copy = list(items)
    rng.shuffle(copy)
    return copy


# ── TASK METADATA ─────────────────────────────────────────────────────────────

TASK_CATALOG: Dict[str, TaskInfo] = {
    "signal_triage_easy": TaskInfo(
        id="signal_triage_easy",
        name="Signal Triage — Easy",
        difficulty=TaskDifficulty.easy,
        description=(
            "Given 5 incoming context signals (calendar, email, Slack), "
            "execute exactly the 2 high-confidence (green) predictions and "
            "dismiss all low-confidence ones. Avoid false positives."
        ),
        max_steps=10,
        reward_range=[0.0, 1.0],
    ),
    "workflow_optimization_medium": TaskInfo(
        id="workflow_optimization_medium",
        name="Workflow Optimization — Medium",
        difficulty=TaskDifficulty.medium,
        description=(
            "Manage a busy workday with 12 signals. You must create at least one "
            "workflow rule, resolve a calendar conflict, and maintain prediction "
            "accuracy above 70% across all decisions."
        ),
        max_steps=20,
        reward_range=[0.0, 1.0],
    ),
    "crisis_management_hard": TaskInfo(
        id="crisis_management_hard",
        name="Crisis Management — Hard",
        difficulty=TaskDifficulty.hard,
        description=(
            "A signal storm with 25+ events. Overlapping meetings, urgent emails, "
            "competing Slack threads. Correctly triage urgency, avoid automation of "
            "sensitive actions, and set resilient workflow rules for the future."
        ),
        max_steps=40,
        reward_range=[0.0, 1.0],
    ),
}


# ── EASY SCENARIO ─────────────────────────────────────────────────────────────

def build_easy_scenario(seed: int | None = None) -> EnvironmentState:
    rng = random.Random(seed)

    signals = [
        ContextSignal(
            id="sig-001",
            signal_type=SignalType.calendar_event,
            urgency=SignalUrgency.high,
            title="Team standup in 15 minutes",
            description="Daily standup with engineering team. No prep notes added yet.",
            source="Google Calendar",
            timestamp=_now(0),
        ),
        ContextSignal(
            id="sig-002",
            signal_type=SignalType.email_received,
            urgency=SignalUrgency.high,
            title="Client deadline moved to tomorrow",
            description="Riya from Acme Corp emailed: project deadline is now April 6th, not April 10th.",
            source="Gmail",
            timestamp=_now(-0.2),
        ),
        ContextSignal(
            id="sig-003",
            signal_type=SignalType.slack_message,
            urgency=SignalUrgency.low,
            title="Random meme shared in #general",
            description="Colleague shared a meme in the general channel.",
            source="Slack",
            timestamp=_now(-0.5),
        ),
        ContextSignal(
            id="sig-004",
            signal_type=SignalType.file_activity,
            urgency=SignalUrgency.low,
            title="Design assets folder updated",
            description="7 new PNG files added to /projects/acme/design/",
            source="File System",
            timestamp=_now(-1),
        ),
        ContextSignal(
            id="sig-005",
            signal_type=SignalType.email_received,
            urgency=SignalUrgency.medium,
            title="Quarterly review scheduled",
            description="HR sent calendar invite for Q2 performance review on April 15.",
            source="Gmail",
            timestamp=_now(-2),
        ),
    ]

    predictions = [
        Prediction(
            id="pred-001",
            description="Add standup prep notes to calendar event",
            action_type=ActionType.create_calendar_event,
            confidence=88.0,
            confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-001",
            trigger_context="Standup in 15 min with no prep notes",
            reward_if_correct=0.4,
            penalty_if_wrong=0.1,
        ),
        Prediction(
            id="pred-002",
            description="Draft urgent reply to Riya about revised deadline",
            action_type=ActionType.draft_email,
            confidence=92.0,
            confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-002",
            trigger_context="Client deadline moved up 4 days",
            reward_if_correct=0.5,
            penalty_if_wrong=0.15,
        ),
        Prediction(
            id="pred-003",
            description="React with emoji to meme in #general",
            action_type=ActionType.send_slack_message,
            confidence=22.0,
            confidence_level=ConfidenceLevel.red,
            trigger_signal_id="sig-003",
            trigger_context="Meme in general channel",
            reward_if_correct=0.05,
            penalty_if_wrong=0.2,
        ),
    ]

    calendar_events = [
        CalendarEvent(
            id="cal-001",
            title="Daily Standup",
            start_time=_now(0.25),
            end_time=_now(0.5),
            attendees=["team@company.com"],
        ),
        CalendarEvent(
            id="cal-002",
            title="Lunch",
            start_time=_now(3),
            end_time=_now(4),
        ),
    ]

    return EnvironmentState(
        task_id="signal_triage_easy",
        max_steps=10,
        pending_signals=_shuffle(signals, rng),
        active_predictions=_shuffle(predictions, rng),
        calendar_events=calendar_events,
        current_time=_now(0),
        user_context=UserContext(),
        info={
            "optimal_actions": ["execute pred-001", "execute pred-002", "dismiss pred-003"],
            "hint": "Focus on green-confidence predictions tied to urgent signals.",
        },
    )


# ── MEDIUM SCENARIO ───────────────────────────────────────────────────────────

def build_medium_scenario(seed: int | None = None) -> EnvironmentState:
    rng = random.Random(seed)

    signals = [
        ContextSignal(id="sig-m01", signal_type=SignalType.email_received, urgency=SignalUrgency.high,
            title="Board deck review requested by EOD", description="CFO asked for Q1 board deck review by 6 PM today.",
            source="Gmail", timestamp=_now(0)),
        ContextSignal(id="sig-m02", signal_type=SignalType.calendar_event, urgency=SignalUrgency.high,
            title="Two meetings overlap at 2 PM", description="Product sync and investor call both scheduled 2–3 PM.",
            source="Google Calendar", timestamp=_now(-0.1), metadata={"conflict": True}),
        ContextSignal(id="sig-m03", signal_type=SignalType.slack_message, urgency=SignalUrgency.medium,
            title="Dev team blocked on API keys", description="#backend: team is waiting for prod API keys to continue.",
            source="Slack", timestamp=_now(-0.5)),
        ContextSignal(id="sig-m04", signal_type=SignalType.email_received, urgency=SignalUrgency.medium,
            title="Vendor invoice due in 3 days", description="Invoice #4421 from AWS for $2,340 due April 8.",
            source="Gmail", timestamp=_now(-1)),
        ContextSignal(id="sig-m05", signal_type=SignalType.file_activity, urgency=SignalUrgency.low,
            title="Old Q4 reports still in Desktop", description="27 files from last quarter sitting in Desktop folder.",
            source="File System", timestamp=_now(-2)),
        ContextSignal(id="sig-m06", signal_type=SignalType.slack_message, urgency=SignalUrgency.low,
            title="Lunch poll in #random", description="Team is voting on lunch spot.",
            source="Slack", timestamp=_now(-3)),
        ContextSignal(id="sig-m07", signal_type=SignalType.email_received, urgency=SignalUrgency.medium,
            title="Candidate interview confirmation needed", description="HR needs confirm/deny for interview slot tomorrow 11 AM.",
            source="Gmail", timestamp=_now(-1.5)),
        ContextSignal(id="sig-m08", signal_type=SignalType.calendar_event, urgency=SignalUrgency.medium,
            title="1:1 with manager tomorrow — no agenda", description="Weekly 1:1 tomorrow 10 AM, agenda not filled.",
            source="Google Calendar", timestamp=_now(-4)),
        ContextSignal(id="sig-m09", signal_type=SignalType.browser_tab, urgency=SignalUrgency.low,
            title="31 browser tabs open", description="Context monitor detected 31 open tabs slowing productivity.",
            source="Chrome", timestamp=_now(-0.2)),
        ContextSignal(id="sig-m10", signal_type=SignalType.email_received, urgency=SignalUrgency.high,
            title="Production issue reported by customer", description="Support ticket: login flow broken for 3 enterprise users.",
            source="Gmail", timestamp=_now(-0.05)),
        ContextSignal(id="sig-m11", signal_type=SignalType.slack_message, urgency=SignalUrgency.high,
            title="Urgent: server latency spike", description="#alerts: p99 latency jumped to 4.2s in prod.",
            source="Slack", timestamp=_now(-0.03)),
        ContextSignal(id="sig-m12", signal_type=SignalType.file_activity, urgency=SignalUrgency.low,
            title="Duplicate files detected", description="6 duplicate files found in /projects directory.",
            source="File System", timestamp=_now(-5)),
    ]

    predictions = [
        Prediction(id="pred-m01", description="Block 1 hour for board deck review before 5 PM",
            action_type=ActionType.create_calendar_event, confidence=91.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-m01", trigger_context="Board deck due EOD", reward_if_correct=0.3, penalty_if_wrong=0.1),
        Prediction(id="pred-m02", description="Reschedule investor call to 3 PM to fix conflict",
            action_type=ActionType.create_calendar_event, confidence=85.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-m02", trigger_context="Calendar conflict at 2 PM", reward_if_correct=0.25, penalty_if_wrong=0.1),
        Prediction(id="pred-m03", description="Send API keys to backend team via Slack",
            action_type=ActionType.send_slack_message, confidence=78.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-m03", trigger_context="Dev team blocked", reward_if_correct=0.2, penalty_if_wrong=0.15),
        Prediction(id="pred-m04", description="Escalate production login issue to on-call engineer",
            action_type=ActionType.send_slack_message, confidence=96.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-m10", trigger_context="Customer-reported prod issue", reward_if_correct=0.35, penalty_if_wrong=0.2),
        Prediction(id="pred-m05", description="Create workflow: auto-escalate #alerts messages to on-call",
            action_type=ActionType.none, confidence=82.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-m11", trigger_context="Latency spike in #alerts", reward_if_correct=0.3, penalty_if_wrong=0.05),
        Prediction(id="pred-m06", description="Vote in lunch poll",
            action_type=ActionType.send_slack_message, confidence=18.0, confidence_level=ConfidenceLevel.red,
            trigger_signal_id="sig-m06", trigger_context="Lunch poll in #random", reward_if_correct=0.02, penalty_if_wrong=0.25),
    ]

    calendar_events = [
        CalendarEvent(id="cal-m01", title="Product Sync", start_time=_now(4), end_time=_now(5),
            attendees=["product@co.com"], is_conflicting=True),
        CalendarEvent(id="cal-m02", title="Investor Call", start_time=_now(4), end_time=_now(5),
            attendees=["investors@vc.com"], is_conflicting=True),
        CalendarEvent(id="cal-m03", title="1:1 with Manager", start_time=_now(18), end_time=_now(19)),
    ]

    return EnvironmentState(
        task_id="workflow_optimization_medium",
        max_steps=20,
        pending_signals=_shuffle(signals, rng),
        active_predictions=_shuffle(predictions, rng),
        calendar_events=calendar_events,
        current_time=_now(0),
        user_context=UserContext(),
        info={
            "optimal_actions": [
                "execute pred-m04 (prod issue — highest priority)",
                "execute pred-m01 (board deck)",
                "execute pred-m02 (resolve conflict)",
                "execute pred-m03 (unblock team)",
                "set_workflow_rule for #alerts",
                "dismiss pred-m06 (low-value)",
            ],
            "hint": "Resolve the production issue first. Then handle the calendar conflict. Create a workflow rule for recurring alert patterns.",
        },
    )


# ── HARD SCENARIO ─────────────────────────────────────────────────────────────

def build_hard_scenario(seed: int | None = None) -> EnvironmentState:
    """Full crisis day — 25 signals, competing priorities, trap predictions."""
    rng = random.Random(seed)

    base_signals = []

    # High urgency cluster
    for i, (stype, title, desc, src) in enumerate([
        (SignalType.email_received, "CEO email: All-hands in 1 hour", "CEO called emergency all-hands at 11 AM.", "Gmail"),
        (SignalType.slack_message, "CRITICAL: Database replication lag", "#db-alerts: replication lag at 47 seconds.", "Slack"),
        (SignalType.email_received, "Legal: NDA to sign before 5 PM today", "Vendor NDA must be returned by COB.", "Gmail"),
        (SignalType.calendar_event, "3 meetings scheduled same slot (3–4 PM)", "Sales call, design review, and sprint retro overlap.", "Google Calendar"),
        (SignalType.email_received, "Customer churn threat — urgent account review", "Enterprise customer ($240k ARR) threatening to cancel.", "Gmail"),
    ]):
        base_signals.append(ContextSignal(
            id=f"sig-h{i+1:02d}", signal_type=stype, urgency=SignalUrgency.high,
            title=title, description=desc, source=src, timestamp=_now(-i*0.1)))

    # Medium urgency cluster
    for i, (stype, title, desc, src) in enumerate([
        (SignalType.email_received, "7 unanswered partnership emails", "7 partner emails older than 48h unanswered.", "Gmail"),
        (SignalType.slack_message, "Design team needs feedback by 3 PM", "Figma designs for v3 landing page awaiting review.", "Slack"),
        (SignalType.file_activity, "Sprint report due tomorrow, folder empty", "/reports/sprint-47/ has no content.", "File System"),
        (SignalType.calendar_event, "Quarterly review prep not started", "Q2 review is in 2 days — no prep doc exists.", "Google Calendar"),
        (SignalType.email_received, "Recruiter: 3 offers expiring this week", "Candidates waiting on offer letters.", "Gmail"),
        (SignalType.browser_tab, "48 open tabs", "Context monitor: 48 open browser tabs.", "Chrome"),
        (SignalType.slack_message, "Marketing wants copy review", "Blog post draft in Google Docs, requested by tomorrow.", "Slack"),
        (SignalType.email_received, "Subscription renewal ($800/yr)", "Zapier annual plan renews tomorrow.", "Gmail"),
    ]):
        base_signals.append(ContextSignal(
            id=f"sig-h{i+6:02d}", signal_type=stype, urgency=SignalUrgency.medium,
            title=title, description=desc, source=src, timestamp=_now(-(i+5)*0.3)))

    # Low urgency / noise signals
    for i, (stype, title, desc, src) in enumerate([
        (SignalType.slack_message, "Someone reacted to your message from yesterday", "👍 on #general post.", "Slack"),
        (SignalType.email_received, "LinkedIn digest email", "Weekly LinkedIn activity summary.", "Gmail"),
        (SignalType.file_activity, "Desktop screenshots folder", "Screenshots folder has 93 files.", "File System"),
        (SignalType.email_received, "Newsletter: Top 10 productivity apps", "Marketing newsletter.", "Gmail"),
        (SignalType.browser_tab, "Wikipedia tab: History of email", "Left open from research 3 days ago.", "Chrome"),
        (SignalType.slack_message, "Office supply order poll", "Admin is ordering office supplies.", "Slack"),
        (SignalType.email_received, "Meetup invite: Local JS user group", "JavaScript meetup next Thursday.", "Gmail"),
        (SignalType.file_activity, "Cache files over 1GB detected", "Browser cache is 1.2 GB.", "File System"),
        (SignalType.slack_message, "Good morning GIF in #random", "GIF posted in #random.", "Slack"),
        (SignalType.email_received, "Loyalty points expiring — coffee app", "Coffee loyalty points expire in 30 days.", "Gmail"),
        (SignalType.slack_message, "Trivia question in #fun channel", "Weekly trivia in #fun.", "Slack"),
        (SignalType.email_received, "App update available: Notion", "Notion desktop app update.", "Gmail"),
    ]):
        base_signals.append(ContextSignal(
            id=f"sig-h{i+14:02d}", signal_type=stype, urgency=SignalUrgency.low,
            title=title, description=desc, source=src, timestamp=_now(-(i+10)*0.5)))

    predictions = [
        Prediction(id="pred-h01", description="Add all-hands to calendar, block prep time",
            action_type=ActionType.create_calendar_event, confidence=94.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-h01", trigger_context="CEO all-hands in 1 hour", reward_if_correct=0.2, penalty_if_wrong=0.05),
        Prediction(id="pred-h02", description="Alert on-call DBA via Slack for replication lag",
            action_type=ActionType.send_slack_message, confidence=97.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-h02", trigger_context="DB replication lag critical", reward_if_correct=0.25, penalty_if_wrong=0.1),
        Prediction(id="pred-h03", description="Draft NDA reminder and calendar block to sign",
            action_type=ActionType.create_calendar_event, confidence=88.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-h03", trigger_context="NDA due COB today", reward_if_correct=0.2, penalty_if_wrong=0.1),
        Prediction(id="pred-h04", description="Reschedule design review to 4–5 PM",
            action_type=ActionType.create_calendar_event, confidence=80.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-h04", trigger_context="3-way calendar conflict at 3 PM", reward_if_correct=0.15, penalty_if_wrong=0.1),
        Prediction(id="pred-h05", description="Draft urgent account review email for at-risk customer",
            action_type=ActionType.draft_email, confidence=93.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-h05", trigger_context="Enterprise customer churn risk", reward_if_correct=0.25, penalty_if_wrong=0.15),
        Prediction(id="pred-h06", description="Auto-respond to partner emails with 'reviewing by EOW'",
            action_type=ActionType.draft_email, confidence=72.0, confidence_level=ConfidenceLevel.yellow,
            trigger_signal_id="sig-h06", trigger_context="7 unanswered partner emails", reward_if_correct=0.1, penalty_if_wrong=0.1),
        Prediction(id="pred-h07", description="Set workflow: auto-escalate #db-alerts to DBA on-call",
            action_type=ActionType.none, confidence=89.0, confidence_level=ConfidenceLevel.green,
            trigger_signal_id="sig-h02", trigger_context="DB alert pattern", reward_if_correct=0.2, penalty_if_wrong=0.05),
        # Trap predictions — should be dismissed
        Prediction(id="pred-h08", description="Reply to LinkedIn digest with engagement",
            action_type=ActionType.send_slack_message, confidence=15.0, confidence_level=ConfidenceLevel.red,
            trigger_signal_id="sig-h16", trigger_context="LinkedIn digest", reward_if_correct=0.01, penalty_if_wrong=0.3),
        Prediction(id="pred-h09", description="Delete all screenshots",
            action_type=ActionType.organize_files, confidence=35.0, confidence_level=ConfidenceLevel.red,
            trigger_signal_id="sig-h17", trigger_context="Screenshots folder large", reward_if_correct=0.02, penalty_if_wrong=0.25),
        Prediction(id="pred-h10", description="Sign up for JS meetup",
            action_type=ActionType.create_calendar_event, confidence=28.0, confidence_level=ConfidenceLevel.red,
            trigger_signal_id="sig-h21", trigger_context="Meetup invite", reward_if_correct=0.01, penalty_if_wrong=0.2),
    ]

    calendar_events = [
        CalendarEvent(id="cal-h01", title="All-Hands (Emergency)", start_time=_now(1), end_time=_now(2), is_conflicting=False),
        CalendarEvent(id="cal-h02", title="Sales Call", start_time=_now(5), end_time=_now(6), is_conflicting=True),
        CalendarEvent(id="cal-h03", title="Design Review", start_time=_now(5), end_time=_now(6), is_conflicting=True),
        CalendarEvent(id="cal-h04", title="Sprint Retro", start_time=_now(5), end_time=_now(6), is_conflicting=True),
        CalendarEvent(id="cal-h05", title="Q2 Quarterly Review", start_time=_now(48), end_time=_now(50)),
    ]

    return EnvironmentState(
        task_id="crisis_management_hard",
        max_steps=40,
        pending_signals=_shuffle(base_signals, rng),
        active_predictions=_shuffle(predictions, rng),
        calendar_events=calendar_events,
        current_time=_now(0),
        user_context=UserContext(),
        info={
            "optimal_actions": [
                "execute pred-h02 (DB critical — top priority)",
                "execute pred-h05 (churn risk)",
                "execute pred-h01 (all-hands prep)",
                "execute pred-h03 (NDA deadline)",
                "execute pred-h04 (resolve conflict)",
                "set workflow pred-h07 (#db-alerts automation)",
                "dismiss pred-h08, pred-h09, pred-h10 (traps)",
            ],
            "hint": "Prioritize system health > revenue risk > legal > scheduling. Dismiss all red-confidence predictions. Create at least one workflow rule.",
        },
    )


# ── Factory ───────────────────────────────────────────────────────────────────

SCENARIO_BUILDERS = {
    "signal_triage_easy":           build_easy_scenario,
    "workflow_optimization_medium": build_medium_scenario,
    "crisis_management_hard":       build_hard_scenario,
}

def build_scenario(task_id: str, seed: int | None = None) -> EnvironmentState:
    builder = SCENARIO_BUILDERS.get(task_id)
    if not builder:
        raise ValueError(f"Unknown task_id: {task_id}. Valid: {list(SCENARIO_BUILDERS.keys())}")
    return builder(seed=seed)
