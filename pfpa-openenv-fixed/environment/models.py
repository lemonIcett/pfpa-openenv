"""
PFPA OpenEnv — Typed Models
All data models for the productivity AI environment.
"""
from __future__ import annotations
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


# ── Enums ────────────────────────────────────────────────────────────────────

class SignalType(str, Enum):
    calendar_event = "calendar_event"
    email_received = "email_received"
    slack_message  = "slack_message"
    file_activity  = "file_activity"
    browser_tab    = "browser_tab"

class SignalUrgency(str, Enum):
    high   = "high"
    medium = "medium"
    low    = "low"

class PredictionStatus(str, Enum):
    pending       = "pending"
    executed      = "executed"
    auto_executed = "auto_executed"
    dismissed     = "dismissed"
    failed        = "failed"

class ConfidenceLevel(str, Enum):
    green  = "green"   # ≥ 75%
    yellow = "yellow"  # 45–74%
    red    = "red"     # < 45%

class ActionType(str, Enum):
    create_calendar_event = "create_calendar_event"
    draft_email           = "draft_email"
    send_slack_message    = "send_slack_message"
    organize_files        = "organize_files"
    reminder              = "reminder"
    none                  = "none"

class TaskDifficulty(str, Enum):
    easy   = "easy"
    medium = "medium"
    hard   = "hard"


# ── Core Domain Models ────────────────────────────────────────────────────────

class ContextSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    signal_type: SignalType
    urgency: SignalUrgency
    title: str
    description: str
    source: str                          # e.g. "Google Calendar", "Gmail", "Slack"
    timestamp: str                       # ISO 8601
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processed: bool = False

class Prediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str
    action_type: ActionType
    confidence: float = Field(ge=0.0, le=100.0)
    confidence_level: ConfidenceLevel
    trigger_signal_id: str
    trigger_context: str
    status: PredictionStatus = PredictionStatus.pending
    reward_if_correct: float = Field(default=1.0, ge=0.0, le=1.0)
    penalty_if_wrong: float = Field(default=0.3, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CalendarEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    start_time: str
    end_time: str
    attendees: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    is_conflicting: bool = False

class WorkflowRule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    trigger_type: SignalType
    condition: str
    action_template: str
    is_active: bool = True
    times_triggered: int = 0

class ActionRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    action_type: str
    description: str
    timestamp: str
    reward_earned: float = 0.0
    was_correct: Optional[bool] = None


# ── Environment State ─────────────────────────────────────────────────────────

class UserContext(BaseModel):
    timezone: str = "Asia/Kolkata"
    work_hours_start: int = 9
    work_hours_end: int = 18
    location: str = "Office"
    name: str = "User"

class EnvironmentState(BaseModel):
    """The full observable state returned by /state and /reset."""
    task_id: str
    step_number: int = 0
    max_steps: int = 20
    done: bool = False
    pending_signals: List[ContextSignal] = Field(default_factory=list)
    active_predictions: List[Prediction] = Field(default_factory=list)
    calendar_events: List[CalendarEvent] = Field(default_factory=list)
    recent_actions: List[ActionRecord] = Field(default_factory=list)
    workflow_rules: List[WorkflowRule] = Field(default_factory=list)
    current_time: str = ""
    user_context: UserContext = Field(default_factory=UserContext)
    cumulative_reward: float = 0.0
    info: Dict[str, Any] = Field(default_factory=dict)


# ── API Request / Response Models ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "signal_triage_easy"

class ActionPayload(BaseModel):
    action: str
    prediction_id: Optional[str] = None
    signal_id: Optional[str] = None
    title: Optional[str] = None
    start_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    attendees: Optional[List[str]] = None
    to: Optional[str] = None
    subject: Optional[str] = None
    body_outline: Optional[str] = None
    channel: Optional[str] = None
    message: Optional[str] = None
    trigger_type: Optional[str] = None
    condition: Optional[str] = None
    action_template: Optional[str] = None
    snooze_minutes: Optional[int] = None

class StepRequest(BaseModel):
    action: ActionPayload

class StepResponse(BaseModel):
    state: EnvironmentState
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: TaskDifficulty
    description: str
    max_steps: int
    reward_range: List[float]

class TasksResponse(BaseModel):
    tasks: List[TaskInfo]
