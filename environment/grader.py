"""
PFPA OpenEnv — Grader
Evaluates agent actions and computes rewards.
Implements partial-progress scoring across 3 difficulty levels.
"""
from __future__ import annotations
from typing import Tuple, Dict, Any
from environment.models import (
    EnvironmentState, ActionPayload, ActionRecord, Prediction,
    PredictionStatus, ConfidenceLevel, WorkflowRule, SignalType,
    ContextSignal
)
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")


class PFPAGrader:
    """
    Grades agent actions within a PFPA environment state.
    Returns (reward: float, done: bool, info: dict).
    
    Reward philosophy:
    - Executing correct (green/high-confidence) predictions:  +reward_if_correct
    - Dismissing low-confidence (red) predictions:            +0.05 to +0.15
    - Executing wrong/trap predictions:                       -penalty_if_wrong
    - Creating workflow rules (medium/hard tasks):            +0.1 to +0.2
    - NOOP on an empty queue:                                 +0.0 (neutral)
    - NOOP with urgent signals pending:                       -0.05 (opportunity cost)
    - Invalid action:                                         -0.05
    
    Final reward is clipped to [0.0, 1.0].
    """

    def grade(
        self,
        state: EnvironmentState,
        action: ActionPayload,
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Apply action to state (mutating it) and return (reward, done, info).
        """
        reward = 0.0
        info: Dict[str, Any] = {}
        now_str = datetime.now(IST).isoformat()

        act = action.action

        # ── execute_prediction ──────────────────────────────────────────────
        if act == "execute_prediction":
            pred = self._find_prediction(state, action.prediction_id)
            if pred is None:
                info["error"] = f"prediction_id '{action.prediction_id}' not found"
                reward = -0.05
            elif pred.status != PredictionStatus.pending:
                info["warning"] = f"Prediction already {pred.status}"
                reward = 0.0
            else:
                if pred.confidence_level == ConfidenceLevel.green:
                    reward = pred.reward_if_correct
                    info["outcome"] = "correct_execution"
                elif pred.confidence_level == ConfidenceLevel.yellow:
                    reward = pred.reward_if_correct * 0.5  # partial credit
                    info["outcome"] = "uncertain_execution"
                else:  # red — trap
                    reward = -pred.penalty_if_wrong
                    info["outcome"] = "false_positive_trap"
                pred.status = PredictionStatus.executed
                state.recent_actions.append(ActionRecord(
                    action_type="execute_prediction",
                    description=f"Executed: {pred.description}",
                    timestamp=now_str,
                    reward_earned=reward,
                    was_correct=reward > 0,
                ))
                self._mark_signal_processed(state, pred.trigger_signal_id)

        # ── dismiss_prediction ──────────────────────────────────────────────
        elif act == "dismiss_prediction":
            pred = self._find_prediction(state, action.prediction_id)
            if pred is None:
                info["error"] = f"prediction_id '{action.prediction_id}' not found"
                reward = -0.05
            elif pred.status != PredictionStatus.pending:
                info["warning"] = f"Prediction already {pred.status}"
                reward = 0.0
            else:
                if pred.confidence_level == ConfidenceLevel.red:
                    reward = 0.12  # correctly avoiding a false positive
                    info["outcome"] = "correct_dismissal"
                elif pred.confidence_level == ConfidenceLevel.yellow:
                    reward = 0.0  # neutral — could go either way
                    info["outcome"] = "neutral_dismissal"
                else:  # dismissing a green — mistake
                    reward = -pred.penalty_if_wrong * 0.5
                    info["outcome"] = "missed_opportunity"
                pred.status = PredictionStatus.dismissed
                state.recent_actions.append(ActionRecord(
                    action_type="dismiss_prediction",
                    description=f"Dismissed: {pred.description}",
                    timestamp=now_str,
                    reward_earned=reward,
                    was_correct=reward >= 0,
                ))

        # ── create_calendar_event ───────────────────────────────────────────
        elif act == "create_calendar_event":
            if not action.title or not action.start_time:
                reward = -0.05
                info["error"] = "create_calendar_event requires title and start_time"
            else:
                # Check if it resolves a conflict
                has_conflict = any(e.is_conflicting for e in state.calendar_events)
                reward = 0.15 if has_conflict else 0.08
                info["outcome"] = "calendar_event_created"
                state.recent_actions.append(ActionRecord(
                    action_type="create_calendar_event",
                    description=f"Created: {action.title}",
                    timestamp=now_str,
                    reward_earned=reward,
                    was_correct=True,
                ))
                # Resolve conflict
                for e in state.calendar_events:
                    e.is_conflicting = False

        # ── draft_email ─────────────────────────────────────────────────────
        elif act == "draft_email":
            if not action.to or not action.subject:
                reward = -0.05
                info["error"] = "draft_email requires 'to' and 'subject'"
            else:
                urgent_email = self._has_urgent_signal(state, SignalType.email_received)
                reward = 0.15 if urgent_email else 0.07
                info["outcome"] = "email_drafted"
                state.recent_actions.append(ActionRecord(
                    action_type="draft_email",
                    description=f"Drafted email to {action.to}: {action.subject}",
                    timestamp=now_str,
                    reward_earned=reward,
                    was_correct=True,
                ))

        # ── send_slack_message ──────────────────────────────────────────────
        elif act == "send_slack_message":
            if not action.channel or not action.message:
                reward = -0.05
                info["error"] = "send_slack_message requires 'channel' and 'message'"
            else:
                urgent_slack = self._has_urgent_signal(state, SignalType.slack_message)
                reward = 0.15 if urgent_slack else 0.06
                info["outcome"] = "slack_message_sent"
                state.recent_actions.append(ActionRecord(
                    action_type="send_slack_message",
                    description=f"Sent to #{action.channel}: {action.message[:50]}",
                    timestamp=now_str,
                    reward_earned=reward,
                    was_correct=True,
                ))

        # ── set_workflow_rule ───────────────────────────────────────────────
        elif act == "set_workflow_rule":
            if not action.trigger_type or not action.condition or not action.action_template:
                reward = -0.05
                info["error"] = "set_workflow_rule requires trigger_type, condition, action_template"
            else:
                # Bonus reward for creating a workflow — shows forward-thinking
                rule = WorkflowRule(
                    trigger_type=action.trigger_type,
                    condition=action.condition,
                    action_template=action.action_template,
                )
                state.workflow_rules.append(rule)
                reward = 0.20  # workflow creation is always high value
                info["outcome"] = f"workflow_rule_created: {rule.id}"
                state.recent_actions.append(ActionRecord(
                    action_type="set_workflow_rule",
                    description=f"Rule: if {action.condition} → {action.action_template}",
                    timestamp=now_str,
                    reward_earned=reward,
                    was_correct=True,
                ))

        # ── snooze_signal ───────────────────────────────────────────────────
        elif act == "snooze_signal":
            sig = self._find_signal(state, action.signal_id)
            if sig is None:
                reward = -0.05
                info["error"] = f"signal_id '{action.signal_id}' not found"
            else:
                sig.processed = True
                reward = 0.03
                info["outcome"] = f"signal snoozed for {action.snooze_minutes or 30} min"

        # ── noop ────────────────────────────────────────────────────────────
        elif act == "noop":
            urgent_pending = self._count_urgent_pending(state)
            if urgent_pending > 0:
                reward = -0.05 * min(urgent_pending, 3)
                info["outcome"] = f"noop with {urgent_pending} urgent signals pending"
            else:
                reward = 0.0
                info["outcome"] = "noop — nothing urgent"

        else:
            reward = -0.05
            info["error"] = f"Unknown action: '{act}'"

        # ── Update cumulative reward ────────────────────────────────────────
        reward = round(max(0.0, min(1.0, reward)), 4)
        state.cumulative_reward = round(
            min(1.0, state.cumulative_reward + reward), 4
        )
        state.step_number += 1

        # ── Termination conditions ──────────────────────────────────────────
        done = self._check_done(state)
        state.done = done

        info["step"] = state.step_number
        info["cumulative_reward"] = state.cumulative_reward
        info["remaining_steps"] = state.max_steps - state.step_number

        return reward, done, info

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _find_prediction(self, state: EnvironmentState, pred_id: str | None) -> Prediction | None:
        if not pred_id:
            return None
        for p in state.active_predictions:
            if p.id == pred_id:
                return p
        return None

    def _find_signal(self, state: EnvironmentState, sig_id: str | None) -> ContextSignal | None:
        if not sig_id:
            return None
        for s in state.pending_signals:
            if s.id == sig_id:
                return s
        return None

    def _mark_signal_processed(self, state: EnvironmentState, sig_id: str):
        for s in state.pending_signals:
            if s.id == sig_id:
                s.processed = True

    def _has_urgent_signal(self, state: EnvironmentState, stype: SignalType) -> bool:
        from environment.models import SignalUrgency
        return any(
            s.signal_type == stype and s.urgency == SignalUrgency.high and not s.processed
            for s in state.pending_signals
        )

    def _count_urgent_pending(self, state: EnvironmentState) -> int:
        from environment.models import SignalUrgency
        return sum(
            1 for s in state.pending_signals
            if s.urgency == SignalUrgency.high and not s.processed
        )

    def _check_done(self, state: EnvironmentState) -> bool:
        if state.step_number >= state.max_steps:
            return True
        # All predictions resolved and no unprocessed high-urgency signals
        pending_preds = [p for p in state.active_predictions if p.status == PredictionStatus.pending]
        from environment.models import SignalUrgency
        urgent_signals = [s for s in state.pending_signals if s.urgency == SignalUrgency.high and not s.processed]
        if not pending_preds and not urgent_signals:
            return True
        # Max reward reached
        if state.cumulative_reward >= 0.99:
            return True
        return False
