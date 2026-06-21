#!/usr/bin/env bash
set -euo pipefail

GATE="${1:-}"
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${TASK_DIR}/pbr-state.json"

fail() {
  echo "BLOCKED[$GATE]: $1" >&2
  exit 2
}

need_file() {
  [ -f "$1" ] || fail "missing file: $1"
}

json_get() {
  python3 - "$STATE_FILE" "$1" <<'PY'
import json, sys
p, key = sys.argv[1], sys.argv[2]
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)
val = data
for part in key.split('.'):
    val = val.get(part)
print('true' if val is True else 'false' if val is False else '' if val is None else val)
PY
}

need_state() {
  need_file "$STATE_FILE"
}

case "$GATE" in
  after-init)
    need_state
    need_file "${TASK_DIR}/pbr-gate.sh"
    ;;

  before-plan-confirm)
    need_state
    need_file "${TASK_DIR}/plan.md"
    [ "$(json_get plan_created)" = "true" ] || fail "plan_created is not true"
    ;;

  before-git-confirm)
    need_state
    need_file "${TASK_DIR}/plan.md"
    [ "$(json_get plan_confirmed)" = "true" ] || fail "plan not confirmed by human"
    ;;

  before-plan-commit)
    need_state
    [ "$(json_get plan_confirmed)" = "true" ] || fail "plan not confirmed"
    [ "$(json_get git_info_confirmed)" = "true" ] || fail "git info not confirmed"
    ;;

  before-plan-quiz)
    need_state
    [ "$(json_get plan_committed)" = "true" ] || fail "plan not committed"
    need_file "${TASK_DIR}/plan.md"
    ;;

  before-impl-loop)
    need_state
    need_file "${TASK_DIR}/plan.md"
    [ "$(json_get plan_quiz_passed)" = "true" ] || fail "Plan Quiz has not passed"
    ;;

  before-impl-round)
    need_state
    [ "$(json_get plan_quiz_passed)" = "true" ] || fail "Plan Quiz has not passed"
    N="$(json_get impl_round)"
    [ -n "$N" ] || fail "impl_round missing"
    if [ "$N" != "0" ]; then
      PREV=$((N-1))
      need_file "${TASK_DIR}/impl/round-${PREV}-review-result.md"
    fi
    ;;

  after-impl-build)
    need_state
    N="$(json_get impl_round)"
    need_file "${TASK_DIR}/impl/round-${N}-summary.md"
    grep -q "本轮实现内容" "${TASK_DIR}/impl/round-${N}-summary.md" || fail "summary missing 本轮实现内容"
    grep -q "AC 推进情况" "${TASK_DIR}/impl/round-${N}-summary.md" || fail "summary missing AC 推进情况"
    grep -q "遗留问题" "${TASK_DIR}/impl/round-${N}-summary.md" || fail "summary missing 遗留问题"
    grep -q "Goal Tracker 更新请求" "${TASK_DIR}/impl/round-${N}-summary.md" || fail "summary missing Goal Tracker 更新请求"
    grep -q "Lesson Delta" "${TASK_DIR}/impl/round-${N}-summary.md" || fail "summary missing Lesson Delta"
    ;;

  after-impl-review)
    need_state
    N="$(json_get impl_round)"
    need_file "${TASK_DIR}/impl/round-${N}-summary.md"
    need_file "${TASK_DIR}/impl/round-${N}-review-result.md"
    head -n 1 "${TASK_DIR}/impl/round-${N}-review-result.md" | grep -Eq '^(COMPLETE|STOP|\[ISSUE\])' || fail "review-result first line must be COMPLETE, STOP, or [ISSUE]"
    ;;

  before-review-loop)
    need_state
    [ "$(json_get impl_loop_complete)" = "true" ] || fail "IMPL LOOP not complete"
    ;;

  after-review-diff)
    need_state
    N="$(json_get review_round)"
    need_file "${TASK_DIR}/review/round-${N}-full-diff.txt"
    grep -q "=== git status --short ===" "${TASK_DIR}/review/round-${N}-full-diff.txt" || fail "full diff missing git status section"
    grep -q "=== working tree diff" "${TASK_DIR}/review/round-${N}-full-diff.txt" || fail "full diff missing working tree diff section"
    ;;

  after-review-result)
    need_state
    N="$(json_get review_round)"
    need_file "${TASK_DIR}/review/round-${N}-review-result.md"
    head -n 1 "${TASK_DIR}/review/round-${N}-review-result.md" | grep -Eq '^(NO_ISSUES|\[P[0-9]\])' || fail "review-result first line must be NO_ISSUES or [P0-P9]"
    ;;

  after-review-fix)
    need_state
    N="$(json_get review_round)"
    need_file "${TASK_DIR}/review/round-${N}-fix-summary.md"
    grep -q "修复内容" "${TASK_DIR}/review/round-${N}-fix-summary.md" || fail "fix summary missing 修复内容"
    grep -q "修复后的文件变更" "${TASK_DIR}/review/round-${N}-fix-summary.md" || fail "fix summary missing 修复后的文件变更"
    ;;

  before-finalize)
    need_state
    [ "$(json_get review_loop_complete)" = "true" ] || fail "REVIEW LOOP not complete"
    ;;

  before-settlement)
    need_state
    [ "$(json_get final_commit_done)" = "true" ] || fail "final commit not done"
    ;;

  *)
    fail "unknown gate. Usage: $0 <gate-name>"
    ;;
esac

echo "PASS[$GATE]"
