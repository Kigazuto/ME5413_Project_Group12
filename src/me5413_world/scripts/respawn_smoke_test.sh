#!/bin/bash
# respawn_smoke_test.sh — regression test for object spawner
# Requires: Gazebo world.launch already running
# Usage:  bash respawn_smoke_test.sh
set -uo pipefail

READY=/me5413_world/objects_ready
STATUS=/me5413_world/respawn_status
CMD=/rviz_panel/respawn_objects

pass=0; fail=0; total=0

# ── helpers ──────────────────────────────────────────────────────────────────

get_ready() {
  # Returns "True" or "False" (or empty on timeout)
  timeout 3 rostopic echo -n1 "$READY" 2>/dev/null \
    | grep -oP '(?<=data: )\w+' || true
}

get_status() {
  # Returns the status string (or empty on timeout)
  timeout 3 rostopic echo -n1 "$STATUS" 2>/dev/null \
    | grep -oP '(?<=data: ").*(?=")' || true
}

wait_status() {
  # wait_status <pattern> [timeout_sec]
  local pat="$1" deadline=$((SECONDS + ${2:-45}))
  while [ $SECONDS -lt $deadline ]; do
    local s; s=$(get_status)
    if [[ "$s" == *"$pat"* ]]; then return 0; fi
    sleep 1
  done
  return 1
}

send_cmd() {
  rostopic pub -1 "$CMD" std_msgs/Int16 "data: $1" >/dev/null 2>&1
}

result() {
  ((total++))
  if [ "$1" = "PASS" ]; then ((pass++)); else ((fail++)); fi
  printf "  %-6s  ready=%-5s  status=%s\n" "$1" "$(get_ready)" "$(get_status)"
}

# ── sanity ───────────────────────────────────────────────────────────────────

echo "=== Smoke Test: Object Spawner ==="
echo ""

# Make sure Gazebo topics exist
if ! timeout 5 rostopic info "$READY" >/dev/null 2>&1; then
  echo "ERROR: $READY not found — is world.launch running?"
  exit 2
fi

# ── Test 1: cmd=1 → RESPAWN_OK + ready=True ─────────────────────────────────

echo "[Test 1] cmd=1 → RESPAWN_OK, ready=True"
send_cmd 1 &
if wait_status "RESPAWN_OK" 45 && [ "$(get_ready)" = "True" ]; then
  result "PASS"
else
  result "FAIL"
fi

# ── Test 2: rapid 1,0,1 → last-wins → RESPAWN_OK + ready=True ──────────────

echo "[Test 2] Rapid 1,0,1 → last-wins → RESPAWN_OK, ready=True"
send_cmd 1 &
sleep 0.05
send_cmd 0 &
sleep 0.05
send_cmd 1 &

# The first cmd=1 starts a worker. cmd=0 and cmd=1 overwrite pending.
# After first worker finishes, pending=1 triggers a second cycle.
# We wait for the final RESPAWN_OK.
# First wait for the initial respawn to start (status changes from RESPAWN_OK)
sleep 2
if wait_status "RESPAWN_OK" 90 && [ "$(get_ready)" = "True" ]; then
  result "PASS"
else
  result "FAIL"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Results: $pass/$total passed ==="
if [ "$fail" -eq 0 ]; then
  echo "PASS"
  exit 0
else
  echo "FAIL"
  exit 1
fi
