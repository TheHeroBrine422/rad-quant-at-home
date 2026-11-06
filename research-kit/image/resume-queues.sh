#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# resume-queues.sh --local [--status|--dry|--kv|--if-boot] -- relaunch the research queues after a container start.
#
# Donors stop the container whenever they like (docker stop, reboots); after the next start nothing of ours runs until
# someone relaunches it. The entrypoint runs this ~15 s after every start. Idempotent and safe to run any time: one
# resume at a time (flock), no duplicate launches (a running queue holds its own lock; MATCH below). Same registry
# format as on our own preemptible queue hosts.
#
# Registry: $QUEUE_D (default /data/queue.d), one file per queue, NAME.sh, run in name order (number them: 10-sweep.sh,
# 20-collect.sh) with a 3 s gap, so a follower that waits on its predecessor with pgrep sees it. Bash, sourced:
#   CMD='bash /data/work/x1/run.sh'                       # the launch line (bash -c, cwd $CWD or /tmp; must be resumable)
#   DONE='grep -q "X1 DONE" /data/work/x1/result.txt'      # exit 0 = finished: never relaunched
#   LOG=/data/work/x1/run.out                             # stdout + stderr appended (default $QUEUE_D/log/NAME.log)
#   MATCH='^bash /data/work/x1/[r]un\.sh( |$)'            # optional: pgrep -f pattern of an instance started by hand
#                                                         #   (default: CMD itself, anchored)
#   AFTER=10-sweep                                        # optional: launch only while that queue runs or is done
#   CWD=/data/work/x1                                     # optional
# Off switches: $QUEUE_D/PAUSE (nothing is launched), $QUEUE_D/NAME.off (that queue is skipped). A queue stopped on
# purpose must get its .off (or lose its entry), or the next start relaunches it.
# Launch: setsid nohup, stdio to LOG, the queue holding $QUEUE_D/.lock/NAME.lock (inherited by its children: a queue
# whose driver died but whose python still runs counts as running).
# Every run removes stale torch-extension build locks: TORCH_EXTENSIONS_DIR/*/lock older than the container's start (a
# kill inside a JIT build leaves one, and every later load waits on it forever). Nothing from before the start is alive,
# so such a lock is always stale.
# --status: one line per queue (NAME running|done|waiting|paused|off|stopped); --kv: a summary (n= registered, miss=
# neither running nor done, names=...); --dry: what would be launched; --if-boot: a normal run, but only if the
# container started since the last run (else exit at once). Log: $QUEUE_D/resume.log.
#
# Writing a queue that survives kills: every step writes its output as NAME.part, then renames it (atomic), and is
# skipped when its output exists; no .done marker before the data; long steps checkpoint the same way.
[ "${1:-}" = --local ] || { echo "usage: resume-queues.sh --local [--status|--dry|--kv|--if-boot]" >&2; exit 2; }
shift; MODE=${1:-run}
QUEUE_D=${QUEUE_D:-/data/queue.d}; TEXT=${TORCH_EXTENSIONS_DIR:-/data/torch_ext}
if [ "$MODE" = --if-boot ]; then [ "$(awk '{print $22}' /proc/1/stat 2> /dev/null)" != "$(cat "$QUEUE_D/.boot" 2> /dev/null)" ] || exit 0; MODE=run; fi
[ -f /etc/profile.d/research-env.sh ] && . /etc/profile.d/research-env.sh > /dev/null 2>&1 < /dev/null
mkdir -p "$QUEUE_D/.lock" "$QUEUE_D/log" 2> /dev/null || { echo "resume-queues: cannot write $QUEUE_D" >&2; exit 1; }
say() { echo "$(date -u +%FT%TZ) $*" >> "$QUEUE_D/resume.log"; [ "$MODE" = --kv ] || echo "$*"; }

# entry NAME: load its fields (in the caller's shell; callers run this in a subshell)
entry() { CMD= DONE= LOG= MATCH= AFTER= CWD=; . "$QUEUE_D/$1.sh"
  LOG=${LOG:-$QUEUE_D/log/$1.log}; [ -n "$MATCH" ] || MATCH="^$(printf '%s' "$CMD" | sed 's/[][\.*^$?+(){}|/]/\\&/g')( |\$)"; }
running() { ! flock -n "$QUEUE_D/.lock/$1.lock" true || { [ -n "$MATCH" ] && pgrep -f -- "$MATCH" > /dev/null; }; }
isdone() { [ -n "$DONE" ] && timeout 60 bash -c "$DONE" > /dev/null 2>&1 < /dev/null; }
state() { ( entry "$1"
  if [ -f "$QUEUE_D/$1.off" ]; then echo off; elif running "$1"; then echo running; elif isdone; then echo done
  elif [ -n "$AFTER" ] && ! { [ -f "$QUEUE_D/$AFTER.sh" ] && case $(state "$AFTER") in running|done) true;; *) false;; esac; }; then echo waiting
  elif [ -f "$QUEUE_D/PAUSE" ]; then echo paused; else echo stopped; fi ); }
names() { for f in "$QUEUE_D"/*.sh; do [ -f "$f" ] && basename "$f" .sh; done; }

if [ "$MODE" = --status ] || [ "$MODE" = --kv ]; then
  n=0 miss=0 mn=
  for q in $(names); do s=$(state "$q"); n=$((n + 1))
    [ "$MODE" = --status ] && echo "$q $s"
    case $s in running|done|off) ;; *) miss=$((miss + 1)); mn="$mn${mn:+,}$q";; esac; done
  [ "$MODE" = --kv ] && echo "n=$n" && echo "miss=$miss" && echo "names=${mn:--}"
  exit 0; fi

exec {G}> "$QUEUE_D/.resume.lock"; flock -w 120 $G || { say "another resume holds $QUEUE_D/.resume.lock: skipped"; exit 0; }
boot=$(awk '{print $22}' /proc/1/stat 2> /dev/null)
t1=$(( $(awk '/^btime/ {print $2}' /proc/stat) + boot / $(getconf CLK_TCK) ))   # the container's start, epoch s
for l in "$TEXT"/*/lock; do [ -f "$l" ] && [ "$(stat -c %Y "$l")" -lt "$t1" ] || continue
  if [ "$MODE" = --dry ]; then say "would remove stale torch-extension lock $l"; else rm -f "$l" && say "removed stale torch-extension lock $l (older than the container start)"; fi; done
if [ "$boot" != "$(cat "$QUEUE_D/.boot" 2> /dev/null)" ] && [ "$MODE" != --dry ]; then
  echo "$boot" > "$QUEUE_D/.boot"; say "boot: container start $(date -u -d @$t1 +%FT%TZ) seen"; fi
[ -f "$QUEUE_D/PAUSE" ] && { say "PAUSE set: nothing launched"; exit 0; }
for q in $(names); do s=$(state "$q")
  if [ "$s" != stopped ]; then say "$q: $s"; continue; fi
  if [ "$MODE" = --dry ]; then ( entry "$q"; say "$q: would launch: $CMD" ); continue; fi
  ( entry "$q"; mkdir -p "$(dirname "$LOG")"; cd "${CWD:-/tmp}" || { say "$q: cannot cd to ${CWD:-/tmp}"; exit 1; }
    echo "== $(date -u +%FT%TZ) resume-queues.sh: launching $q" >> "$LOG"
    exec {G}>&-   # the resume lock stays here, not in the queue
    # a simple command (no && list) in the background: nothing between us and the queue keeps our stdout open
    setsid nohup bash -c 'exec {lk}> "$1"; flock -n $lk || exit 0; exec bash -c "$2"' _ \
      "$QUEUE_D/.lock/$q.lock" "$CMD" >> "$LOG" 2>&1 < /dev/null &
    say "$q: launched (pid $!): $CMD" )
  sleep 3; done
exit 0
