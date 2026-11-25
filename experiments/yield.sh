#!/bin/bash
# Pause every ts-* match unit while you play: the Steam game (Dominion.exe) or the advisor (advisor.py) is running.
# Resumes them (SIGCONT) when both are gone. Runs as the systemd user unit ts-yield; checks every 10 s.
# Log: logs/yield.log (one line per pause / resume).
cd "$(dirname "$0")/.."
paused=0
while true; do
  if pgrep -f 'Dominion\.exe' > /dev/null || pgrep -f 'autopilot/advisor\.py' > /dev/null; then
    if [ $paused = 0 ]; then
      for u in $(systemctl --user list-units 'ts-*' --state=running --plain --no-legend | awk '{print $1}' | grep -v 'ts-yield'); do
        systemctl --user kill -s SIGSTOP "$u"
      done
      paused=1; echo "$(date '+%F %T') paused (game or advisor running)" >> logs/yield.log
    fi
  elif [ $paused = 1 ]; then
    for u in $(systemctl --user list-units 'ts-*' --state=running --plain --no-legend | awk '{print $1}' | grep -v 'ts-yield'); do
      systemctl --user kill -s SIGCONT "$u"
    done
    paused=0; echo "$(date '+%F %T') resumed" >> logs/yield.log
  fi
  sleep 10
done
