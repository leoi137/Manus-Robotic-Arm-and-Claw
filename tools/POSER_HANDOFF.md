# Handoff — arm_poser.html debugging session

You own `tools/arm_poser.html` (single-file WebGL posing tool for the SO-101
arm) and nothing else. Goal: the user must be able to pose the arm on the
standing cylinder, click **▶ CLOSE GRIPPER**, watch the jaw close with
quasi-static physics (CLAMPED / TIPPED / SHOVED verdicts), and on CLAMPED have
the object attach and move with the arm until the gripper reopens.

## Current state (2026-08-13)

- The FK core is verified against `src/manus/kinematics.py` to ~1e-7 (do NOT
  restructure it). Jaw swing model, close-sim physics (section 8b), and
  clamped-carry (`state.hold`) are implemented and were verified.
- Sidebar subtitle shows **v7** — that's the version marker; if the user's
  browser doesn't show v7, they're on a stale page.
- Git history of the file: `git log --oneline -- tools/arm_poser.html`.

## THE OPEN BUG

The user reports the CLOSE GRIPPER button **does nothing at all** in their
real desktop Chrome, even after (they say) refreshing. Yet a real-interaction
CDP test on this exact machine and file (headless Chrome, real
Input.dispatchMouseEvent on the button, real time, no virtual-time flags)
shows it working: click registers, gripper animates 1.5 → 0.21 rad, verdict
renders, zero exceptions. Test driver scripts live in the session scratchpad
(drive2.js pattern: chrome --headless=new --remote-debugging-port=9333, npm
package chrome-remote-interface).

So either (a) the user's tab/browser serves a stale copy, (b) their
GPU/driver/Chrome-version combo breaks something ours doesn't, or (c) there's
an interaction sequence we haven't reproduced (e.g. a slider interaction that
wedges state before the click). Do not assume — verify IN THE USER'S ACTUAL
BROWSER. First steps: have the user confirm the v7 marker; if the Claude in
Chrome extension is available connect to their live tab (tabs_context_mcp)
and read the console + click state directly; otherwise have them open
DevTools (F12 → Console) and paste what appears when they click.

Known past bugs in this file, all fixed — don't re-fix, but they show the
failure classes: initial dirty-flag deadlock (nothing ever rendered), CSS
inset:auto mis-positioning, jaw drawn sliding instead of swinging (gripper
grew huge), rAF-timestamp-before-click-timestamp crash killing the close
animation permanently (intermittent, invisible under --virtual-time-budget),
silent no-repaint on empty-hand verdicts.

Verification bar: real mouse events, real time, screenshots that visibly show
the jaw closing — and ultimately the user saying it works in THEIR browser.
