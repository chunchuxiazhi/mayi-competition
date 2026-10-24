# R2 baseline running — startup confirmed

Confirmed UTC: 2026-10-15T18:21:25.878395+00:00

Real two-worker smoke passed. lift_pot seed100001 completed400steps and hanging_mug seed100000 completed900steps, both valid policy failures. The previously completed lift_pot failure is retained. No repeated expert check or substitute seed was used for these cases. Two distinct sessions overlapped, and every action chunk's session/seed/counter assertion passed.

Full50-task clean+seen queue is now running on one resident original pretrained model with two simulator workers. Both workers have produced verified first actions beyond their smoke slices. Counts at this confirmation: 3 valid policy rounds, 0 successes, planned500. No completed failure is retried. This is a10-round quick baseline, not standard100-round or competition score.

Results: /home/zzj/competitions/lingbot/runs/pretrained-clean-quick-50x10-20261016
tmux: lingbot-clean-quick-r2-20261016
Live source of status: execution-status.json, task ledgers and summary.json.

Static checks and8regressions passed. Expert collection quota scope and normalization method interpretation remain formally unconfirmed. Final competition model requires joint training on50clean tasks. No monitoring automation was installed; after this startup confirmation the server runs independently.
