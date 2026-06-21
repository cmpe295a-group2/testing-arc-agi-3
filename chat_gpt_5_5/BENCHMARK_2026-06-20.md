# Anonymous public benchmark — 2026-06-20

This is an evaluator report, not policy input. Public labels appear here only
after each run so failures can be reproduced. `GeneralistAgent` received anonymous
observations and a capability-limited black-box simulator.

```powershell
.\.venv-arc\Scripts\python.exe -m chat_gpt_5_5.evaluate `
  --mode simulator --bfs 0.10 --search 0.90 --cap 8 `
  --actions 500 --nodes 50000 --depth 120 --clicks 32
```

| Public label | Anonymous label | Levels | Real actions | Simulated nodes | RHAE |
|---|---:|---:|---:|---:|---:|
| ar25 | env-001 | 0/8 | 6 | 988 | 0.00% |
| bp35 | env-002 | 0/9 | 6 | 1,015 | 0.00% |
| cd82 | env-003 | 0/6 | 7 | 1,337 | 0.00% |
| cn04 | env-004 | 0/6 | 8 | 2,701 | 0.00% |
| dc22 | env-005 | 0/6 | 7 | 1,224 | 0.00% |
| ft09 | env-006 | 0/6 | 6 | 704 | 0.00% |
| g50t | env-007 | 0/7 | 8 | 740 | 0.00% |
| ka59 | env-008 | 0/7 | 7 | 1,692 | 0.00% |
| lf52 | env-009 | 0/10 | 7 | 1,517 | 0.00% |
| lp85 | env-010 | **1/8** | 7 | 609 | **2.78%** |
| ls20 | env-011 | 0/7 | 8 | 292 | 0.00% |
| m0r0 | env-012 | 0/6 | 7 | 989 | 0.00% |
| r11l | env-013 | 0/6 | 7 | 1,568 | 0.00% |
| re86 | env-014 | 0/8 | 8 | 810 | 0.00% |
| s5i5 | env-015 | 0/8 | 8 | 1,376 | 0.00% |
| sb26 | env-016 | 0/8 | 6 | 952 | 0.00% |
| sc25 | env-017 | 0/6 | 6 | 828 | 0.00% |
| sk48 | env-018 | 0/8 | 7 | 814 | 0.00% |
| sp80 | env-019 | 0/6 | 7 | 1,443 | 0.00% |
| su15 | env-020 | 0/9 | 7 | 1,023 | 0.00% |
| tn36 | env-021 | 0/7 | 5 | 576 | 0.00% |
| tr87 | env-022 | 0/6 | 8 | 320 | 0.00% |
| tu93 | env-023 | 0/9 | 8 | 456 | 0.00% |
| vc33 | env-024 | **1/7** | 10 | 1,563 | **3.57%** |
| wa30 | env-025 | 0/9 | 8 | 710 | 0.00% |

Total: **2/183 levels, 0/25 games, 0.25% mean game RHAE**.

## Reading the result

- The solved tutorial levels show that the anonymous action/perception/search
  pipeline is connected correctly.
- Twenty-three first levels remained unsolved. More raw search is therefore not a
  credible primary plan; mechanic and win-predicate induction must improve.
- Wall-clock caps are checked between real decisions, so individual runs can exceed
  eight seconds by the duration of one final search call.
- This score must never be combined with the deleted answer-replay result.
