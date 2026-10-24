# About the owner (for later sessions)

What I learned about how the owner likes to work. Update it whenever the owner contradicts me or states a preference.

- **Reading is tiring.** They said "i find it hard to read everything". Keep replies short, lead with the result, and use `AskUserQuestion` (2 to 4 short choices) to get decisions instead of long explanations or tables of options. Put the detail in files (`log.md`, plans) and link it.
- **"status" means a short status**, nothing more: what is running, how far, any failure.
- **Finish first, then experiment.** Build the tools and the demo completely, then experiment freely with inputs, methods, physics and rewards.
- **Quality of the demo matters a lot** ("make the game polish real good"): visuals, effects, menus, camera and sound should feel finished.
- **Everything must be mergeable.** While a queue runs in `main`, do not touch code there; work in a worktree, keep its files disjoint from what the queue writes, merge `main` in regularly, and tell the owner when it is merged.
- **Audio:** on when watching a race, off for training and analysis, always toggleable in the window.
- **Plots and analysis should be live** in the sandbox (the same Python runtime), not export-then-plot.
- **Decisions are theirs.** Ask when a choice changes scope or an old decision; otherwise pick the sensible default, say which, and list it in the final message.
- They like seeing evidence: screenshots and real browser checks of new UI, and measured numbers (frame rates, test counts).
