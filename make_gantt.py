import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
wps = [("WP1 Management & data", 0, 18), ("WP2 Test sets & CS-Audit tool", 0, 8),
       ("WP3 Zero-shot evaluation", 5, 12), ("WP4 Fine-tuning study", 9, 17),
       ("WP5 Dissemination & transfer", 13, 18)]
fig, ax = plt.subplots(figsize=(9, 2.6))
for k, (n, s, e) in enumerate(wps[::-1]):
    ax.barh(k, e - s, left=s, color="#1f4e79", height=.55)
ax.set_yticks(range(len(wps)), [w[0] for w in wps[::-1]], fontsize=9)
for m, x in (("M1", 8), ("M2", 12), ("M3", 17), ("M4", 18)):
    ax.axvline(x, ls=":", c="#c0392b", lw=1); ax.text(x, len(wps) - .35, m, ha="center", fontsize=8, color="#c0392b")
ax.set_xlim(0, 18.5); ax.set_xticks(range(0, 19, 3)); ax.set_xlabel("Month")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig_gantt.png", dpi=200)
