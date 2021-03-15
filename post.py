#!/usr/bin/env python
import ksienie as ks
import numpy as np
from scipy import stats
import latextabs as lt
from math import pi
from matplotlib import pyplot as plt

# Parameters
used_test = stats.wilcoxon
# used_test = stats.ranksums
used_p = 0.05

# Load results
legend = ks.json2object("results/legend.json")
datasets = legend["datasets"]
classifiers = legend["classifiers"]
metrics = legend["metrics"]
folds = legend["folds"]
rescube = np.load("results/rescube.npy")
# storage for ranks
ranks = np.zeros((len(metrics), len(datasets), len(classifiers)))

# First generate tables for each metric
for mid, metric in enumerate(metrics):

    table_file = open("results/tab_%s.tex" % metric, "w")
    table_file.write(lt.header4classifiers(classifiers))

    for did, dataset in enumerate(datasets):
        dataset = dataset.replace("_", "-")
        print(dataset)
        # continue

        # Subtable is 2d (clf, fold)
        subtable = rescube[did, :, mid, :]

        # Check if metric was valid
        if np.isnan(subtable).any():
            print("Unvaild")
            continue

        # Scores as mean over folds
        scores = np.mean(subtable, axis=1)
        stds = np.std(subtable, axis=1)

        # ranks
        rank = stats.rankdata(scores, method='average')
        ranks[mid, did] = rank

        # Get leader and check dependency
        # dependency = np.zeros(len(classifiers)).astype(int)
        dependency = np.zeros((len(classifiers), len(classifiers)))

        for cida, clf_a in enumerate(classifiers):
            a = subtable[cida]
            for cid, clf in enumerate(classifiers):
                b = subtable[cid]
                test = used_test(a, b, zero_method="zsplit")
                dependency[cida, cid] = test.pvalue > used_p

        table_file.write(lt.row(dataset, scores, stds))
        table_file.write(lt.row_stats(dataset, dependency, scores, stds))

    table_file.write(lt.footer("Results for %s metric" % metric))
    table_file.close()

for i, metric in enumerate(metrics):
    table_file = open("results/tab_%s_mean_ranks.tex" % metric, "w")
    table_file.write(lt.header4classifiers_ranks(classifiers))
    dependency2 = np.zeros((len(classifiers), len(classifiers)))
    for cida in range(len(classifiers)):
        a = ranks[i].T[cida]
        for cid in range(len(classifiers)):
            b = ranks[i].T[cid]
            test = used_test(a, b, zero_method="zsplit")
            dependency2[cida, cid] = test.pvalue > used_p

    table_file.write(lt.row_ranks(np.mean(ranks[i], axis=0)))
    table_file.write(lt.row_stats(dataset, dependency2, np.mean(ranks[i], axis=0), np.zeros((7))))
    table_file.write(lt.footer("Results for mean ranks according to %s metric" % metric))
    table_file.close()

# ------------------
# Radar rank
# ------------------
N = len(metrics)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# No shitty border
ax.spines["polar"].set_visible(False)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles, metrics)

# colors = [(0.1, 0.3, 0.9), (0.1, 0.5, 0.1), (0.1, 0.5, 0.1)]
# # colors = [(0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1)]
# ls = ["-", "-.", "--"]
# lw = [1, 0.5, 0.5]

colors = [(0.1, 0.3, 0.9), (0.1, 0.5, 0.1), (0.1, 0.5, 0.1), (0.1, 0.5, 0.1), (0.1, 0.5, 0.1), (0.5, 0.1, 0.1), (0.5, 0.1, 0.1), (0.5, 0.1, 0.1), (0.5, 0.1, 0.1)]
# colors = [(0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1)]
ls = ["-", "-", ":", "-.", "--", "-", ":", "-.", "--"]
lw = [1, 0.5, 0.7, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5]

new_ranks = []
for mid, metric in enumerate(metrics):
    new_ranks.append(np.mean(ranks[mid], axis=0).tolist())

new_ranks.append(np.mean(ranks[0], axis=0).tolist())

print(new_ranks)

classifiers = [
               "EUC",
               "MAN",
               "CHE",
               "JSH",
               "BRY",
               "CAN",
               "COR",
               "SQE",
              ]

for i, (clf_name, rank) in enumerate(zip(classifiers, np.array(new_ranks).T)):
    ax.plot(angles, rank, label=clf_name, c=colors[i], ls=ls[i], lw=lw[i])

# Add legend
plt.legend(
    loc="lower center",
    ncol=3,
    columnspacing=1,
    frameon=False,
    bbox_to_anchor=(0.5, -0.32),
    fontsize=6,
)

# Add a grid
plt.grid(ls=":", c=(0.7, 0.7, 0.7))

# Add a title
plt.tight_layout()

# Draw labels
a = np.linspace(1, 7, 7)
plt.yticks(a[1:], ["%.1f" % f for f in a[1:]], fontsize=6, rotation=90)
plt.ylim(1.0, 7.0)
plt.gcf().set_size_inches(4, 3.5)
plt.gcf().canvas.draw()
angles = np.rad2deg(angles)

ax.set_rlabel_position((angles[0] + angles[1]) / 2)

har = [(a >= 90) * (a <= 270) for a in angles]

for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
    x, y = label.get_position()
    lab = ax.text(
        x, y, label.get_text(), transform=label.get_transform(), fontsize=6,
    )
    lab.set_rotation(angle)

    if har[z]:
        lab.set_rotation(180 - angle)
    else:
        lab.set_rotation(-angle)
    lab.set_verticalalignment("center")
    lab.set_horizontalalignment("center")
    lab.set_rotation_mode("anchor")

for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
    x, y = label.get_position()
    lab = ax.text(
        x,
        y,
        label.get_text(),
        transform=label.get_transform(),
        fontsize=4,
        c=(0.7, 0.7, 0.7),
    )
    lab.set_rotation(-(angles[0] + angles[1]) / 2)

    lab.set_verticalalignment("bottom")
    lab.set_horizontalalignment("center")
    lab.set_rotation_mode("anchor")

ax.set_xticklabels([])
ax.set_yticklabels([])

plt.savefig("results/radar.png", bbox_inches='tight', dpi=500, pad_inches=0.0)
plt.savefig("results/radar.eps", bbox_inches='tight', dpi=500, pad_inches=0.0)
plt.close()
