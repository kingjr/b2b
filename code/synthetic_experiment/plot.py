import argparse
import numpy as np

from ast import literal_eval
from matplotlib import pyplot as plt

stats = {
    "result_error_in_all": "error in-domain",
    "result_error_out_all": "error out-domain",
    "result_error_in_mask": "error in-domain (masked)",
    "result_error_out_mask": "error out-domain (masked)",
    "result_false_positives": "false positives",
    "result_false_negatives": "false negatives",
    "result_auc": "AUC"
}

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times}')
plt.rc('font', family='serif')
plt.rc('font', size=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JRR synthetic experiment')
    parser.add_argument('--file', type=str, default='results_05.txt')
    args = parser.parse_args()

    with open(args.file, "r") as f:
        lines = f.readlines()

    result_names = []
    first_result = literal_eval(lines[0])

    for k in first_result.keys():
        if "result" in k:
            result_names.append(k)

    results = {}
    models = set()

    for line in lines:
        result_dict = {}
        line_dict = literal_eval(line)
        line_str = ""
        for k, v in line_dict.items():
            if ("result" not in k) and ("model" not in k) and ("seed" not in k):
                line_str += f"{k}={v}_"
            elif k == "model":
                this_model = v
                models.add(v)
            elif k == "seed":
                this_seed = int(v)
            else:
                result_dict[k] = float(v)

        line_str = line_str[:-1]

        if line_str not in results:
            results[line_str] = {}
        if this_model not in results[line_str]:
            results[line_str][this_model] = {
                "mean": np.zeros(len(result_names)),
                "power": np.zeros(len(result_names)),
                "counter": 0
            }

        for r, result in enumerate(result_names):
            results[line_str][this_model]["mean"][r] += result_dict[result]
            results[line_str][this_model]["power"][r] += result_dict[result] ** 2

        results[line_str][this_model]["counter"] += 1

    for eid in results:
        for model in results[eid]:
            results[eid][model]["mean"] /= results[eid][model]["counter"]
            results[eid][model]["power"] /= results[eid][model]["counter"]
            results[eid][model]["power"] -= results[eid][model]["mean"] ** 2
            results[eid][model]["variance"] = results[eid][model]["power"]
            results[eid][model].pop("power")
            results[eid][model].pop("counter")

    models.discard("JRR")

    plot_rows = len(models)
    plot_columns = len(stats)
    plot_counter = 1

    plt.figure(figsize=(len(stats) * 2 + 0.85, len(models) * 2))

    print(models)

    for m, model in enumerate(models):
        for r, result in enumerate(result_names):
            plt.subplot(plot_rows, plot_columns, plot_counter)
            plot_counter += 1

            if r == 0:
                plt.ylabel(model.upper())
            if m == 0:
                plt.title(stats[result], fontsize=10)

            for eid in results:
                if ("JRR" in results[eid]) and (model in results[eid]):
                    competitor_mean = results[eid][model]["mean"][r]
                    competitor_variance = results[eid][model]["variance"][r]

                    jrr_mean = results[eid]["JRR"]["mean"][r]
                    jrr_variance = results[eid]["JRR"]["variance"][r]

                    # plt.errorbar(jrr_mean,
                    #              competitor_mean,
                    #              xerr=jrr_variance,
                    #              yerr=competitor_variance)
                    plt.scatter(jrr_mean, competitor_mean, c="black", alpha=0.5)

            ax = plt.gca()
            xl = ax.get_xlim()
            yl = ax.get_ylim()

            min_ = min(min(xl), min(yl))
            max_ = max(max(xl), max(yl))

            ax.plot([min_, max_], [min_, max_], ls="--", c=".8")
            ax.margins(0)
    plt.tight_layout(0, 0, 0)
    plt.savefig(args.file + ".pdf")
