import argparse
import numpy as np

from ast import literal_eval
from matplotlib import pyplot as plt

stats = {
    "result_error_in_all" : "error in-domain",
    "result_error_out_all" : "error out-domain",
    "result_error_in_mask" : "error in-domain (masked)",
    "result_error_out_mask" : "error out-domain (masked)",
    "result_false_positives" : "false positives",
    "result_false_negatives" : "false_negatives"
}
        
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{times}')
plt.rc('font', family='serif')
plt.rc('font', size=16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JRR synthetic experiment')
    parser.add_argument('--file', type=str, default='result_jrr.txt')
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

    models.discard("jrr")

    for r, result in enumerate(result_names):
        for model in models:
            ax = plt.figure(figsize=(4, 4))
            plt.title(stats[result])
            plt.xlabel(model.upper())
            plt.ylabel("jrr".upper())

            for eid in results:
                competitor_mean = results[eid][model]["mean"][r]
                competitor_variance = results[eid][model]["variance"][r]
                
                jrr_mean = results[eid]["jrr"]["mean"][r]
                jrr_variance = results[eid]["jrr"]["variance"][r]

                print(jrr_mean, competitor_variance)
                plt.errorbar(jrr_mean,
                             competitor_mean,
                             xerr=jrr_variance,
                             yerr=competitor_variance)

            ax = plt.gca()
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".8")
            plt.margins(0)
            plt.tight_layout(0, 0, 0)
            plt.show()
