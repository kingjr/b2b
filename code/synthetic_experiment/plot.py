import argparse
import numpy as np
from ast import literal_eval
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

stats = {
    "result_error_in_all": "error in",
    "result_error_out_all": "error out",
    "result_error_in_mask": "error in (E)",
    "result_error_out_mask": "error out (E)",
    "result_false_positives": "false positives",
    "result_false_negatives": "false negatives",
    # "result_response_active": "response at active",
    # "result_response_inactive": "response at inactive",
    "result_auc": "AUC"
}

models = {
    "Ridge" : "C1",
    "Lasso" : "C2",
    "PLS" : "C3",
    "CCA" : "C4",
    "RRR" : "C5"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JRR synthetic experiment')
    parser.add_argument('--file', type=str, default='synthetic_result.txt')
    parser.add_argument('--nonlinear', type=int, default=1)
    args = parser.parse_args()

    with open(args.file, "r") as f:
        lines = f.readlines()

    results = {}

    for line in lines:
        line_dict = literal_eval(line)

        # if int(line_dict["nonlinear"]) != args.nonlinear:
        #     continue

        eid = "{}_{}_{}_{}_{}_{}_{}".format(line_dict["n_samples"],
                                            line_dict["dim_x"],
                                            line_dict["dim_y"],
                                            line_dict["snr"],
                                            line_dict["nc"],
                                            line_dict["nonlinear"],
                                            line_dict["n_seeds"])
        model = line_dict["model"]

        if eid not in results:
            results[eid] = {}

        if model not in results[eid]: 
            results[eid][model] = {}

        for stat in stats:
            if stat not in results[eid][model]:
                results[eid][model][stat] = []
            results[eid][model][stat].append(line_dict[stat])

    
    plt.figure(figsize=(8, 5))
    counter = 0
    for m, model in enumerate(models):
        for s, stat in enumerate(stats):
            counter += 1
            ax = plt.subplot(len(models), len(stats), counter)
            if m == 0:
                plt.title(stats[stat], fontsize=10)
            if s == 0:
                plt.ylabel(model)
            data_x = []
            data_y = []
            data_c = []
            if model != "JRR":
                for eid in results:
                    if model in results[eid] and "JRR" in results[eid]:
                        data_x.append(np.mean(results[eid]["JRR"][stat]))
                        data_y.append(np.mean(results[eid][model][stat]))

                        is_nonlinear = int(eid.split("_")[5])

                        if is_nonlinear == 0:
                            data_c.append("C0")
                        elif is_nonlinear == 1:
                            data_c.append("C1")

            plt.scatter(data_x, data_y, c=data_c, alpha=0.25, s=7)#, rasterized=True)
            xl = ax.get_xlim()
            yl = ax.get_ylim()

            min_ = min(min(xl), min(yl))
            max_ = max(max(xl), max(yl))

            ax.plot([min_, max_], [min_, max_], ls="--", c=".8", zorder=0)

            ax.set_xlim(min_, max_)
            ax.set_ylim(min_, max_)
            
            if m != (len(models) - 1):
                plt.xticks([])
                plt.yticks([])
            else:
                plt.yticks([])
            ax.margins(0)
    plt.tight_layout(0, 0, 0)
    plt.savefig(args.file + ".pdf")
    # plt.show()
