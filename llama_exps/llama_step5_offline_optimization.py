import numpy as np
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

embedding_dir = Path.cwd().parent / "embeddings"
critics_dir = Path.cwd().parent / "critics"
active_results_dir = Path.cwd().parent / "active_results"

# NOTE: Select you task here
for TASK in ["gsm8k", "svamp"]:
    if TASK == "gsm8k":
        inserted = "gsm8k_"
    elif TASK == "svamp":
        inserted = "svamp_"
    elif TASK == "mawps":
        inserted = "mawps_"
    for add_n in range(10, 11):
        # load embeddings of the questions
        if TASK == "gsm8k":
            question_embedding_train = np.load(
                embedding_dir / "GSM8K_train_Q_embeddings.npy"
            )
            question_embedding_test = np.load(
                embedding_dir / "GSM8K_test_Q_embeddings.npy"
            )[:1000]
            with open("LMllama2/gsm8k_names.json", "r") as f:
                offline_names = json.load(f)
        elif TASK == "svamp":
            svamp_embeddings = np.load(embedding_dir / "aug_svamp_Q_embeddings.npy")
            question_embedding_train = svamp_embeddings[:15000]
            question_embedding_test = svamp_embeddings[15000:]
            with open("LMllama2/svamp_names.json", "r") as f:
                offline_names = json.load(f)
        elif TASK == "mawps":
            svamp_embeddings = np.load(embedding_dir / "svamp_Q_embeddings.npy")
            question_embedding_train = svamp_embeddings[:6000]
            question_embedding_test = svamp_embeddings[6000:]
            with open("LMllama2/mawps_names.json", "r") as f:
                offline_names = json.load(f)
        # load embeddings of the prompts

        out_dict = {}

        # load test performance of the prompts

        model_name_list = []
        for file_i in os.listdir("llama_models"):
            if TASK == "gsm8k":
                if file_i.startswith("gsm8k"):
                    model_name_list.append(file_i)
            elif TASK == "mawps":
                if file_i.startswith("mawps"):
                    model_name_list.append(file_i)
            elif TASK == "svamp":
                if file_i.startswith("svamp"):
                    model_name_list.append(file_i)

        for model in tqdm(model_name_list):
            for ADDITIONAL_N in range(add_n, add_n + 1):
                prompt_embedding = np.load(
                    embedding_dir / "gpt4_prompt_embeddings.npy"
                )[:ADDITIONAL_N]
                train_embedding = np.load(embedding_dir / "prompt_embeddings.npy")

                test_gold = []
                for file_i in range(ADDITIONAL_N):
                    with open(
                        f"llama2_chat_logs/{TASK}/llama_{TASK}_corr_gpt4_{file_i}.txt",
                        "r",
                    ) as f:
                        file_i_context = f.readlines()
                    file_i_context = ["True" in i for i in file_i_context]
                    test_gold.append(file_i_context)
                test_gold = np.asarray(test_gold)

                train_gold = []
                for file_i in offline_names["test_set"]:
                    train_gold.append(np.load(f"{file_i}"))
                train_gold = np.asarray(train_gold)

                # load xgboost model
                print(f"model: {model}")
                # load i, j, k, ... from model name

                ijk_numbers = len(model.split("_")) - 2
                if ijk_numbers == 1:
                    i = int(model.split("_")[2][0])
                elif ijk_numbers == 2:
                    i = int(model.split("_")[2][0])
                    j = int(model.split("_")[3][0])
                elif ijk_numbers == 3:
                    i = int(model.split("_")[2][0])
                    j = int(model.split("_")[3][0])
                    k = int(model.split("_")[4][0])
                elif ijk_numbers == 4:
                    i = int(model.split("_")[2][0])
                    j = int(model.split("_")[3][0])
                    k = int(model.split("_")[4][0])
                    l = int(model.split("_")[5][0])  # noqa: E741
                elif ijk_numbers == 5:
                    i = int(model.split("_")[2][0])
                    j = int(model.split("_")[3][0])
                    k = int(model.split("_")[4][0])
                    l = int(model.split("_")[5][0])  # noqa: E741
                    m = int(model.split("_")[6][0])
                elif ijk_numbers == 6:
                    i = int(model.split("_")[2][0])
                    j = int(model.split("_")[3][0])
                    k = int(model.split("_")[4][0])
                    l = int(model.split("_")[5][0])  # noqa: E741
                    m = int(model.split("_")[6][0])
                    n = int(model.split("_")[7][0])
                # load test performance of the prompts

                bst = xgb.Booster()  # init model
                bst.load_model(f"llama_models/{model}")  # load data

                # fuse all 10 prompts
                if ijk_numbers == 1:
                    train_data = train_gold[[i]]
                    train_embedding = train_embedding[[i]]
                elif ijk_numbers == 2:
                    train_data = train_gold[[i, j]]
                    train_embedding = train_embedding[[i, j]]
                elif ijk_numbers == 3:
                    train_data = train_gold[[i, j, k]]
                    train_embedding = train_embedding[[i, j, k]]
                elif ijk_numbers == 4:
                    train_data = train_gold[[i, j, k, l]]
                    train_embedding = train_embedding[[i, j, k, l]]
                elif ijk_numbers == 5:
                    train_data = train_gold[[i, j, k, l, m]]
                    train_embedding = train_embedding[[i, j, k, l, m]]
                elif ijk_numbers == 6:
                    train_data = train_gold[[i, j, k, l, m, n]]
                    train_embedding = train_embedding[[i, j, k, l, m, n]]
                test_data = np.concatenate((test_gold, train_data), axis=0)
                combined_test = []
                baseline_test = []
                only_combined = []
                smart_combined = []
                # tot_prompt_scores = []

                critic_confidence = []
                for prompt_i in range(6):
                    critic_confidence.append(
                        np.load(
                            critics_dir
                            / f"Processed_llama_{TASK}_prompt_{prompt_i}_confidence.npy"
                        )
                    )
                inandout_dist_gold = np.concatenate((train_data, test_gold), axis=0)
                critic_confidence = np.array(critic_confidence)
                if ijk_numbers == 1:
                    critic_confidence = critic_confidence[[i]]
                elif ijk_numbers == 2:
                    critic_confidence = critic_confidence[[i, j]]
                elif ijk_numbers == 3:
                    critic_confidence = critic_confidence[[i, j, k]]
                elif ijk_numbers == 4:
                    critic_confidence = critic_confidence[[i, j, k, l]]
                elif ijk_numbers == 5:
                    critic_confidence = critic_confidence[[i, j, k, l, m]]
                elif ijk_numbers == 6:
                    critic_confidence = critic_confidence[[i, j, k, l, m, n]]

                llm_combined = inandout_dist_gold[
                    np.argmax(critic_confidence, axis=0),
                    np.arange(len(question_embedding_test)),
                ]

                test_prompt_scores = []
                for all_dom_prompt_j in range(len(prompt_embedding)):
                    test_prompt_scores.append(
                        bst.predict(
                            xgb.DMatrix(
                                np.concatenate(
                                    (
                                        question_embedding_test,
                                        prompt_embedding[all_dom_prompt_j]
                                        .reshape(1, -1)
                                        .repeat(len(question_embedding_test), axis=0),
                                    ),
                                    axis=1,
                                )
                            )
                        )
                    )
                train_prompt_scores = []
                for all_dom_prompt_j in range(len(train_embedding)):
                    train_prompt_scores.append(
                        bst.predict(
                            xgb.DMatrix(
                                np.concatenate(
                                    (
                                        question_embedding_test,
                                        train_embedding[all_dom_prompt_j]
                                        .reshape(1, -1)
                                        .repeat(len(question_embedding_test), axis=0),
                                    ),
                                    axis=1,
                                )
                            )
                        )
                    )
                tot_prompt_scores = np.concatenate(
                    (test_prompt_scores, train_prompt_scores), axis=0
                )
                # only combined: only test prompts: as long as this is better than the avg score, the reward model is good
                only_combined = test_data[
                    np.argmax(test_prompt_scores, axis=0),
                    np.arange(len(question_embedding_test)),
                ]
                # combined test: combining the training prompts and test prompts
                combined_test = test_data[
                    np.argmax(
                        np.concatenate(
                            (test_prompt_scores, train_prompt_scores), axis=0
                        ),
                        axis=0,
                    ),
                    np.arange(len(question_embedding_test)),
                ]
                # baseline test: only training prompts on the test set
                baseline_test = test_data[
                    np.argmax(train_prompt_scores, axis=0) + ADDITIONAL_N,
                    np.arange(len(question_embedding_test)),
                ]
                # a new combination method: mostly in-domain, only when the prob < 0.5, use the max of the out-domain
                smart_combined = []

                smt_idx = np.max(train_prompt_scores, axis=0) < 0.5
                smart_combined = test_data[
                    np.argmax(train_prompt_scores, axis=0) + ADDITIONAL_N,
                    np.arange(len(question_embedding_test)),
                ]
                smart_combined[smt_idx] = test_data[
                    np.argmax(test_prompt_scores, axis=0)[smt_idx],
                    np.arange(len(question_embedding_test))[smt_idx],
                ]

                combined_test = np.array(combined_test)
                baseline_test = np.array(baseline_test)
                only_combined = np.array(only_combined)
                smart_combined = np.array(smart_combined)
                if ADDITIONAL_N == 10:
                    print(
                        "training prompt test q combined acc:",
                        (baseline_test.sum() / len(baseline_test)).round(3),
                    )
                    print(
                        "smart combined accuracies (in+out-domain test):",
                        (smart_combined.sum() / len(smart_combined)).round(3),
                    )
                    print(
                        "training prompt accuracies avg:",
                        train_data.mean(),
                        "individual performance: \n",
                        train_data.mean(1).round(3),
                    )
                    print(
                        "test prompt accuracies avg:",
                        test_data.mean(),
                        "individual performance: \n",
                        test_data.mean(1).round(3),
                    )

                print(
                    "N=",
                    ADDITIONAL_N,
                    "all combined accuracies (in+out-domain test):",
                    (combined_test.sum() / len(combined_test)).round(3),
                    "only test combined accuracies (only out-of-domain test):",
                    (only_combined.sum() / len(only_combined)).round(3),
                )

                print(
                    "accuracy on the held-out prompts:",
                    ((tot_prompt_scores > 0.5) == test_data)
                    .mean(1)[:ADDITIONAL_N]
                    .mean()
                    .round(3),
                )
                print(
                    "those accs:",
                    ((tot_prompt_scores > 0.5) == test_data)
                    .mean(1)[:ADDITIONAL_N]
                    .round(3),
                )
                print(
                    "accuracy on the held-out test set:",
                    ((tot_prompt_scores > 0.5) == test_data)
                    .mean(1)[ADDITIONAL_N:]
                    .mean()
                    .round(3),
                )
                print(
                    "those accs:",
                    ((tot_prompt_scores > 0.5) == test_data)
                    .mean(1)[ADDITIONAL_N:]
                    .round(3),
                )

                # calculate precision and recall, and F1
                # precision = TP / (TP + FP)
                # recall = TP / (TP + FN)
                # F1 = 2 * precision * recall / (precision + recall)
                # held-out prompts
                TP = ((tot_prompt_scores > 0.5) * test_data).sum(1)[:ADDITIONAL_N]
                FP = ((tot_prompt_scores > 0.5) * (1 - test_data)).sum(1)[:ADDITIONAL_N]
                FN = ((tot_prompt_scores < 0.5) * test_data).sum(1)[:ADDITIONAL_N]
                precision_test_prompt = TP / (TP + FP)
                recall_test_prompt = TP / (TP + FN)
                F1_test_prompt = (
                    2
                    * precision_test_prompt
                    * recall_test_prompt
                    / (precision_test_prompt + recall_test_prompt)
                )
                print("test prompt precision:", precision_test_prompt.mean().round(3))
                print("test prompt recall:", recall_test_prompt.mean().round(3))
                print("test prompt F1:", F1_test_prompt.mean().round(3))
                # held-out test set
                TP = ((tot_prompt_scores > 0.5) * test_data).sum(1)[ADDITIONAL_N:]
                FP = ((tot_prompt_scores > 0.5) * (1 - test_data)).sum(1)[ADDITIONAL_N:]
                FN = ((tot_prompt_scores < 0.5) * test_data).sum(1)[ADDITIONAL_N:]
                precision_test_query = TP / (TP + FP)
                recall_test_query = TP / (TP + FN)
                F1_test_query = (
                    2
                    * precision_test_query
                    * recall_test_query
                    / (precision_test_query + recall_test_query)
                )
                print("test query precision:", precision_test_query.mean().round(3))
                print("test query recall:", recall_test_query.mean().round(3))
                print("test query F1:", F1_test_query.mean().round(3))

                out_dict[f"{model}_{ADDITIONAL_N}"] = {
                    "acc": (tot_prompt_scores > 0.5) == test_data,
                    "in_dom_prompt_test_q": baseline_test,
                    "all_prompt_test_q": combined_test,
                    "out_dom_prompt_test_q": only_combined,
                    "smart_combined": smart_combined,
                    "llm_combined": llm_combined,
                    "test_gold": test_data.mean(1),
                    "train_gold": train_data.mean(1),
                    "test_prompt_metric": (
                        precision_test_prompt.mean().round(3),
                        recall_test_prompt.mean().round(3),
                        F1_test_prompt.mean().round(3),
                    ),
                    "test_query_metric": (
                        precision_test_query.mean().round(3),
                        recall_test_query.mean().round(3),
                        F1_test_query.mean().round(3),
                    ),
                }

        # load the results from critic
        test_data = np.concatenate((train_gold, test_gold), axis=0)
        critic_preds = []
        critic_confidence = []
        for prompt_i in range(16):
            critic_preds.append(
                np.load(
                    critics_dir / f"Processed_llama_{TASK}_prompt_{prompt_i}_critic.npy"
                )
            )
            critic_confidence.append(
                np.load(
                    critics_dir
                    / f"Processed_llama_{TASK}_prompt_{prompt_i}_confidence.npy"
                )
            )
        critic_preds = np.array(critic_preds)  # 16 x 1000
        # precisions on training prompts, test query (the first 6 prompts)
        print(1)
        TP = (critic_preds * test_data).sum(1)[:6]
        FP = (critic_preds * (1 - test_data)).sum(1)[:6]
        FN = ((1 - critic_preds) * test_data).sum(1)[:6]
        lm_accuracy_test_query = (critic_preds == test_data).mean(1)[:6]
        lm_precision_test_query = TP / (TP + FP)
        lm_recall_test_query = TP / (TP + FN)
        lm_F1_test_query = (
            2
            * lm_precision_test_query
            * lm_recall_test_query
            / (lm_precision_test_query + lm_recall_test_query)
        )
        print("lm test query accuracy:", lm_accuracy_test_query.mean().round(3))
        print("lm test query precision:", lm_precision_test_query.mean().round(3))
        print("lm test query recall:", lm_recall_test_query.mean().round(3))
        print("lm test query F1:", lm_F1_test_query.mean().round(3))
        # precisions on test prompts, test query (the last 10 prompts)
        TP = (critic_preds * test_data).sum(1)[6:]
        FP = (critic_preds * (1 - test_data)).sum(1)[6:]
        FN = ((1 - critic_preds) * test_data).sum(1)[6:]
        lm_accuracy_test_prompt = (critic_preds == test_data).mean(1)[6:]
        lm_precision_test_prompt = TP / (TP + FP)
        lm_recall_test_prompt = TP / (TP + FN)
        lm_F1_test_prompt = (
            2
            * lm_precision_test_prompt
            * lm_recall_test_prompt
            / (lm_precision_test_prompt + lm_recall_test_prompt)
        )
        print("lm test prompt accuracy:", lm_accuracy_test_prompt.mean().round(3))
        print("lm test prompt precision:", lm_precision_test_prompt.mean().round(3))
        print("lm test prompt recall:", lm_recall_test_prompt.mean().round(3))
        print("lm test prompt F1:", lm_F1_test_prompt.mean().round(3))

        out_dict["lm_preds"] = {
            "test_query_metric": (
                lm_precision_test_query.mean().round(3),
                lm_recall_test_query.mean().round(3),
                lm_F1_test_query.mean().round(3),
                lm_accuracy_test_query.mean().round(3),
            ),
            "test_prompt_metric": (
                lm_precision_test_prompt.mean().round(3),
                lm_recall_test_prompt.mean().round(3),
                lm_F1_test_prompt.mean().round(3),
                lm_accuracy_test_prompt.mean().round(3),
            ),
        }

        critic_confidence = np.array(critic_confidence)
        all_gold = np.concatenate((train_gold, test_gold), axis=0)
        select_by_confidence_succ_rate = all_gold[
            np.argmax(critic_confidence, axis=0),
            np.arange(len(question_embedding_test)),
        ]
        select_by_confidence_succ_rate_train = all_gold[
            np.argmax(critic_confidence[:6], axis=0),
            np.arange(len(question_embedding_test)),
        ]
        print(
            "select by confidence succ rate:",
            select_by_confidence_succ_rate.mean().round(3),
        )
        out_dict["select_by_confidence"] = {
            "all_prompt_test_q": select_by_confidence_succ_rate,
            "in_dom_prompt_test_q": select_by_confidence_succ_rate_train,
        }
        Path("temp_folder").mkdir(parents=True, exist_ok=True)
        np.save(f"temp_folder/{TASK}_checkpointed_{ADDITIONAL_N}_llama2.npy", out_dict)

        if TASK == "gsm8k":
            dataset_name = "GSM8K"
        elif TASK == "mawps":
            dataset_name = "MAWPS"
        elif TASK == "svamp":
            dataset_name = "SVAMP"

        irl = np.load(
            active_results_dir / f"ALL_{TASK}_{ADDITIONAL_N}_llama.npy",
            allow_pickle=True,
        )
        print(out_dict.keys())

        train_best_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        train_mean_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        test_mean_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        all_mean_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        train_prompt_succ_rate_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        held_out_prompt_succ_rate_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        all_prompt_succ_rate_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        ours_combined_succ_rate_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        irl_100_succ_rate_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        llm_combined_succ_rate_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        test_acc_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        train_acc_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        # precision, recall, F1
        test_prompt_precision_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        test_prompt_recall_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        test_prompt_F1_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        test_query_precision_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        test_query_recall_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }
        test_query_F1_list = {
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": [],
            "10": [],
        }

        for key in out_dict.keys():
            if key == "lm_preds" or key == "select_by_confidence":
                continue
            orig_key = key
            print(" key now", key)
            temp_results = out_dict[key]
            n_train = len(temp_results["train_gold"])

            print("\n")
            print("train_best", temp_results["train_gold"].max())
            print("train_mean", temp_results["train_gold"].mean())
            print("test mean", temp_results["test_gold"][:-n_train].mean())
            print("all mean", temp_results["test_gold"].mean())
            print("train prompt succ rate", temp_results["in_dom_prompt_test_q"].mean())
            print(
                "held-out prompt succ rate",
                temp_results["out_dom_prompt_test_q"].mean(),
            )
            print("all prompt succ rate", temp_results["all_prompt_test_q"].mean())
            print("Ours combined succ rate", temp_results["smart_combined"].mean())
            print("IRL 100: ", irl.item()[key]["active_combined"].mean())
            print("test acc", temp_results["acc"].mean(1)[:-n_train].mean())
            print("train acc", temp_results["acc"].mean(1)[-n_train:].mean())
            if "gsm8k" in key:
                key = str(len(key.split("_")) - 2)
            elif "svamp" in key:
                key = str(len(key.split("_")) - 2)
            elif "mawps" in key:
                key = str(len(key.split("_")) - 2)

            train_best_list[key].append(temp_results["train_gold"].max())
            train_mean_list[key].append(temp_results["train_gold"].mean())
            test_mean_list[key].append(temp_results["test_gold"][:-n_train].mean())
            all_mean_list[key].append(temp_results["test_gold"].mean())
            train_prompt_succ_rate_list[key].append(
                temp_results["in_dom_prompt_test_q"].mean()
            )
            held_out_prompt_succ_rate_list[key].append(
                temp_results["out_dom_prompt_test_q"].mean()
            )
            all_prompt_succ_rate_list[key].append(
                temp_results["all_prompt_test_q"].mean()
            )
            ours_combined_succ_rate_list[key].append(
                temp_results["smart_combined"].mean()
            )
            irl_100_succ_rate_list[key].append(
                irl.item()[orig_key]["active_combined"].mean()
            )
            llm_combined_succ_rate_list[key].append(temp_results["llm_combined"].mean())
            test_acc_list[key].append(temp_results["acc"].mean(1)[:-n_train].mean())
            train_acc_list[key].append(temp_results["acc"].mean(1)[-n_train:].mean())
            # only append not nan
            if np.isnan(temp_results["test_prompt_metric"][0]).sum() == 0:
                test_prompt_precision_list[key].append(
                    temp_results["test_prompt_metric"][0]
                )
            test_prompt_recall_list[key].append(temp_results["test_prompt_metric"][1])
            if np.isnan(temp_results["test_prompt_metric"][2]).sum() == 0:
                test_prompt_F1_list[key].append(temp_results["test_prompt_metric"][2])
            if np.isnan(temp_results["test_query_metric"][0]).sum() == 0:
                test_query_precision_list[key].append(
                    temp_results["test_query_metric"][0]
                )
            test_query_recall_list[key].append(temp_results["test_query_metric"][1])
            if np.isnan(temp_results["test_query_metric"][2]).sum() == 0:
                test_query_F1_list[key].append(temp_results["test_query_metric"][2])

        # compare training best (this corresponds to using the training time best prompt)
        # with the in-domain combination (train_propmpt_succ_rate)
        # and ours_combined_succ_rate

        plt.figure(figsize=(6, 3))
        plt.plot(
            range(1, 7),
            [np.mean(train_best_list[str(i)]) for i in range(2, 8)],
            label="Training Best",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(train_best_list[str(i)]) - np.std(train_best_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(train_best_list[str(i)]) + np.std(train_best_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        # plt.plot(range(1,7), [np.mean(held_out_prompt_succ_rate_list[str(i)]) for i in range(2,8)], label='Out-of-domain Combined')
        # plt.fill_between(range(1,7), [np.mean(held_out_prompt_succ_rate_list[str(i)])-np.std(held_out_prompt_succ_rate_list[str(i)]) for i in range(2,8)], [np.mean(held_out_prompt_succ_rate_list[str(i)])+np.std(held_out_prompt_succ_rate_list[str(i)]) for i in range(2,8)], alpha=0.2)
        plt.plot(
            range(1, 7),
            [np.mean(all_prompt_succ_rate_list[str(i)]) for i in range(2, 8)],
            label="Naive Mixture",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(all_prompt_succ_rate_list[str(i)])
                - np.std(all_prompt_succ_rate_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(all_prompt_succ_rate_list[str(i)])
                + np.std(all_prompt_succ_rate_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        plt.plot(
            range(1, 7),
            [np.mean(train_prompt_succ_rate_list[str(i)]) for i in range(2, 8)],
            label="IRL Test 0",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(train_prompt_succ_rate_list[str(i)])
                - np.std(train_prompt_succ_rate_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(train_prompt_succ_rate_list[str(i)])
                + np.std(train_prompt_succ_rate_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        plt.plot(
            range(1, 7),
            [np.mean(ours_combined_succ_rate_list[str(i)]) for i in range(2, 8)],
            label="IRL Test 10",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(ours_combined_succ_rate_list[str(i)])
                - np.std(ours_combined_succ_rate_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(ours_combined_succ_rate_list[str(i)])
                + np.std(ours_combined_succ_rate_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        plt.plot(
            range(1, 7),
            [np.mean(irl_100_succ_rate_list[str(i)]) for i in range(2, 8)],
            label="IRL Test 100",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(irl_100_succ_rate_list[str(i)])
                - np.std(irl_100_succ_rate_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(irl_100_succ_rate_list[str(i)])
                + np.std(irl_100_succ_rate_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        # baseline of llms # select_by_confidence_succ_rate
        plt.hlines(
            out_dict["select_by_confidence"]["all_prompt_test_q"].mean(),
            1,
            6,
            linestyles="dashed",
            label="LLM Conf. Test 10",
            color="gray",
        )
        plt.hlines(
            out_dict["select_by_confidence"]["in_dom_prompt_test_q"].mean(),
            1,
            6,
            linestyles="dashed",
            label="LLM Conf. Train ",
            color="orange",
        )
        plt.legend()
        plt.xlabel("Number of Training Prompts")
        plt.ylabel("Performance")
        plt.title(f"{dataset_name}, N Test Prompt = {ADDITIONAL_N} ")
        Path("figs").mkdir(parents=True, exist_ok=True)
        plt.savefig(
            f"figs/{dataset_name}_llama2_performance_comparison_{ADDITIONAL_N}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        import csv

        # Step 1: Collect your data

        x_data = list(range(1, 7))

        # For Means
        training_best_data = [np.mean(train_best_list[str(i)]) for i in range(2, 8)]
        irl_0_data = [np.mean(train_prompt_succ_rate_list[str(i)]) for i in range(2, 8)]
        all_mixed_data = [
            np.mean(all_prompt_succ_rate_list[str(i)]) for i in range(2, 8)
        ]
        irl_10_data = [
            np.mean(ours_combined_succ_rate_list[str(i)]) for i in range(2, 8)
        ]
        irl_100_data = [np.mean(irl_100_succ_rate_list[str(i)]) for i in range(2, 8)]
        llm_combined_data = [
            np.mean(llm_combined_succ_rate_list[str(i)]) for i in range(2, 8)
        ]
        llm_conf_test_data = [
            out_dict["select_by_confidence"]["all_prompt_test_q"].mean()
        ] * len(x_data)
        llm_conf_train_data = [
            out_dict["select_by_confidence"]["in_dom_prompt_test_q"].mean()
        ] * len(x_data)

        # For Standard Deviations
        training_best_std = [np.std(train_best_list[str(i)]) for i in range(2, 8)]
        irl_0_std = [np.std(train_prompt_succ_rate_list[str(i)]) for i in range(2, 8)]
        all_mixed_std = [np.std(all_prompt_succ_rate_list[str(i)]) for i in range(2, 8)]
        irl_10_std = [np.std(ours_combined_succ_rate_list[str(i)]) for i in range(2, 8)]
        irl_100_std = [np.std(irl_100_succ_rate_list[str(i)]) for i in range(2, 8)]

        # Step 2: Write data to CSV
        Path("csvs").mkdir(parents=True, exist_ok=True)
        with open(
            f"csvs/{dataset_name}_llama2_performance_comparison.csv", "w", newline=""
        ) as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write header for means and standard deviations
            csvwriter.writerow(
                [
                    "X",
                    "Training Best Mean",
                    "Training Best Std",
                    "IRL 0 Mean",
                    "IRL 0 Std",
                    "All Mixed Mean",
                    "All Mixed Std",
                    "IRL 10 Mean",
                    "IRL 10 Std",
                    "IRL 100 Mean",
                    "IRL 100 Std",
                    "LLM Confidence w/ Test Prompts",
                    "LLM Confidence Train Prompts",
                    "LLM combined",
                ]
            )

            # Write data for means and standard deviations
            for i in range(len(x_data)):
                csvwriter.writerow(
                    [
                        x_data[i],
                        training_best_data[i],
                        training_best_std[i],
                        irl_0_data[i],
                        irl_0_std[i],
                        all_mixed_data[i],
                        all_mixed_std[i],
                        irl_10_data[i],
                        irl_10_std[i],
                        irl_100_data[i],
                        irl_100_std[i],
                        llm_conf_test_data[i],
                        llm_conf_train_data[i],
                        llm_combined_data[i],
                    ]
                )

        # plot acc, precision, recall, F1
        plt.figure(figsize=(6, 3))
        plt.plot(
            range(1, 7),
            [np.mean(test_acc_list[str(i)]) for i in range(2, 8)],
            label="Reward Model, Test Prompts",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(test_acc_list[str(i)]) - np.std(test_acc_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(test_acc_list[str(i)]) + np.std(test_acc_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        plt.plot(
            range(1, 7),
            [np.mean(train_acc_list[str(i)]) for i in range(2, 8)],
            label="Reward Model, Train Prompts",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(train_acc_list[str(i)]) - np.std(train_acc_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(train_acc_list[str(i)]) + np.std(train_acc_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        # acc baseline of llms # lm_preds
        plt.hlines(
            out_dict["lm_preds"]["test_prompt_metric"][3],
            1,
            6,
            linestyles="dashed",
            label="LLM, Test Prompts",
        )
        plt.hlines(
            out_dict["lm_preds"]["test_query_metric"][3],
            1,
            6,
            linestyles="dashed",
            label="LLM, Train Prompts",
            color="orange",
        )
        plt.legend()
        plt.xlabel("Number of Training Prompts")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset_name}, N Test Prompt = {ADDITIONAL_N} ")
        plt.savefig(
            f"figs/{dataset_name}_llama2_acc_comparison_{ADDITIONAL_N}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        plt.figure(figsize=(6, 3))
        plt.plot(
            range(1, 7),
            [np.mean(test_prompt_precision_list[str(i)]) for i in range(2, 8)],
            label="Reward Model, Test Prompts",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(test_prompt_precision_list[str(i)])
                - np.std(test_prompt_precision_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(test_prompt_precision_list[str(i)])
                + np.std(test_prompt_precision_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        plt.plot(
            range(1, 7),
            [np.mean(test_query_precision_list[str(i)]) for i in range(2, 8)],
            label="Reward Model, Train Prompts",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(test_query_precision_list[str(i)])
                - np.std(test_query_precision_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(test_query_precision_list[str(i)])
                + np.std(test_query_precision_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        # precision baseline of llms # lm_preds
        plt.hlines(
            out_dict["lm_preds"]["test_prompt_metric"][0],
            1,
            6,
            linestyles="dashed",
            label="LLM, Test Prompts",
        )
        plt.hlines(
            out_dict["lm_preds"]["test_query_metric"][0],
            1,
            6,
            linestyles="dashed",
            label="LLM, Train Prompts",
            color="orange",
        )
        plt.legend()
        plt.xlabel("Number of Training Prompts")
        plt.ylabel("Precision")
        plt.title(f"{dataset_name}, N Test Prompt = {ADDITIONAL_N} ")
        plt.savefig(
            f"figs/{dataset_name}_llama2_precision_comparison_{ADDITIONAL_N}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        plt.figure(figsize=(6, 3))
        plt.plot(
            range(1, 7),
            [np.mean(test_prompt_recall_list[str(i)]) for i in range(2, 8)],
            label="Reward Model, Test Prompts",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(test_prompt_recall_list[str(i)])
                - np.std(test_prompt_recall_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(test_prompt_recall_list[str(i)])
                + np.std(test_prompt_recall_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        plt.plot(
            range(1, 7),
            [np.mean(test_query_recall_list[str(i)]) for i in range(2, 8)],
            label="Reward Model, Train Prompts",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(test_query_recall_list[str(i)])
                - np.std(test_query_recall_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(test_query_recall_list[str(i)])
                + np.std(test_query_recall_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        # recall baseline of llms # lm_preds
        plt.hlines(
            out_dict["lm_preds"]["test_prompt_metric"][1],
            1,
            6,
            linestyles="dashed",
            label="LLM, Test Prompts",
        )
        plt.hlines(
            out_dict["lm_preds"]["test_query_metric"][1],
            1,
            6,
            linestyles="dashed",
            label="LLM, Train Prompts",
            color="orange",
        )
        plt.legend()
        plt.xlabel("Number of Training Prompts")
        plt.ylabel("Recall")
        plt.title(f"{dataset_name}, N Test Prompt = {ADDITIONAL_N} ")
        plt.savefig(
            f"figs/{dataset_name}_llama2_recall_comparison_{ADDITIONAL_N}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        plt.figure(figsize=(6, 3))
        plt.plot(
            range(1, 7),
            [np.mean(test_prompt_F1_list[str(i)]) for i in range(2, 8)],
            label="Reward Model, Test Prompts",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(test_prompt_F1_list[str(i)])
                - np.std(test_prompt_F1_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(test_prompt_F1_list[str(i)])
                + np.std(test_prompt_F1_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        plt.plot(
            range(1, 7),
            [np.mean(test_query_F1_list[str(i)]) for i in range(2, 8)],
            label="Reward Model, Train Prompts",
        )
        plt.fill_between(
            range(1, 7),
            [
                np.mean(test_query_F1_list[str(i)]) - np.std(test_query_F1_list[str(i)])
                for i in range(2, 8)
            ],
            [
                np.mean(test_query_F1_list[str(i)]) + np.std(test_query_F1_list[str(i)])
                for i in range(2, 8)
            ],
            alpha=0.2,
        )
        # F1 baseline of llms # lm_preds
        plt.hlines(
            out_dict["lm_preds"]["test_prompt_metric"][2],
            1,
            6,
            linestyles="dashed",
            label="LLM, Test Prompts",
        )
        plt.hlines(
            out_dict["lm_preds"]["test_query_metric"][2],
            1,
            6,
            linestyles="dashed",
            label="LLM, Train Prompts",
            color="orange",
        )
        plt.legend()
        plt.xlabel("Number of Training Prompts")
        plt.ylabel("F1")
        plt.title(f"{dataset_name}, N Test Prompt = {ADDITIONAL_N} ")
        plt.savefig(
            f"figs/{dataset_name}_llama2_F1_comparison_{ADDITIONAL_N}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        import csv

        # Function to save metrics to CSV
        def save_to_csv(
            filename, metric_name, test_list, train_list, test_metric, train_metric
        ):
            x_data = list(range(1, 7))

            # Collecting Mean Data
            test_mean_data = [np.mean(test_list[str(i)]) for i in range(2, 8)]
            train_mean_data = [np.mean(train_list[str(i)]) for i in range(2, 8)]

            # Collecting Standard Deviation Data
            test_std_data = [np.std(test_list[str(i)]) for i in range(2, 8)]
            train_std_data = [np.std(train_list[str(i)]) for i in range(2, 8)]

            # Writing to CSV
            with open(filename, "w", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)

                # Header
                csvwriter.writerow(
                    [
                        "X",
                        f"Test {metric_name} Mean",
                        f"Test {metric_name} Std",
                        f"Train {metric_name} Mean",
                        f"Train {metric_name} Std",
                        "LLM Test Metric",
                        "LLM Train Metric",
                    ]
                )

                # Data
                for i in range(len(x_data)):
                    csvwriter.writerow(
                        [
                            x_data[i],
                            test_mean_data[i],
                            test_std_data[i],
                            train_mean_data[i],
                            train_std_data[i],
                            test_metric[i],
                            train_metric[i],
                        ]
                    )

        # Saving Accuracy
        save_to_csv(
            f"csvs/{dataset_name}_llama2_acc_comparison.csv",
            "Accuracy",
            test_acc_list,
            train_acc_list,
            [out_dict["lm_preds"]["test_prompt_metric"][3]] * 6,
            [out_dict["lm_preds"]["test_query_metric"][3]] * 6,
        )

        # Saving Precision
        save_to_csv(
            f"csvs/{dataset_name}_llama2_precision_comparison.csv",
            "Precision",
            test_prompt_precision_list,
            test_query_precision_list,
            [out_dict["lm_preds"]["test_prompt_metric"][0]] * 6,
            [out_dict["lm_preds"]["test_query_metric"][0]] * 6,
        )

        # Saving Recall
        save_to_csv(
            f"csvs/{dataset_name}_llama2_recall_comparison.csv",
            "Recall",
            test_prompt_recall_list,
            test_query_recall_list,
            [out_dict["lm_preds"]["test_prompt_metric"][1]] * 6,
            [out_dict["lm_preds"]["test_query_metric"][1]] * 6,
        )

        # Saving F1
        save_to_csv(
            f"csvs/{dataset_name}_llama2_F1_comparison.csv",
            "F1",
            test_prompt_F1_list,
            test_query_F1_list,
            [out_dict["lm_preds"]["test_prompt_metric"][2]] * 6,
            [out_dict["lm_preds"]["test_query_metric"][2]] * 6,
        )
