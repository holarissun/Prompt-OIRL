import matplotlib.pyplot as plt
import numpy as np
import json

# NOTE: Change the TASK to the task you want to process
for TASK in ["svamp", "gsm8k"]:
    json_file = f"LMllama2/{TASK}_names.json"

    # load the dictionary
    with open(json_file, "r") as f:
        offline_names = json.load(f)

    # load the training results
    train_result_list = []
    for file in offline_names["training_set"]:
        train_result_list.append(np.load(file))

    train_result_list = np.asarray(train_result_list)

    # load the test results
    test_result_list = []
    for file in offline_names["test_set"]:
        test_result_list.append(np.load(file))

    test_result_list = np.asarray(test_result_list)

    test_data = []
    test_labels = []
    for q_i in range(len(test_result_list[0])):
        # a classification task on whether the prompt can lead to a correct answer
        for idx_i in range(len(test_result_list)):
            test_data.append((q_i, idx_i))
            if test_result_list[idx_i, q_i]:
                test_labels.append(1)
            else:
                test_labels.append(0)

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    print("test data shape", test_data.shape)

    # save data, labels, index, and corr_sum
    np.save("LMllama2/{}_test_query_data.npy".format(TASK), test_data)
    np.save("LMllama2/{}_test_query_labels.npy".format(TASK), test_labels)

    # load the prompt embeddings
    prompt_embedding = np.load("../embeddings/prompt_embeddings.npy")
    assert len(prompt_embedding) == len(offline_names["prompts"])

    # calculate the correlation between the results
    corr = np.corrcoef(train_result_list)
    corr = np.abs(corr)
    corr = corr - np.eye(corr.shape[0])
    corr = np.triu(corr)

    plt.figure(figsize=(3, 3))
    plt.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()

    # for each potential combination of three methods, calculate the sum of the correlation (on the training set)
    # then construct a dataset for each combination of three methods

    # create dataset with a single method
    for i in range(corr.shape[0]):
        method_index = [i]
        corr_sum = [corr[i, i]]

        training_data = []
        training_labels = []
        for q_i in range(len(train_result_list[0])):
            # a classification task on whether the prompt can lead to a correct answer
            training_data.append((q_i, i))
            if train_result_list[i, q_i]:
                training_labels.append(1)
            else:
                training_labels.append(0)

        training_data = np.array(training_data)
        training_labels = np.array(training_labels)
        print("index", i, "corr_sum", corr_sum[-1])
        print("training data shape", training_data.shape)
        # save data, labels, index, and corr_sum
        np.save("LMllama2/{}_training_data_{}.npy".format(TASK, i), training_data)
        np.save("LMllama2/{}_training_labels_{}.npy".format(TASK, i), training_labels)
        np.save("LMllama2/{}_method_index_{}.npy".format(TASK, i), method_index)
        np.save("LMllama2/{}_corr_sum_{}.npy".format(TASK, i), corr_sum)

    # create dataset for combining 6 methods
    method_index = []
    corr_sum = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            for k in range(j + 1, corr.shape[0]):
                for l in range(k + 1, corr.shape[0]):  # noqa: E741
                    for m in range(l + 1, corr.shape[0]):
                        for n in range(m + 1, corr.shape[0]):
                            method_index.append([i, j, k, l, m, n])
                            corr_sum.append(
                                corr[i, j]
                                + corr[i, k]
                                + corr[i, l]
                                + corr[i, m]
                                + corr[i, n]
                                + corr[j, k]
                                + corr[j, l]
                                + corr[j, m]
                                + corr[j, n]
                                + corr[k, l]
                                + corr[k, m]
                                + corr[k, n]
                                + corr[l, m]
                                + corr[l, n]
                                + corr[m, n]
                            )

                            training_data = []
                            training_labels = []
                            for q_i in range(len(train_result_list[0])):
                                # a classification task on whether the prompt can lead to a correct answer
                                for idx_i in [i, j, k, l, m, n]:
                                    training_data.append((q_i, idx_i))
                                    if train_result_list[idx_i, q_i]:  # ==True
                                        training_labels.append(1)
                                    else:
                                        training_labels.append(0)

                            training_data = np.array(training_data)
                            training_labels = np.array(training_labels)
                            print("index", i, j, k, l, m, n, "corr_sum", corr_sum[-1])
                            print("training data shape", training_data.shape)
                            # save data, labels, index, and corr_sum
                            np.save(
                                "LMllama2/{}_training_data_{}_{}_{}_{}_{}_{}.npy".format(
                                    TASK, i, j, k, l, m, n
                                ),
                                training_data,
                            )
                            np.save(
                                "LMllama2/{}_training_labels_{}_{}_{}_{}_{}_{}.npy".format(
                                    TASK, i, j, k, l, m, n
                                ),
                                training_labels,
                            )
                            np.save(
                                "LMllama2/{}_method_index_{}_{}_{}_{}_{}_{}.npy".format(
                                    TASK, i, j, k, l, m, n
                                ),
                                method_index[-1],
                            )
                            np.save(
                                "LMllama2/{}_corr_sum_{}_{}_{}_{}_{}_{}.npy".format(
                                    TASK, i, j, k, l, m, n
                                ),
                                corr_sum[-1],
                            )

    # create dataset for combining 3 methods
    method_index = []
    corr_sum = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            for k in range(j + 1, corr.shape[0]):
                method_index.append([i, j, k])
                corr_sum.append(corr[i, j] + corr[i, k] + corr[j, k])

                training_data = []
                training_labels = []
                for q_i in range(len(train_result_list[0])):
                    # a classification task on whether the prompt can lead to a correct answer
                    for idx_i in [i, j, k]:
                        training_data.append((q_i, idx_i))
                        if train_result_list[idx_i, q_i]:  # ==True
                            training_labels.append(1)
                        else:
                            training_labels.append(0)

                training_data = np.array(training_data)
                training_labels = np.array(training_labels)
                print("index", i, j, k, "corr_sum", corr_sum[-1])
                print("training data shape", training_data.shape)
                # save data, labels, index, and corr_sum
                np.save(
                    "LMllama2/{}_training_data_{}_{}_{}.npy".format(TASK, i, j, k),
                    training_data,
                )
                np.save(
                    "LMllama2/{}_training_labels_{}_{}_{}.npy".format(TASK, i, j, k),
                    training_labels,
                )
                np.save(
                    "LMllama2/{}_method_index_{}_{}_{}.npy".format(TASK, i, j, k),
                    method_index[-1],
                )
                np.save(
                    "LMllama2/{}_corr_sum_{}_{}_{}.npy".format(TASK, i, j, k),
                    corr_sum[-1],
                )

    # create dataset for combining 2 methods
    method_index = []
    corr_sum = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            method_index.append([i, j])
            corr_sum.append(corr[i, j])

            training_data = []
            training_labels = []
            for q_i in range(len(train_result_list[0])):
                # a classification task on whether the prompt can lead to a correct answer
                for idx_i in [i, j]:
                    training_data.append((q_i, idx_i))
                    if train_result_list[idx_i, q_i]:  # ==True
                        training_labels.append(1)
                    else:
                        training_labels.append(0)

            training_data = np.array(training_data)
            training_labels = np.array(training_labels)
            print("index", i, j, "corr_sum", corr_sum[-1])
            print("training data shape", training_data.shape)
            # save data, labels, index, and corr_sum
            np.save(
                "LMllama2/{}_training_data_{}_{}.npy".format(TASK, i, j), training_data
            )
            np.save(
                "LMllama2/{}_training_labels_{}_{}.npy".format(TASK, i, j),
                training_labels,
            )
            np.save(
                "LMllama2/{}_method_index_{}_{}.npy".format(TASK, i, j),
                method_index[-1],
            )
            np.save("LMllama2/{}_corr_sum_{}_{}.npy".format(TASK, i, j), corr_sum[-1])

    # create dataset for combining 4 method
    method_index = []
    corr_sum = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            for k in range(j + 1, corr.shape[0]):
                for l in range(k + 1, corr.shape[0]):  # noqa: E741
                    method_index.append([i, j, k, l])
                    corr_sum.append(
                        corr[i, j]
                        + corr[i, k]
                        + corr[i, l]
                        + corr[j, k]
                        + corr[j, l]
                        + corr[k, l]
                    )

                    training_data = []
                    training_labels = []
                    for q_i in range(len(train_result_list[0])):
                        # a classification task on whether the prompt can lead to a correct answer
                        for idx_i in [i, j, k, l]:
                            training_data.append((q_i, idx_i))
                            if train_result_list[idx_i, q_i]:  # ==True
                                training_labels.append(1)
                            else:
                                training_labels.append(0)

                    training_data = np.array(training_data)
                    training_labels = np.array(training_labels)
                    print("index", i, j, k, l, "corr_sum", corr_sum[-1])
                    print("training data shape", training_data.shape)
                    # save data, labels, index, and corr_sum
                    np.save(
                        "LMllama2/{}_training_data_{}_{}_{}_{}.npy".format(
                            TASK, i, j, k, l
                        ),
                        training_data,
                    )
                    np.save(
                        "LMllama2/{}_training_labels_{}_{}_{}_{}.npy".format(
                            TASK, i, j, k, l
                        ),
                        training_labels,
                    )
                    np.save(
                        "LMllama2/{}_method_index_{}_{}_{}_{}.npy".format(
                            TASK, i, j, k, l
                        ),
                        method_index[-1],
                    )
                    np.save(
                        "LMllama2/{}_corr_sum_{}_{}_{}_{}.npy".format(TASK, i, j, k, l),
                        corr_sum[-1],
                    )

    # create dataset for combining 5 method
    method_index = []
    corr_sum = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            for k in range(j + 1, corr.shape[0]):
                for l in range(k + 1, corr.shape[0]):  # noqa: E741
                    for m in range(l + 1, corr.shape[0]):
                        method_index.append([i, j, k, l, m])
                        corr_sum.append(
                            corr[i, j]
                            + corr[i, k]
                            + corr[i, l]
                            + corr[i, m]
                            + corr[j, k]
                            + corr[j, l]
                            + corr[j, m]
                            + corr[k, l]
                            + corr[k, m]
                            + corr[l, m]
                        )

                        training_data = []
                        training_labels = []
                        for q_i in range(len(train_result_list[0])):
                            # a classification task on whether the prompt can lead to a correct answer
                            for idx_i in [i, j, k, l, m]:
                                training_data.append((q_i, idx_i))
                                if train_result_list[idx_i, q_i]:  # == True
                                    training_labels.append(1)
                                else:
                                    training_labels.append(0)

                        training_data = np.array(training_data)
                        training_labels = np.array(training_labels)
                        print("index", i, j, k, l, m, "corr_sum", corr_sum[-1])
                        print("training data shape", training_data.shape)
                        # save data, labels, index, and corr_sum
                        np.save(
                            "LMllama2/{}_training_data_{}_{}_{}_{}_{}.npy".format(
                                TASK, i, j, k, l, m
                            ),
                            training_data,
                        )
                        np.save(
                            "LMllama2/{}_training_labels_{}_{}_{}_{}_{}.npy".format(
                                TASK, i, j, k, l, m
                            ),
                            training_labels,
                        )
                        np.save(
                            "LMllama2/{}_method_index_{}_{}_{}_{}_{}.npy".format(
                                TASK, i, j, k, l, m
                            ),
                            method_index[-1],
                        )
                        np.save(
                            "LMllama2/{}_corr_sum_{}_{}_{}_{}_{}.npy".format(
                                TASK, i, j, k, l, m
                            ),
                            corr_sum[-1],
                        )
