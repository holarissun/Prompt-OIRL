import numpy as np
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score
import json
from pathlib import Path

# NOTE: Select your task here
TASK = "gsm8k"  # 'svamp', 'mawps'
if TASK == "gsm8k":
    inserted = "gsm8k_"
elif TASK == "mawps":
    inserted = "mawps_"
elif TASK == "svamp":
    inserted = "svamp_"


# create a folder to store the models if not exists
Path("llama_models").mkdir(parents=True, exist_ok=True)

# load embeddings of the questions
if TASK == "gsm8k":
    question_embedding_train = np.load("../embeddings/GSM8K_train_Q_embeddings.npy")
    question_embedding_test = np.load("../embeddings/GSM8K_test_Q_embeddings.npy")[
        :1000
    ]
    with open("LMllama2/gsm8k_names.json", "r") as f:
        offline_names = json.load(f)
elif TASK == "svamp":
    svamp_embeddings = np.load("../embeddings/aug_svamp_Q_embeddings.npy")
    question_embedding_train = svamp_embeddings[:15000]
    question_embedding_test = svamp_embeddings[15000:]
    with open("LMllama2/mawps_names.json", "r") as f:
        offline_names = json.load(f)
elif TASK == "mawps":
    svamp_embeddings = np.load("../embeddings/svamp_Q_embeddings.npy")
    question_embedding_train = svamp_embeddings[:6000]
    question_embedding_test = svamp_embeddings[6000:]
    with open("LMllama2/svamp_names.json", "r") as f:
        offline_names = json.load(f)

# load embeddings of the prompts
prompt_embedding = np.load("../embeddings/prompt_embeddings.npy")

# load test performance of the prompts
test_gold = []
for file_i in offline_names["test_set"]:
    test_gold.append(np.load(f"{file_i}"))
test_gold = np.asarray(test_gold)

# specify parameters via map
param = {"max_depth": 10, "eta": 0.001, "objective": "binary:logistic"}
num_round = 2000

print("===============================1 prompt====================================")
# enumerate all the combinations of 1 prompt
for i in range(len(prompt_embedding)):
    train_data = np.load(f"LMllama2/{inserted}training_data_{i}.npy")
    train_label = np.load(f"LMllama2/{inserted}training_labels_{i}.npy")
    concate_train_data = []

    # concate question embedding and prompt embedding
    for data in train_data:
        q_idx, p_idx = data
        concate_train_data.append(
            np.concatenate(
                (question_embedding_train[q_idx], prompt_embedding[p_idx]), axis=0
            )
        )
    concate_train_data = np.array(concate_train_data)
    train_label = np.array(train_label)

    shuffle_idx = np.arange(concate_train_data.shape[0])

    data_train = concate_train_data[
        shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
    ]
    label_train = train_label[shuffle_idx[: int(0.8 * concate_train_data.shape[0])]]
    data_val = concate_train_data[shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]]
    label_val = train_label[shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]]

    start_time = time.time()

    # convert data to 2-dim array first
    data_train = data_train.reshape(data_train.shape[0], -1)
    data_val = data_val.reshape(data_val.shape[0], -1)

    dtrain = xgb.DMatrix(data_train, label=label_train)
    dval = xgb.DMatrix(data_val, label=label_val)

    # check if the model exists
    try:
        bst = xgb.Booster()  # init model
        bst.load_model(f"llama_models/{inserted}xgboost_{i}.model")  # load data
        print("Model exists, loading...")
    except xgb.core.XGBoostError:
        print("Model does not exist, training...")
        bst = xgb.train(param, dtrain, num_round)

    # make prediction
    preds = bst.predict(dval)
    preds = preds > 0.5
    acc = accuracy_score(label_val, preds)
    print(f"Validation accuracy: {acc:.4f}")
    print(f"trivial acc pred: {sum(preds) / len(preds) :.4f}")
    print(f"trivial acc label: {sum(label_val) / len(label_val) :.4f}")
    # create llama_models folder if not exists
    Path("llama_models").mkdir(parents=True, exist_ok=True)
    bst.save_model(f"llama_models/{inserted}xgboost_{i}.model")

    print(f"Training time (single_model): {time.time() - start_time:.4f} seconds")


print("===============================6 prompts====================================")
# enumerate all the combinations of 6 prompts
for i in range(len(prompt_embedding)):
    for j in range(i + 1, len(prompt_embedding)):
        for k in range(j + 1, len(prompt_embedding)):
            for l in range(k + 1, len(prompt_embedding)):  # noqa: E741
                for m in range(l + 1, len(prompt_embedding)):
                    for n in range(m + 1, len(prompt_embedding)):
                        train_data = np.load(
                            f"LMllama2/{inserted}training_data_{i}_{j}_{k}_{l}_{m}_{n}.npy"
                        )
                        train_label = np.load(
                            f"LMllama2/{inserted}training_labels_{i}_{j}_{k}_{l}_{m}_{n}.npy"
                        )
                        concate_train_data = []

                        # concate question embedding and prompt embedding
                        for data in train_data:
                            q_idx, p_idx = data
                            concate_train_data.append(
                                np.concatenate(
                                    (
                                        question_embedding_train[q_idx],
                                        prompt_embedding[p_idx],
                                    ),
                                    axis=0,
                                )
                            )
                        concate_train_data = np.array(concate_train_data)
                        train_label = np.array(train_label)

                        shuffle_idx = np.arange(concate_train_data.shape[0])

                        data_train = concate_train_data[
                            shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
                        ]
                        label_train = train_label[
                            shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
                        ]
                        data_val = concate_train_data[
                            shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]
                        ]
                        label_val = train_label[
                            shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]
                        ]

                        start_time = time.time()

                        # convert data to 2-dim array first
                        data_train = data_train.reshape(data_train.shape[0], -1)
                        data_val = data_val.reshape(data_val.shape[0], -1)

                        dtrain = xgb.DMatrix(data_train, label=label_train)
                        dval = xgb.DMatrix(data_val, label=label_val)

                        # check if the model exists
                        try:
                            bst = xgb.Booster()  # init model
                            bst.load_model(
                                f"llama_models/{inserted}xgboost_{i}_{j}_{k}_{l}_{m}_{n}.model"
                            )  # load data
                            print("Model exists, loading...")
                        except xgb.core.XGBoostError:
                            print("Model does not exist, training...")
                            bst = xgb.train(param, dtrain, num_round)

                        # make prediction
                        preds = bst.predict(dval)
                        preds = preds > 0.5
                        acc = accuracy_score(label_val, preds)
                        print(f"Validation accuracy: {acc:.4f}")
                        print(f"trivial acc pred: {sum(preds) / len(preds) :.4f}")
                        print(
                            f"trivial acc label: {sum(label_val) / len(label_val) :.4f}"
                        )
                        bst.save_model(
                            f"llama_models/{inserted}xgboost_{i}_{j}_{k}_{l}_{m}_{n}.model"
                        )

                        print(
                            f"Training time (single_model): {time.time() - start_time:.4f} seconds"
                        )


print("===============================2 prompts====================================")
# enumerate all the combinations of 2 prompt
for i in range(len(prompt_embedding)):
    for j in range(i + 1, len(prompt_embedding)):
        # load training examples and labels
        train_data = np.load(f"LMllama2/{inserted}training_data_{i}_{j}.npy")
        train_label = np.load(f"LMllama2/{inserted}training_labels_{i}_{j}.npy")
        concate_train_data = []

        # concate question embedding and prompt embedding
        for data in train_data:
            q_idx, p_idx = data
            concate_train_data.append(
                np.concatenate(
                    (question_embedding_train[q_idx], prompt_embedding[p_idx]), axis=0
                )
            )
        concate_train_data = np.array(concate_train_data)
        train_label = np.array(train_label)

        shuffle_idx = np.arange(concate_train_data.shape[0])

        data_train = concate_train_data[
            shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
        ]
        label_train = train_label[shuffle_idx[: int(0.8 * concate_train_data.shape[0])]]
        data_val = concate_train_data[
            shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]
        ]
        label_val = train_label[shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]]

        start_time = time.time()

        # convert data to 2-dim array first
        data_train = data_train.reshape(data_train.shape[0], -1)
        data_val = data_val.reshape(data_val.shape[0], -1)

        dtrain = xgb.DMatrix(data_train, label=label_train)
        dval = xgb.DMatrix(data_val, label=label_val)

        # check if the model exists
        try:
            bst = xgb.Booster()  # init model
            bst.load_model(f"llama_models/{inserted}xgboost_{i}_{j}.model")  # load data
            print("Model exists, loading...")
        except xgb.core.XGBoostError:
            print("Model does not exist, training...")
            bst = xgb.train(param, dtrain, num_round)

        # make prediction
        preds = bst.predict(dval)
        preds = preds > 0.5
        acc = accuracy_score(label_val, preds)
        print(f"Validation accuracy: {acc:.4f}")
        print(f"trivial acc pred: {sum(preds) / len(preds) :.4f}")
        print(f"trivial acc label: {sum(label_val) / len(label_val) :.4f}")
        bst.save_model(f"llama_models/{inserted}xgboost_{i}_{j}.model")

        print(f"Training time (single_model): {time.time() - start_time:.4f} seconds")


print("===============================3 prompts====================================")
# enumerate all the combinations of 3 prompts
for i in range(len(prompt_embedding)):
    for j in range(i + 1, len(prompt_embedding)):
        for k in range(j + 1, len(prompt_embedding)):
            # load training examples and labels
            train_data = np.load(f"LMllama2/{inserted}training_data_{i}_{j}_{k}.npy")
            train_label = np.load(f"LMllama2/{inserted}training_labels_{i}_{j}_{k}.npy")
            concate_train_data = []

            # concate question embedding and prompt embedding
            for data in train_data:
                q_idx, p_idx = data
                concate_train_data.append(
                    np.concatenate(
                        (question_embedding_train[q_idx], prompt_embedding[p_idx]),
                        axis=0,
                    )
                )
            concate_train_data = np.array(concate_train_data)
            train_label = np.array(train_label)

            shuffle_idx = np.arange(concate_train_data.shape[0])

            data_train = concate_train_data[
                shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
            ]
            label_train = train_label[
                shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
            ]
            data_val = concate_train_data[
                shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]
            ]
            label_val = train_label[
                shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]
            ]

            start_time = time.time()

            # convert data to 2-dim array first
            data_train = data_train.reshape(data_train.shape[0], -1)
            data_val = data_val.reshape(data_val.shape[0], -1)

            dtrain = xgb.DMatrix(data_train, label=label_train)
            dval = xgb.DMatrix(data_val, label=label_val)

            # check if the model exists
            try:
                bst = xgb.Booster()  # init model
                bst.load_model(
                    f"llama_models/{inserted}xgboost_{i}_{j}_{k}.model"
                )  # load data
                print("Model exists, loading...")
            except xgb.core.XGBoostError:
                print("Model does not exist, training...")
                bst = xgb.train(param, dtrain, num_round)

            # make prediction
            preds = bst.predict(dval)
            preds = preds > 0.5
            acc = accuracy_score(label_val, preds)
            print(f"Validation accuracy: {acc:.4f}")
            print(f"trivial acc pred: {sum(preds) / len(preds) :.4f}")
            print(f"trivial acc label: {sum(label_val) / len(label_val) :.4f}")
            bst.save_model(f"llama_models/{inserted}xgboost_{i}_{j}_{k}.model")

            print(
                f"Training time (single_model): {time.time() - start_time:.4f} seconds"
            )


print("===============================4 prompts====================================")
# enumerate all the combinations of 4 prompts
for i in range(len(prompt_embedding)):
    for j in range(i + 1, len(prompt_embedding)):
        for k in range(j + 1, len(prompt_embedding)):
            for l in range(k + 1, len(prompt_embedding)):  # noqa: E741
                # load training examples and labels
                train_data = np.load(
                    f"LMllama2/{inserted}training_data_{i}_{j}_{k}_{l}.npy"
                )
                train_label = np.load(
                    f"LMllama2/{inserted}training_labels_{i}_{j}_{k}_{l}.npy"
                )
                concate_train_data = []

                # concate question embedding and prompt embedding
                for data in train_data:
                    q_idx, p_idx = data
                    concate_train_data.append(
                        np.concatenate(
                            (question_embedding_train[q_idx], prompt_embedding[p_idx]),
                            axis=0,
                        )
                    )
                concate_train_data = np.array(concate_train_data)
                train_label = np.array(train_label)

                shuffle_idx = np.arange(concate_train_data.shape[0])

                data_train = concate_train_data[
                    shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
                ]
                label_train = train_label[
                    shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
                ]
                data_val = concate_train_data[
                    shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]
                ]
                label_val = train_label[
                    shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]
                ]

                start_time = time.time()

                # convert data to 2-dim array first
                data_train = data_train.reshape(data_train.shape[0], -1)
                data_val = data_val.reshape(data_val.shape[0], -1)

                dtrain = xgb.DMatrix(data_train, label=label_train)
                dval = xgb.DMatrix(data_val, label=label_val)

                # check if the model exists
                try:
                    bst = xgb.Booster()  # init model
                    bst.load_model(
                        f"llama_models/{inserted}xgboost_{i}_{j}_{k}_{l}.model"
                    )  # load data
                    print("Model exists, loading...")
                except xgb.core.XGBoostError:
                    print("Model does not exist, training...")
                    bst = xgb.train(param, dtrain, num_round)

                # make prediction
                preds = bst.predict(dval)
                preds = preds > 0.5
                acc = accuracy_score(label_val, preds)
                print(f"Validation accuracy: {acc:.4f}")
                print(f"trivial acc pred: {sum(preds) / len(preds) :.4f}")
                print(f"trivial acc label: {sum(label_val) / len(label_val) :.4f}")
                bst.save_model(f"llama_models/{inserted}xgboost_{i}_{j}_{k}_{l}.model")

                print(
                    f"Training time (single_model): {time.time() - start_time:.4f} seconds"
                )


print("===============================5 prompts====================================")

# enumerate all the combinations of 5 prompts
for i in range(len(prompt_embedding)):
    for j in range(i + 1, len(prompt_embedding)):
        for k in range(j + 1, len(prompt_embedding)):
            for l in range(k + 1, len(prompt_embedding)):  # noqa: E741
                for m in range(l + 1, len(prompt_embedding)):
                    # load training examples and labels
                    train_data = np.load(
                        f"LMllama2/{inserted}training_data_{i}_{j}_{k}_{l}_{m}.npy"
                    )
                    train_label = np.load(
                        f"LMllama2/{inserted}training_labels_{i}_{j}_{k}_{l}_{m}.npy"
                    )
                    concate_train_data = []

                    # concate question embedding and prompt embedding
                    for data in train_data:
                        q_idx, p_idx = data
                        concate_train_data.append(
                            np.concatenate(
                                (
                                    question_embedding_train[q_idx],
                                    prompt_embedding[p_idx],
                                ),
                                axis=0,
                            )
                        )
                    concate_train_data = np.array(concate_train_data)
                    train_label = np.array(train_label)

                    shuffle_idx = np.arange(concate_train_data.shape[0])

                    data_train = concate_train_data[
                        shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
                    ]
                    label_train = train_label[
                        shuffle_idx[: int(0.8 * concate_train_data.shape[0])]
                    ]
                    data_val = concate_train_data[
                        shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]
                    ]
                    label_val = train_label[
                        shuffle_idx[int(0.8 * concate_train_data.shape[0]) :]
                    ]

                    start_time = time.time()

                    # convert data to 2-dim array first
                    data_train = data_train.reshape(data_train.shape[0], -1)
                    data_val = data_val.reshape(data_val.shape[0], -1)

                    dtrain = xgb.DMatrix(data_train, label=label_train)
                    dval = xgb.DMatrix(data_val, label=label_val)

                    # check if the model exists
                    try:
                        bst = xgb.Booster()  # init model
                        bst.load_model(
                            f"llama_models/{inserted}xgboost_{i}_{j}_{k}_{l}_{m}.model"
                        )  # load data
                        print("Model exists, loading...")
                    except xgb.core.XGBoostError:
                        print("Model does not exist, training...")
                        bst = xgb.train(param, dtrain, num_round)

                    # make prediction
                    preds = bst.predict(dval)
                    preds = preds > 0.5
                    acc = accuracy_score(label_val, preds)
                    print(f"Validation accuracy: {acc:.4f}")
                    print(f"trivial acc pred: {sum(preds) / len(preds) :.4f}")
                    print(f"trivial acc label: {sum(label_val) / len(label_val) :.4f}")
                    bst.save_model(
                        f"llama_models/{inserted}xgboost_{i}_{j}_{k}_{l}_{m}.model"
                    )

                    print(
                        f"Training time (single_model): {time.time() - start_time:.4f} seconds"
                    )
