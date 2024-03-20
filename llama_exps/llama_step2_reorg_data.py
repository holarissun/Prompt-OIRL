import numpy as np
import json

unified_alias_list = ["nan", "CoT", "APE", "Dis", "ToT", "Decomp"]

for dataset in ["mawps", "gsm8k", "svamp"]:
    # split train and test
    for prompt_i in unified_alias_list:
        # read txt file
        with open(
            f"llama2_chat_logs/{dataset}/llama_{dataset}_corr_{prompt_i}.txt", "r"
        ) as f:
            prompt_i_file = f.readlines()
        # remove the \n
        prompt_i_file_check = ["False" in i for i in prompt_i_file]
        prompt_i_file = ["True" in i for i in prompt_i_file]
        prompt_i_file = np.array(prompt_i_file)
        prompt_i_file_check = np.array(prompt_i_file_check)
        assert sum(prompt_i_file_check == prompt_i_file) == 0

        # split the data into train and test
        if dataset == "svamp":
            train_i = prompt_i_file[:15000]
            test_i = prompt_i_file[15000:]
        elif dataset == "mawps":
            train_i = prompt_i_file[:6000]
            test_i = prompt_i_file[6000:]
        elif dataset == "gsm8k":
            train_i = prompt_i_file
            with open(
                f"llama2_chat_logs/{dataset}/llama_{dataset}_test_corr_{prompt_i}.txt",
                "r",
            ) as f:
                prompt_i_file = f.readlines()
            # remove the \n
            prompt_i_file_check = ["False" in i for i in prompt_i_file]
            prompt_i_file = ["True" in i for i in prompt_i_file]
            prompt_i_file = np.array(prompt_i_file)
            prompt_i_file_check = np.array(prompt_i_file_check)
            assert sum(prompt_i_file_check == prompt_i_file) == 0
            test_i = prompt_i_file
        # save the data
        np.save(f"LMllama2/{dataset}_train_{prompt_i}.npy", train_i)
        np.save(f"LMllama2/{dataset}_test_{prompt_i}.npy", test_i)

    # save all those information as a dictionary, and save the dictionary as a json file
    save_dict = {}
    save_dict["prompts"] = [
        "The answer is:",
        "Let's think step by step: ",
        "Let’s work this out in a step by step way to be sure we have the right answer: ",
        "3 experts are discussing the question with a panel discussion, trying to solve it step by step, and make sure the result is correct and avoid penalty: ",
        "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave",
        "First, decompose the question into several sub-questions that needs to solve, and then solve each question step by step: ",
    ]

    save_dict["training_set"] = [
        f"LMllama2/{dataset}_train_nan.npy",
        f"LMllama2/{dataset}_train_CoT.npy",
        f"LMllama2/{dataset}_train_APE.npy",
        f"LMllama2/{dataset}_train_Dis.npy",
        f"LMllama2/{dataset}_train_ToT.npy",
        f"LMllama2/{dataset}_train_Decomp.npy",
    ]

    save_dict["test_set"] = [
        f"LMllama2/{dataset}_test_nan.npy",
        f"LMllama2/{dataset}_test_CoT.npy",
        f"LMllama2/{dataset}_test_APE.npy",
        f"LMllama2/{dataset}_test_Dis.npy",
        f"LMllama2/{dataset}_test_ToT.npy",
        f"LMllama2/{dataset}_test_Decomp.npy",
    ]

    # save the dictionary as a json file
    with open(f"LMllama2/{dataset}_names.json", "w") as f:
        json.dump(save_dict, f)
