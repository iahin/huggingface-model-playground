class Preproc:
    text_col = "text"
    label_col = "label"

class ModelConst:
    label_map = {'LABEL_0': '1',
                 'LABEL_1': '2',
                 'LABEL_2': '3',
                 'LABEL_3': '4',
                 'LABEL_4': '5'}
    max_length = 256


class FilePaths:
    resources_dir = "resources"
    
    model_dir = f"{resources_dir}/local_models"
    model_bin_path = f"{model_dir}/model_bin"
    tokenizer_path = f"{model_dir}/tokenizer"

    other_dir = f"{resources_dir}/other" #for other constraints, instructions, blacklists, etc

    dataset_dir = f"dataset"
    input_data = f"{dataset_dir}/input.csv"
    output_cache = f"{dataset_dir}/output.csv"
