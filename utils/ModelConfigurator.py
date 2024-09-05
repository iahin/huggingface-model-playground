import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from utils.helper import file_exist
from config import FilePaths, ModelConst, Preproc

tqdm.pandas()


class ModelConfigurator:
    def __init__(self):
        self.output_cache = FilePaths().output_cache
        self.model_bin_path = FilePaths().model_bin_path
        self.tokenizer_path = FilePaths().tokenizer_path

        self.max_length = ModelConst().max_length

        gpu_check = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"INFO: Using {gpu_check}")
        self.device = torch.device(gpu_check)
        self.pipe = None

    def load_model(self):
        if not file_exist(self.model_bin_path) or not file_exist(self.tokenizer_path):
            print("model or tokenizer configs does not exist in path.")

        print(f"loading model from path: {self.model_bin_path}")
        print(f"loading tokenizer from path: {self.tokenizer_path}")

        model = AutoModelForSequenceClassification.from_pretrained(self.model_bin_path)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': self.max_length}
        self.pipe = pipeline("text-classification",
                             model=model, tokenizer=tokenizer, device=self.device,
                             **tokenizer_kwargs)

    def pipe_process(self, text):
        prediction = self.pipe(text)[0]
        label_result = ModelConst().label_map[prediction["label"]]
        return label_result

    def predict(self, df):
        print("carrying out inferance on data")
        df[[Preproc().label_col]] = df[Preproc().text_col].apply(lambda x: pd.Series(self.pipe_process(x)))
        df.drop(['description'], axis=1, inplace=True)
        print("inferance on data completed.")
        return df
