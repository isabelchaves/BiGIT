import pandas as pd

from src.configs.experiment_config import ExperimentConfig
from src.configs.variables_const import VariablesConsts
from src.data.processing.textual_processing import PreProcessing
from src.evaluation.evaluation_method import EvaluationMethod


def run_predictions(model_class, product_vector_space, product_ids):
    data_to_predict = pd.read_parquet(ExperimentConfig.data_path + 'test_set.parquet')
    language_process = PreProcessing(language=ExperimentConfig.language)
    data_to_predict[VariablesConsts.SEARCH_TERM_PROCESSED] = data_to_predict[VariablesConsts.SEARCH_TERM].apply(
        lambda x: language_process.tokenizer(x))
    data_to_predict[VariablesConsts.PRODUCT_TITLE_PROCESSED] = data_to_predict[VariablesConsts.PRODUCT_TITLE].apply(
        lambda x: language_process.tokenizer(x))

    products, queries = model_class.prepare_data(data=data_to_predict)

    EvaluationMethod(product_ids=product_ids).run(data=data_to_predict,
                                                  data_to_evaluate=queries,
                                                  vector_space_to_search=product_vector_space,
                                                  evaluate_column=VariablesConsts.SEARCH_TERM_PROCESSED)
