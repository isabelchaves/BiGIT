import pandas as pd

from configs.experiment_config import ExperimentConfig
from data.processing.textual_processing import PreProcessing
from evaluation.evaluation_method import EvaluationMethod


def run_predictions(model_class, product_vector_space, product_ids):
    data_to_predict = pd.read_csv(ExperimentConfig.data_path + 'test_set.csv', encoding='latin-1', index_col=0)
    language_process = PreProcessing(language=ExperimentConfig.language)
    data_to_predict['search_term_processed'] = data_to_predict['search_term'].apply(
        lambda x: language_process.tokenizer(x))
    data_to_predict['product_title_processed'] = data_to_predict['product_title'].apply(
        lambda x: language_process.tokenizer(x))

    products, queries = model_class._prepare_data(data=data_to_predict)

    EvaluationMethod(product_ids=product_ids).run(data=data_to_predict,
                                                  data_to_evaluate=queries,
                                                  vector_space_to_search=product_vector_space,
                                                  evaluate_column='search_term_processed')
