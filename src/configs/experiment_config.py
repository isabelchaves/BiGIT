class ExperimentConfig:
    data_path = '/home-depot-product-search-relevance/'
    # data_path = '/CrowdFlowerDataset/'
    language = {'acronym': 'ENG',
                'stemmer': 'english',
                'nltk': 'english'}

    vector_space = 'cosinesimil'
    # Brute force method (seq_search) -> Fast indexing, but slow for querying
    vector_method = 'seq_search'
    word_vectors_strategy = 'sum'
    bigit_interaction_number = 100
    bigit_initialization = 'document'
