class ExperimentConfig:
    # data_path = '/Users/isabel.chaves/Documents/MADSAD/MasterThesis/home-depot-product-search-relevance/'
    data_path = '/Users/isabel.chaves/Documents/MADSAD/MasterThesis/CrowdFlowerDataset/'
    language = {'acronym': 'ENG',
                'stemmer': 'english',
                'nltk': 'english'}

    vector_space = 'cosinesimil'
    # Brute force method (seq_search) -> Fast indexing, but slow for querying
    vector_method = 'seq_search'
    word_vectors_strategy = 'sum'
    click_graph_interaction_number = 100
