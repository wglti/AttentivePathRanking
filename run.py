from main.playground.model2.CompositionalVectorAlgorithm import CompositionalVectorAlgorithm


if __name__ == "__main__":
    # 1. uncomment this part to train and test all relations for FB15k-237
    cvsm = CompositionalVectorAlgorithm(experiment_dir="data/freebase15k237/cvsm_entity",
                                        entity_type2vec_filename=None)
    cvsm.train_and_test()

    # 2. uncomment this part to train and test all relations for WN18RR
    #cvsm = CompositionalVectorAlgorithm(experiment_dir="data/wordnet18rr/cvsm_entity",
    #                                    entity_type2vec_filename="data/wordnet18rr/entity_type2vec.pkl")
    #cvsm.train_and_test()