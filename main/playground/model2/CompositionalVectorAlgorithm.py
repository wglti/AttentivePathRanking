import time
import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import os
import json

import torch
import torch.optim as optim

from main.playground.model2.CompositionalVectorSpaceModel import CompositionalVectorSpaceModel
from main.playground.BatcherFileList import BatcherFileList
from main.experiments.Metrics import compute_scores
from main.playground.Logger import Logger
from main.playground.Visualizer import Visualizer


class CompositionalVectorAlgorithm:

    def __init__(self, experiment_dir, entity_type2vec_filename, learning_rate=0.1, weight_decay=0.0001, number_of_epochs=30, learning_rate_step_size=50, learning_rate_decay=0.5, visualize=False, best_models=None):
        self.entity_type2vec_filename = entity_type2vec_filename
        self.input_dirs = []
        self.entity_vocab = None
        self.relation_vocab = None
        self.entity_type_vocab = None
        self.load_data(experiment_dir)

        self.logger = Logger()

        # For visualizing results.
        # Note that visualization of paths takes huge large amount of disk space.
        self.visualize = visualize
        # if best models is provided, only train network till the iteration specified by the best model to save time.
        # Only used for visualization.
        self.best_models = best_models

        self.idx2entity = {v: k for k, v in self.entity_vocab.items()}
        self.idx2entity_type = {v: k for k, v in self.entity_type_vocab.items()}
        self.idx2relation = {v: k for k, v in self.relation_vocab.items()}
        self.visualizer = Visualizer(self.idx2entity, self.idx2entity_type, self.idx2relation,
                                     save_dir=os.path.join(experiment_dir, "results"))

        self.all_best_epoch_val_test = {}
        # best_epoch_val_test = {"epoch": -1, "val_acc": -1, "val_ap": -1, "test_acc": -1, "test_ap": -1}
        self.number_of_epochs = number_of_epochs

    def load_data(self, experiment_dir):
        data_dir = os.path.join(experiment_dir, "data")
        for folder in os.listdir(data_dir):
            if "data_output" in folder:
                input_dir = os.path.join(data_dir, folder)
                for fld in os.listdir(input_dir):
                    self.input_dirs.append(os.path.join(input_dir, fld))
            if "vocab" in folder:
                vocab_dir = os.path.join(data_dir, folder)
                for fld in os.listdir(vocab_dir):
                    if "entity_type_vocab" in fld:
                        entity_type_vocab_filename = os.path.join(vocab_dir, fld)
                        entity_type_vocab = json.load(open(entity_type_vocab_filename, "r"))
                        self.entity_type_vocab = entity_type_vocab
                    if "entity_vocab" in fld:
                        entity_vocab_filename = os.path.join(vocab_dir, fld)
                        self.entity_vocab = json.load(open(entity_vocab_filename, "r"))
                    if "relation_vocab" in fld:
                        relation_vocab_filename = os.path.join(vocab_dir, fld)
                        self.relation_vocab = json.load(open(relation_vocab_filename, "r"))

    def train_and_test(self):
        print("Training data directory: ", self.input_dirs)
        for input_dir in self.input_dirs:
            self.train(input_dir)
        accs = []
        aps = []
        for rel in self.all_best_epoch_val_test:
            best_model_score = self.all_best_epoch_val_test[rel]
            accs.append(best_model_score["test_acc"])
            aps.append(best_model_score["test_ap"])
        print("Average Accuracy:", sum(accs)/len(accs))
        print("Mean Average Precision:", sum(aps) / len(aps))

    def train(self, input_dir):
        print("Setting up model")
        model = CompositionalVectorSpaceModel(relation_vocab_size=len(self.relation_vocab),
                                              entity_vocab_size=len(self.entity_vocab),
                                              entity_type_vocab_size=len(self.entity_type_vocab),
                                              relation_embedding_dim=50,
                                              entity_embedding_dim=0,
                                              entity_type_embedding_dim=300, # entity_type_embedding_dim is fixed as 300.
                                              entity_type_vocab=self.entity_type_vocab,
                                              entity_type2vec_filename=self.entity_type2vec_filename,
                                              attention_dim=50,
                                              relation_encoder_dim=150,
                                              full_encoder_dim=150)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # self.optimizer = optim.Adagrad(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=learning_rate_step_size, gamma=learning_rate_decay)
        optimizer = optim.Adam(model.parameters())
        criterion = torch.nn.BCELoss().cuda()

        best_epoch_val_test = {"epoch": -1, "val_acc": -1, "val_ap": -1, "test_acc": -1, "test_ap": -1}
        rel = input_dir.split("/")[-1]
        train_files_dir = os.path.join(input_dir, "train")
        val_files_dir = os.path.join(input_dir, "dev")
        test_files_dir = os.path.join(input_dir, "test")
        print("Setting up train, val, and test batcher")
        train_batcher = BatcherFileList(train_files_dir, batch_size=24, shuffle=True, max_number_batchers_on_gpu=100)
        val_batcher = BatcherFileList(val_files_dir, batch_size=24, shuffle=False, max_number_batchers_on_gpu=100)
        test_batcher = BatcherFileList(test_files_dir, batch_size=24, shuffle=True, max_number_batchers_on_gpu=100)

        count = 0
        while True:
            data = train_batcher.get_batch()
            if data is None:
                break
            count += 1

        if self.best_models is not None:
            run_epochs = self.best_models[rel]["epoch"] + 1
        else:
            run_epochs = self.number_of_epochs

        # 1. training process
        for epoch in range(run_epochs):
            # self.scheduler.step()
            total_loss = 0
            start = time.time()

            for i in tqdm(range(count + 1)):
                data = train_batcher.get_batch()
                if data is not None:

                    inputs, labels = data
                    model.train()
                    model.zero_grad()
                    probs, path_weights, type_weights = model(inputs)
                    loss = criterion(probs, labels)

                    loss.backward()
                    # IMPORTANT: grad clipping is important if loss is large. May not be necessary for LSTM
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    total_loss += loss.item()

            time.sleep(1)
            print("Epoch", epoch, "spent", time.time() - start, "with total loss:", total_loss)

            if self.best_models is None:
                train_acc, train_ap = self.compute_acc_and_ap(model, train_batcher, rel, "train", epoch)
                val_acc, val_ap = self.compute_acc_and_ap(model, val_batcher, rel, "val", epoch)
                test_acc, test_ap = self.compute_acc_and_ap(model, test_batcher, rel, "test", epoch)
                self.logger.log_loss(total_loss, epoch, rel)
                self.logger.log_accuracy(train_acc, val_acc, test_acc, epoch, rel)
                self.logger.log_ap(train_ap, val_ap, test_ap, epoch, rel)
                for name, param in model.named_parameters():
                    self.logger.log_param(name, param, epoch)

                # selecting the best model based on performance on validation set
                if val_acc > best_epoch_val_test["val_acc"]:
                    best_epoch_val_test = {"epoch": epoch,
                                           "val_acc": val_acc, "val_ap": val_ap,
                                           "test_acc": test_acc, "test_ap": test_ap}

            else:
                if epoch == self.best_models[rel]["epoch"]:
                    train_acc, train_ap = self.compute_acc_and_ap(model, train_batcher, rel, "train", epoch)
                    test_acc, test_ap = self.compute_acc_and_ap(model, test_batcher, rel, "test", epoch)

        if self.best_models is None:
            print("Best model", best_epoch_val_test)
            # self.visualizer.save_space(rel, best_epoch_val_test["epoch"])
            self.all_best_epoch_val_test[rel] = best_epoch_val_test

    def compute_acc_and_ap(self, model, batcher, rel, split, epoch):
        score_instances = []
        with torch.no_grad():
            model.eval()
            batcher.reset()
            while True:
                data = batcher.get_batch()
                if data is None:
                    break
                inputs, labels = data
                probs, path_weights, type_weights = model(inputs)

                if self.visualize and split != "val":
                    if (self.best_models is None) or (epoch == self.best_models[rel]["epoch"]):
                        self.visualizer.visualize_paths_with_relation_and_type(inputs.clone().cpu().data.numpy(),
                                                        labels.clone().cpu().data.numpy(),
                                                        type_weights.clone().cpu().data.numpy(),
                                                        path_weights.clone().cpu().data.numpy(),
                                                        rel, split, epoch)
                        self.visualizer.visualize_paths(inputs.clone().cpu().data.numpy(),
                                                                               labels.clone().cpu().data.numpy(),
                                                                               type_weights.clone().cpu().data.numpy(),
                                                                               path_weights.clone().cpu().data.numpy(),
                                                                               rel, split, epoch)

                for label, prob in zip(labels, probs):
                    score_instances.append((None, label.item(), prob.item()))
                # print("accuracy for this batch of", inputs.shape[0], "examples is", num_correct / inputs.shape[0])
            # print("Total accuracy for training set:", total_num_correct / total_pairs)
        ap, rr, acc = compute_scores(score_instances)
        return acc, ap


if __name__ == "__main__":
    # 1. uncomment this part to train and test all relations for FB15k-237
    #cvsm = CompositionalVectorAlgorithm(experiment_dir="absolute_path_to/data/freebase15k237/cvsm_entity",
    #                                    entity_type2vec_filename=None)
    #cvsm.train_and_test()

    # 2. uncomment this part to train and test all relations for WN18RR
    cvsm = CompositionalVectorAlgorithm(experiment_dir="absolute_path_to/data/wordnet18rr/cvsm_entity",
                                        entity_type2vec_filename="absolute_path_to/data/wordnet18rr/entity_type2vec.pkl")
    cvsm.train_and_test()