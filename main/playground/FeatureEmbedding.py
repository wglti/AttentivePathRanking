import torch
import torch.nn as nn

torch.manual_seed(1)


class FeatureEmbedding(nn.Module):

    def __init__(self, relation_vocab_size, relation_embedding_dim,
                 entity_vocab_size, entity_embedding_dim,
                 entity_type_vocab_size, entity_type_embedding_dim):
        super(FeatureEmbedding, self).__init__()

        # Improvement: zero embeddings for pad tokens
        self.relation_embeddings = nn.Embedding(relation_vocab_size, relation_embedding_dim).cuda()
        self.entity_types_embeddings = nn.Embedding(entity_type_vocab_size, entity_type_embedding_dim).cuda()
        self.entity_embeddings = nn.Embedding(entity_vocab_size, entity_embedding_dim).cuda()

    def forward(self, x):
        # debug: only support embedding all now.
        # the input dimension is #paths x #steps x #feats

        # 1. compute embeds for each feature
        # for each feature, num_entity_types type, 1 entity, 1 relation in order
        relation_embeds = self.relation_embeddings(x[:, :, -1])
        entity_embeds = self.entity_embeddings(x[:, :, -2])
        all_types_embeds = self.entity_types_embeddings(x[:, :, 0:-2])
        # sum all embeddings for types to get one summarized type embedding
        type_embeds = torch.sum(all_types_embeds, dim=2)

        # concat all embeds
        feature_embeds = torch.cat((type_embeds, entity_embeds, relation_embeds), dim=2)
        return feature_embeds
