import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import models

class NetUtils:
    def create_config(self, locals):
        config = {'model_class': self.__class__.__name__}
        locals_dict = {k: v for k, v in locals.items() if k not in ['self', '__class__']}
        config.update(locals_dict)
        return config

    @classmethod
    def from_config(cls, config):
        del config['model_class']
        return cls(**config)

class FacenetPytorchWrapper(torch.nn.Module, NetUtils):
    embedding_size = 512

    def __init__(self, model_name: str = 'vggface2'):
        super(self.__class__, self).__init__()
        self.config = self.create_config(locals())
        print(f'config: {self.config}')

        self.model = InceptionResnetV1(pretrained=model_name)

    def forward(self, x):
        return self.model(x)


class BasicResnet(torch.nn.Module, NetUtils):
    def __init__(self, embedding_size=512):
        super(self.__class__, self).__init__()
        self.config = self.create_config(locals())

        backbone = models.resnet50(pretrained=True)

        # Remove final classification layer
        backbone_modules = list(backbone.children())[:-1]
        self.backbone = torch.nn.Sequential(*backbone_modules)

        self.embedding = torch.nn.Linear(backbone.fc.in_features, embedding_size)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding(features)
        return embeddings


# class GraphRegressorBasic(torch.nn.Module, NetUtils):
#     def __init__(self, in_features = 2, conv_features = 16, conv_features_2 = 32, out_features = 2):
#         super(self.__class__, self).__init__()
#         self.config = self.create_config(locals())

#         # self.conv1 = GCNConv(in_features, conv_features)
#         # self.conv2 = GCNConv(conv_features, conv_features_2)
#         # self.pool = global_mean_pool
#         # self.fc = torch.nn.Linear(conv_features_2, out_features)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = torch.relu(x)
#         x = self.conv2(x, edge_index)
#         x = torch.relu(x)

#         # Aggregate node features into a single vector
#         x = self.pool(x, data.batch)  # TODO data MUST be batched (or add if-else)

#         # Process the graph representation
#         x = self.fc(x)

#         return x


# class GraphRegressorEmbed(torch.nn.Module, NetUtils):
#     def __init__(self, in_features = 2, embed_size = 32, conv_layers = 9, out_features = 2):
#         super(self.__class__, self).__init__()
#         self.config = self.create_config(locals())

#         self.embed = torch.nn.Linear(in_features, embed_size)
#         self.conv_layers = torch.nn.ModuleList([
#             GCNConv(embed_size, embed_size) for _ in range(conv_layers)
#         ])

#         self.pool = global_mean_pool
#         self.fc = torch.nn.Linear(embed_size, out_features)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.embed(x)

#         # Process the graph with multiple GCN layers
#         for conv in self.conv_layers:
#             x = conv(x, edge_index)
#             x = torch.relu(x)

#         x = self.pool(x, data.batch)  # data MUST be batched (or add if-else)
#         x = self.fc(x)

#         return x

# class EnhancedGraphRegressor(torch.nn.Module, NetUtils):
#     def __init__(self, in_features = 2, embed_size = 32, conv_layers = 9, attention_heads = 4, decoder_layers = 4, out_features = 2):
#         super(self.__class__, self).__init__()
#         self.config = self.create_config(locals())

#         self.embed = torch.nn.Linear(in_features, embed_size)
#         self.conv_layers = torch.nn.ModuleList([
#             GCNConv(embed_size, embed_size) for _ in range(conv_layers)
#         ])

#         self.attention = torch.nn.TransformerEncoderLayer(d_model=embed_size, nhead=attention_heads)
#         self.transformer_encoder = torch.nn.TransformerEncoder(self.attention, num_layers=decoder_layers)
#         self.pool = global_mean_pool
#         self.fc = torch.nn.Linear(embed_size, out_features)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.embed(x)

#         # Process the graph with multiple GCN layers
#         for conv in self.conv_layers:
#             x = conv(x, edge_index)
#             x = torch.relu(x)

#         # Apply attention mechanism
#         x = self.transformer_encoder(x)

#         # Aggregate node features into a single vector
#         x = self.pool(x, data.batch)

#         # Process the graph representation
#         x = self.fc(x)

#         return x


# class GraphRegressorBasicBlocksUsed(torch.nn.Module, NetUtils):
#     def __init__(self, in_features = 2, embed_size = 32, conv_layers = 9, out_features = 1):
#         super(self.__class__, self).__init__()
#         self.config = self.create_config(locals())

#         self.embed = torch.nn.Linear(in_features, embed_size)
#         self.conv_layers = torch.nn.ModuleList([
#             GCNConv(embed_size, embed_size) for _ in range(conv_layers)
#         ])

#         self.pool = global_add_pool
#         self.fc = torch.nn.Linear(embed_size, out_features)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.embed(x)

#         # Process the graph with multiple GCN layers
#         for conv in self.conv_layers:
#             x = conv(x, edge_index)
#             x = torch.relu(x)

#         x = self.pool(x, data.batch)  # data MUST be batched (or add if-else)
#         x = self.fc(x)

#         return x

# # ---------------------- New models ----------------------

# class GraphRegressorEmbedNew(torch.nn.Module, NetUtils):
#     def __init__(self, in_features = 2, embed_size = 32, conv_layers = 9, out_features = 2):
#         super(self.__class__, self).__init__()
#         self.config = self.create_config(locals())

#         self.embed = torch.nn.Linear(in_features, embed_size)
#         self.conv_layers = torch.nn.ModuleList([
#             GCNConv(embed_size, embed_size) for _ in range(conv_layers)
#         ])

#         self.pool = global_mean_pool
#         self.fc = torch.nn.Linear(embed_size, out_features)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.embed(x)

#         # Process the graph with multiple GCN layers
#         for conv in self.conv_layers:
#             x = conv(x, edge_index)
#             x = torch.relu(x)

#         x = self.pool(x, data.batch)  # data MUST be batched (or add if-else)
#         x = self.fc(x)

#         return x


# if __name__ == '__main__':
#     print('Net definitions')
#     # print all available functions of GraphRegressorBasic
#     print('GraphRegressorBasic')

#     model = GraphRegressorBasic()
#     print(f'model config: {model.config}')