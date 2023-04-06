import models as models
import dataloaders as dataloaders

MODELS = {
    'LightGCN': models.LightGCN,
}
DATASETS = {
    'gowalla': dataloaders.DataLoader,
    'yelp2018': dataloaders.DataLoader,
    'amazon-book': dataloaders.DataLoader,
}
