from dataloaders.fcvid import FCVID
from dataloaders.activitynet import ActivityNet


def get_training_set(opt, spatial_transform, temporal_transform):

    if opt.dataset == 'FCVID':
        training_data = FCVID(
            opt,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform)
    elif opt.dataset == 'ActivityNet':
        training_data = ActivityNet(
            opt,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform)
    return training_data


def get_test_set(opt, spatial_transform, temporal_transform):

    if opt.dataset == 'FCVID':
        validation_data = FCVID(
            opt,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform)
    elif opt.dataset == 'ActivityNet':
        validation_data = ActivityNet(
            opt,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform)

    return validation_data
