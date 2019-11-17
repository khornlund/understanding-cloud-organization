{
    "type": "FPN",
    "args": {
        "encoder_name": "dpn131",
        "dropout": dropout,
        "decoder_merge_policy": "cat",
    },
    "batch_size": 10,
    "augmentation": {"type": transforms, "args": {"height": 320, "width": 480}},
},
{
    "type": "FPN",
    "args": {
        "encoder_name": "densenet161",
        "dropout": dropout,
        "decoder_merge_policy": "cat",
    },
    "batch_size": 16,
    "augmentation": {"type": transforms, "args": {"height": 320, "width": 480}},
},
