# Build your own CenterFace/Centernet

For English developer, I still believe this repo will be helpful.

### Reading

Codes in this repo are almost self-explained, but trying to read the codes in order will be better.

- 0: reading the Paper, Centernet, CenterFace, optional Cornernet
- 1: datasets.py: parsing annotations, image preprocessing, label generating
- 2: config.py: hyperparameters
- 3: utils.py: image preprocessing, postprocessing
- 4: models: loss function, backbone
- 5: train.py: training pipeline
- 6: api.py: inference pipeline

### Train your own data

Firstly, you should be familiar with the data and annotation format. In WiderFace and other common dataset, the formats are simple.

- images: folder contains images
- annotations: always a txt file

In this repo, I assume your annotation format is `retinaface` like. If what you need is object detection, then the following format is enough.

```sh
# image name
left top width height
left top width height
# image name
....
```

Secondly, remove everything about facial landmarks including labels generating, image preprocessing and training. Start from here

```py
# code in datasets.py
im, bboxes, landmarks = self.preprocess(im, anns)
hm = self.make_heatmaps(im, bboxes, landmarks, self.downscale)
```

### Wishes

Anyone who have saw, heard, inspected or used this repo gains temporary happiness and everlasting happiness.
