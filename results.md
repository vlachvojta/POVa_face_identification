# Results

## Meassure of preprocessing effect on accuracy 
Accuracy: (highest similarity of all classes is the correct one)
Data: Val dataset (first 1000 images)

| **Normalize** |    **Squarify**    | **Resize** | **Accuracy** |
|:-------------:|:------------------:|:----------:|:------------:|
|    -1 až 1    | AROUND_FACE_STRICT |     160    |   **0.876**  |
|    -1 až 1    |     AROUND_FACE    |     --     |     0.862    |
|    -1 až 1    |     AROUND_FACE    |     160    |     0.862    |
|    mean_std   |     AROUND_FACE    |     --     |     0.862    |
|    mean_std   |     AROUND_FACE    |     160    |     0.862    |
|    min_max    |     AROUND_FACE    |     --     |     0.819    |
|    min_max    |     AROUND_FACE    |     160    |     0.819    |
|     0 až 1    |     AROUND_FACE    |     --     |     0.817    |
|     0 až 1    |     AROUND_FACE    |     160    |     0.817    |
|    mean_std   |        CROP        |     --     |     0.796    |
|    mean_std   |        CROP        |     160    |     0.796    |
|    -1 až 1    |        CROP        |     --     |     0.795    |
|    -1 až 1    |        CROP        |     160    |     0.795    |
|    min_max    |        CROP        |     --     |     0.760    |
|    min_max    |        CROP        |     160    |     0.760    |
|     0 až 1    |        CROP        |     --     |     0.756    |
|     0 až 1    |        CROP        |     160    |     0.756    |
|    mean_std   |         --         |     --     |     0.751    |
|    mean_std   |         --         |     160    |     0.751    |
|    -1 až 1    |         --         |     --     |     0.750    |
|    -1 až 1    |         --         |     160    |     0.749    |
|     0 až 1    |         --         |     160    |     0.714    |
|     0 až 1    |         --         |     --     |     0.710    |
|    min_max    |         --         |     --     |     0.710    |
|    min_max    |         --         |     160    |     0.710    |
|       --      |     AROUND_FACE    |     --     |     0.028    |
|       --      |     AROUND_FACE    |     160    |     0.028    |
|       --      |         --         |     --     |     0.026    |
|       --      |         --         |     160    |     0.026    |
|       --      |        CROP        |     --     |     0.023    |
|       --      |        CROP        |     160    |     0.023    |


Je to nějaký divný, padalo mi to v průběhu a při dalším měření už to vyšlo jinak, idk....									
Accuracy: 0.079, Normalize: Normalization.IMAGE_NET, Squarify: Squarify.AROUND_FACE, Resize: None     Accuracy: 0.031, Normalize: Normalization.IMAGE_NET, Squarify: Squarify.AROUND_FACE_STRICT, Resize: 1>Accuracy: 0.031, Normalize: Normalization.IMAGE_NET, Squarify: Squarify.AROUND_FACE_STRICT, Resize: 2>Accuracy: 0.082, Normalize: Normalization.IMAGE_NET, Squarify: Squarify.AROUND_FACE, Resize: 160      Accuracy: 0.079, Normalize: Normalization.IMAGE_NET, Squarify: Squarify.AROUND_FACE, Resize: 224      Accuracy: 0.073, Normalize: Normalization.IMAGE_NET, Squarify: None, Resize: 160                      Accuracy: 0.072, Normalize: Normalization.IMAGE_NET, Squarify: None, Resize: 224
Accuracy: 0.077, Normalize: Normalization.IMAGE_NET, Squarify: Squarify.CROP, Resize: 160             Accuracy: 0.073, Normalize: Normalization.IMAGE_NET, Squarify: Squarify.CROP, Resize: 224
Accuracy: 0.023, Normalize: Normalization._0_1, Squarify: Squarify.AROUND_FACE_STRICT, Resize: 160    Accuracy: 0.023, Normalize: Normalization._0_1, Squarify: Squarify.AROUND_FACE_STRICT, Resize: 224
Accuracy: 0.077, Normalize: Normalization._0_1, Squarify: Squarify.AROUND_FACE, Resize: 160 