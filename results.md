# Results

## Measure of preprocessing effect on accuracy 
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

## Measure of accuracy based on filtered attributes
DataLoader is filtered by given attribute, then evaluted (limit = 100)

| **Attribute**          | **Accuracy** |
|:----------------------:|:------------:|
| FIVE_O_CLOCK_SHADOW    |     0.960    |
| ARCHED_EYEBROWS        |     0.960    |
| ATTRACTIVE             |     0.907    |
| BAGS_UNDER_EYES        |     0.987    |
| BALD                   |     0.988    |
| BANGS                  |     0.813    |
| BIG_LIPS               |     0.844    |
| BIG_NOSE               |     0.987    |
| BLACK_HAIR             |     0.933    |
| BLOND_HAIR             |     0.853    |
| BLURRY                 |     0.859    |
| BROWN_HAIR             |     0.974    |
| BUSHY_EYEBROWS         |     1.000    |
| CHUBBY                 |     0.936    |
| DOUBLE_CHIN            |     1.000    |
| EYEGLASSES             |     0.857    |
| GOATEE                 |     0.896    |
| GRAY_HAIR              |     0.922    |
| HEAVY_MAKEUP           |     0.960    |
| HIGH_CHEEKBONES        |     0.933    |
| MALE                   |     0.921    |
| MOUTH_SLIGHTLY_OPEN    |     0.973    |
| MUSTACHE               |     0.987    |
| NARROW_EYES            |     0.895    |
| NO_BEARD               |     0.853    |
| OVAL_FACE              |     0.893    |
| PALE_SKIN              |     0.831    |
| POINTY_NOSE            |     0.987    |
| RECEDING_HAIRLINE      |     0.987    |
| ROSY_CHEEKS            |     0.907    |
| SIDEBURNS              |     0.908    |
| SMILING                |     0.960    |
| STRAIGHT_HAIR          |     0.960    |
| WAVY_HAIR              |     0.907    |
| WEARING_EARRINGS       |     0.973    |
| WEARING_HAT            |     0.760    |
| WEARING_LIPSTICK       |     0.920    |
| WEARING_NECKLACE       |     0.961    |
| WEARING_NECKTIE        |     0.987    |
| YOUNG                  |     0.880    |

## Measure of accuracy based on paired attributes (image with/without attribute)
Evaluated pairs of images with/without attribute (same person) (limit = 1000)

| **Attribute**          | **Accuracy (0)** | **Accuracy (0.4)** | **Accuracy (0.8)** | **Number of pairs** |
|:----------------------:|:----------------:|:------------------:|:------------------:|:-------------------:|
| BALD                   |       1.00       |        0.96        |        0.00        |         23          |
| DOUBLE_CHIN            |       1.00       |        0.97        |        0.23        |         65          |
| GOATEE                 |       1.00       |        0.92        |        0.18        |        191          |
| MALE                   |       1.00       |        0.89        |        0.00        |          9          |
| SIDEBURNS              |       1.00       |        0.92        |        0.23        |        177          |
| WEARING_NECKTIE        |       1.00       |        0.97        |        0.24        |        161          |
| BLACK_HAIR             |       1.00       |        0.95        |        0.19        |        423          |
| BANGS                  |       0.99       |        0.90        |        0.15        |        364          |
| MUSTACHE               |       0.99       |        0.91        |        0.19        |        130          |
| BROWN_HAIR             |       0.99       |        0.92        |        0.14        |        505          |
| CHUBBY                 |       0.99       |        0.87        |        0.31        |        122          |
| NO_BEARD               |       0.99       |        0.93        |        0.16        |        226          |
| FIVE_O_CLOCK_SHADOW    |       0.99       |        0.92        |        0.23        |        304          |
| RECEDING_HAIRLINE      |       0.99       |        0.92        |        0.28        |        174          |
| BIG_LIPS               |       0.99       |        0.93        |        0.17        |        227          |
| ATTRACTIVE             |       0.99       |        0.91        |        0.19        |        617          |
| BIG_NOSE               |       0.98       |        0.89        |        0.17        |        424          |
| SMILING                |       0.98       |        0.92        |        0.18        |        769          |
| WEARING_EARRINGS       |       0.98       |        0.91        |        0.16        |        548          |
| HIGH_CHEEKBONES        |       0.98       |        0.92        |        0.19        |        645          |
| POINTY_NOSE            |       0.98       |        0.92        |        0.20        |        688          |
| OVAL_FACE              |       0.98       |        0.93        |        0.21        |        420          |
| MOUTH_SLIGHTLY_OPEN    |       0.98       |        0.92        |        0.20        |        886          |
| NARROW_EYES            |       0.98       |        0.91        |        0.18        |        254          |
| WEARING_HAT            |       0.98       |        0.88        |        0.09        |         96          |
| STRAIGHT_HAIR          |       0.98       |        0.93        |        0.19        |        517          |
| BUSHY_EYEBROWS         |       0.98       |        0.91        |        0.20        |        327          |
| WAVY_HAIR              |       0.98       |        0.90        |        0.20        |        615          |
| WEARING_NECKLACE       |       0.98       |        0.89        |        0.21        |        384          |
| BAGS_UNDER_EYES        |       0.97       |        0.89        |        0.22        |        553          |
| ROSY_CHEEKS            |       0.97       |        0.92        |        0.26        |        274          |
| PALE_SKIN              |       0.97       |        0.84        |        0.16        |        178          |
| ARCHED_EYEBROWS        |       0.96       |        0.88        |        0.20        |        505          |
| HEAVY_MAKEUP           |       0.96       |        0.83        |        0.12        |        433          |
| BLURRY                 |       0.96       |        0.83        |        0.09        |        150          |
| EYEGLASSES             |       0.96       |        0.73        |        0.03        |        162          |
| GRAY_HAIR              |       0.95       |        0.88        |        0.17        |         98          |
| WEARING_LIPSTICK       |       0.95       |        0.79        |        0.13        |        312          |
| BLOND_HAIR             |       0.94       |        0.83        |        0.10        |        267          |
| YOUNG                  |       0.94       |        0.82        |        0.21        |        185          |
