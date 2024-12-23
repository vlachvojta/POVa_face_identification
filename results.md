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
Evaluated pairs of images with/without attribute (same person) (no limit)

+---------------------+--------------+----------------+----------------+----------------+----------------+-----------------+
|      **Attribute**      | **Accuracy (0)** | **Accuracy (0.4)** | **Accuracy (0.6)** | **Accuracy (0.7)** | **Accuracy (0.8)** | **Number of pairs** |
+---------------------+--------------+----------------+----------------+----------------+----------------+-----------------+
|   WEARING_NECKTIE   |     1.00     |      0.95      |      0.82      |      0.65      |      0.34      |      19442      |
|        GOATEE       |     1.00     |      0.93      |      0.75      |      0.55      |      0.26      |      13997      |
| FIVE_O_CLOCK_SHADOW |     1.00     |      0.94      |      0.77      |      0.54      |      0.23      |      23441      |
|       NO_BEARD      |     1.00     |      0.92      |      0.73      |      0.51      |      0.20      |      19214      |
|      SIDEBURNS      |     1.00     |      0.94      |      0.77      |      0.56      |      0.25      |      15122      |
|  RECEDING_HAIRLINE  |     0.99     |      0.92      |      0.75      |      0.54      |      0.25      |      18178      |
|        CHUBBY       |     0.99     |      0.93      |      0.77      |      0.58      |      0.31      |      12283      |
|    BUSHY_EYEBROWS   |     0.99     |      0.93      |      0.75      |      0.53      |      0.22      |      28628      |
|       MUSTACHE      |     0.99     |      0.92      |      0.75      |      0.54      |      0.26      |      10101      |
|     NARROW_EYES     |     0.99     |      0.92      |      0.72      |      0.49      |      0.20      |      28322      |
|   BAGS_UNDER_EYES   |     0.99     |      0.93      |      0.75      |      0.53      |      0.22      |      51535      |
|      BLACK_HAIR     |     0.99     |      0.93      |      0.74      |      0.52      |      0.22      |      41291      |
|     WEARING_HAT     |     0.99     |      0.87      |      0.56      |      0.33      |      0.11      |      14279      |
|     DOUBLE_CHIN     |     0.99     |      0.93      |      0.77      |      0.59      |      0.30      |      11232      |
|      OVAL_FACE      |     0.99     |      0.94      |      0.76      |      0.53      |      0.22      |      48393      |
|       SMILING       |     0.99     |      0.92      |      0.71      |      0.46      |      0.17      |      82188      |
| MOUTH_SLIGHTLY_OPEN |     0.99     |      0.92      |      0.71      |      0.47      |      0.18      |      89788      |
|      GRAY_HAIR      |     0.99     |      0.87      |      0.63      |      0.42      |      0.19      |       8006      |
|   HIGH_CHEEKBONES   |     0.99     |      0.92      |      0.71      |      0.46      |      0.17      |      75694      |
|       BIG_LIPS      |     0.99     |      0.92      |      0.69      |      0.45      |      0.18      |      19928      |
|    STRAIGHT_HAIR    |     0.99     |      0.92      |      0.72      |      0.49      |      0.20      |      51029      |
|       BIG_NOSE      |     0.99     |      0.92      |      0.72      |      0.49      |      0.21      |      38561      |
|     ROSY_CHEEKS     |     0.99     |      0.95      |      0.79      |      0.56      |      0.22      |      25609      |
|      WAVY_HAIR      |     0.99     |      0.92      |      0.71      |      0.46      |      0.17      |      61494      |
|     POINTY_NOSE     |     0.99     |      0.92      |      0.72      |      0.48      |      0.19      |      58770      |
|        BANGS        |     0.99     |      0.90      |      0.64      |      0.39      |      0.13      |      37360      |
|         BALD        |     0.99     |      0.94      |      0.75      |      0.53      |      0.24      |       3267      |
|   WEARING_NECKLACE  |     0.99     |      0.92      |      0.72      |      0.48      |      0.18      |      42621      |
|      ATTRACTIVE     |     0.99     |      0.91      |      0.69      |      0.44      |      0.16      |      60856      |
|   WEARING_EARRINGS  |     0.99     |      0.92      |      0.71      |      0.46      |      0.16      |      49356      |
|      BROWN_HAIR     |     0.99     |      0.92      |      0.70      |      0.46      |      0.16      |      53602      |
|        BLURRY       |     0.99     |      0.84      |      0.50      |      0.26      |      0.07      |      16870      |
|      EYEGLASSES     |     0.99     |      0.84      |      0.47      |      0.21      |      0.04      |      14512      |
|      PALE_SKIN      |     0.99     |      0.90      |      0.67      |      0.43      |      0.15      |      17770      |
|     HEAVY_MAKEUP    |     0.99     |      0.89      |      0.63      |      0.37      |      0.11      |      44659      |
|   ARCHED_EYEBROWS   |     0.99     |      0.91      |      0.69      |      0.44      |      0.16      |      50049      |
|   WEARING_LIPSTICK  |     0.99     |      0.87      |      0.58      |      0.31      |      0.09      |      34745      |
|        YOUNG        |     0.98     |      0.86      |      0.62      |      0.40      |      0.15      |      19025      |
|      BLOND_HAIR     |     0.98     |      0.88      |      0.63      |      0.37      |      0.12      |      26260      |
|         MALE        |     0.98     |      0.74      |      0.40      |      0.21      |      0.07      |       1820      |
+---------------------+--------------+----------------+----------------+----------------+----------------+-----------------+