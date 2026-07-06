# Data

This directory intentionally does not include the raw datasets.

The original data files are excluded from the public version of this project because of copyright and redistribution restrictions. To run the notebook locally, create a `raw/` folder here and place the local data files in this structure:

```text
data/raw/DONOR.csv
data/raw/used_clothes_images.zip
```

Expected local files:

- `DONOR.csv`: structured donor dataset used for binary donation-response prediction.
- `used_clothes_images.zip`: image archive used for unsupervised clothing-condition clustering.

These files are required only when reproducing the notebook workflow locally.

The notebook excludes `AMOUNT` because it creates target leakage and excludes `ID` because it is an identifier rather than a predictive feature.
