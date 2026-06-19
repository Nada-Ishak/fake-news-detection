# Dataset

Download **WELFake_Dataset.csv** and place it in this folder so the path
matches `ml/config.py`:

```
ml/data/WELFake_Dataset.csv
```

Source: WELFake dataset (Kaggle) - ~72k labeled news articles
(`label` = 0 -> Fake, 1 -> Real).

The CSV is intentionally **not committed to git** (see `.gitignore`) because
of its size — each developer/teammate downloads it locally before running
`python -m ml.train`.
