# Automated Essay Scoring with Bert and ReaderBench textual complexity indices

## Dataset preprocessing
python preprocess.py --file all.csv --dest out

## Hyperparameter tuning
python tuning.py out --gpu --minutes 300
