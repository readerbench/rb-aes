# Automated Essay Scoring with Bert and ReaderBench textual complexity indices

## Dataset preprocessing
python preprocess.py --file all.csv --dest out

## Hyperparameter tuning
python tuning.py out --gpu --minutes 300

## License
Apache 2.0. Please review the "LICENSE" file in the root of this directory.

## Acknowledgement
This research was supported by a grant of the Romanian National Authority for Scientific Research and Innovation, CNCS – UEFISCDI, project number TE 70 PN-III-P1-1.1-TE-2019-2209, ATES – “Automated Text Evaluation and Simplification”
