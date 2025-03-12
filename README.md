# KMProt

## Reproduce best submission
This will execute the code to generate our second-best submission (0.7193, ranked 2/37 on the leaderboard), as our top-performing submission (0.7227, also ranked 2/37) requires computing a weighted sum of 15 kernels, which is significantly more time-intensive.

```
pip install -r requirements.txt
python -m src.start
````
If you still want to get the best submission, change `config.json` by `best_config.json` in `src/start.py`

## Run a gridsearch

First update the paths to the folder that fill contain the saved gram matrices and experiment reports (json files) in `src/config.py`. Then update `src/gridsearch.py` file to change the ranges of the hyperparameters that you want to test. Then run : 

```
python -m src.gridsearch
```

## Run ensembling search

First update the paths to the folder that fill contain the ensemble experiment reports (json files) in `src/config.py`. Then update `src/ensemble_search.py` file to change the number of kernels included in the sum of the weighted sum kernel. Then run : 
```
python -m src.ensemble_search
```
