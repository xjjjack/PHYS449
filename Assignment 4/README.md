# PHYS449

## Dependencies

- pathlib
- numpy
- argparse
- matplotlib
- itertools

## Running `main.py`

To run `main.py`, use

```sh
python main.py data/in.txt --iteration 1000 --learning-rate 0.01 --step 50 --regularizer 1e-2 --res-path results 
```

## Note
All arguments are optional. Values in above are already set as default value. So it's equivalent to:

```sh
python main.py data/in.txt
```

Output KL divergence as .txt file and .pdf figure (examples are stored in 'result' folder).

The program also prints out the dictionary of couplers where keys are pairs of indices and values are the predicted values for the couplers

There are arguments computing the expectation and empirical possibility of the input dataset. Example usage:

```sh
python main.py --compute-expectation data/in.txt
```

```sh
python main.py --compute-empirical data/in.txt
```