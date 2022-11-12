# PHYS449

## Dependencies

- json
- pathlib
- numpy
- argparse
- matplotlib
- copy

'requirements.txt' can be used to install non-built-in packages.

## Running `main.py`

To run `main.py`, use

```sh
python main.py --param param/param.json --train-size 10000 --test-size 100 --seed 8888 --res-path results
```

## Note
All arguments are optional. Values in above are already set as default value.

Output figure is modified to show both loss and accuracy.

Tuned json file is provided.