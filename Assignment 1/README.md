# PHYS449

## Dependencies

- json
- numpy
- sys
- os

## Running `main.py`

To run `main.py`, use

```sh
python main.py data/file_name.in data/file_name.json
```

## Note

1. Assume *.in and *.json are in the same folder and have the same file name, and output *.out at the same directory and with same name.

2. Test case 2 demonstrates divergence handling.

3. Test case 3 has the same dataset as test case 2 but with lower "learning rate" to avoid divergence.