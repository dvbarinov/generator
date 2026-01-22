# Generator
File generator for testing purposes.

## Tecnologies
- Python

## Launch
### Binary files
```bash
python generate_files.py --count 100 --size 1024 --output ./test_data
```
### Formatted files
Pay attention to the content type.
```bash
python generate_files.py -n 100 --size 0 --content-type text --text-lines 20
python generate_files.py -n 100 --size 0 --content-type json --json-schema log
python generate_files.py -n 100 --content-type image --image-size 640x480 --image-format png
python generate_files.py -n 10 --size 0 --content-type csv --csv-rows 1000
python generate_files.py -n 10 --size 0 --content-type xml --xml-items 100
python generate_files.py -n 10 --size 0 --content-type pdf
python generate_files.py -n 10 --content-type svg --svg-size 500x300
```