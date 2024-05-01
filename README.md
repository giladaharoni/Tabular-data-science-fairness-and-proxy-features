# Fairness and proxy variables: Identifying and Mitigating Information Leakage for Fair Data Science
This is the repository of our final project in Tabular Data Science course in Bar Ilan University. 
## Jupyter notebook example
You can run a notebook version of our methods `TDS_example.ipynb` demonstrated on the Students Performances dataset. This notebook demand online connection for downloading the data.
## main.py - Reconstruction of the experiment
You can try the whole exepriment again using the the `main.py` file and see the results in live by the whole 4 datasets: German credit, Students performances, Adults salary and COMPAS (criminial data).
for running the program please use one of the words 'german','students','adults','comapss' as an argument for running the process using the respective dataset.
. For example:
```
python main.py german
```
Some of the data may be downloaded during running the program. This program may take a while due the exponential complexity of the brute force method.
