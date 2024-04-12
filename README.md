# Bayesian optimization over the probability simplex
This repository contains the code of the algorithm BORIS used in the following paper:  

Candelieri, A., Ponti, A., & Archetti, F. **Bayesian optimization over the probability simplex.** _Annals of Mathematics and Artificial Intelligence, 1-15_ (2023). [https://doi.org/10.1007/s10472-023-09883-w](https://doi.org/10.1007/s10472-023-09883-w)

## Python dependencies
Use the `requirements.txt` file as reference.  
You can automatically install all the dependencies using the following command. 
````bash
pip install -r requirements.txt
````

## How to use the code
There are two main entrypoints:
- `run_bo.py`: run the experiments using the standard BO algorithm.
- `run_boris.py`: run the experiments using the BORIS algorithm.

In both scripts, it is possible to modify the test function and the number of variables.

## How to cite us
If you use this repository, please cite the following paper:
> [Candelieri, A., Ponti, A., & Archetti, F. Bayesian optimization over the probability simplex. Annals of Mathematics and Artificial Intelligence, 1-15 (2023) https://doi.org/10.1007/s10472-023-09883-w](https://doi.org/10.1007/s10472-023-09883-w)

```
@Article{candelieri2023bayesian,
  AUTHOR = {Candelieri, Antonio and Ponti, Andrea and Archetti, Francesco},
  TITLE = {Wasserstein enabled Bayesian optimization of composite functions},
  JOURNAL = {Annals of Mathematics and Artificial Intelligence},
  PAGES = {1--15},
  YEAR = {2023},
  PUBLISHER = {Springer}
  URL = {https://doi.org/10.1007/s10472-023-09883-w},
  DOI = {10.1007/s10472-023-09883-w}
}
```
