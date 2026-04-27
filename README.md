# Sparse Independent Component Analysis without Assuming Non-Gaussianity

This repository contains an implementation of the independent component analysis method described in ["On the Identifiability of Sparse ICA without Assuming Non-Gaussianity"](https://arxiv.org/abs/2408.10353). 

If you find it useful, please consider citing:
```bibtex
@inproceedings{ng2023identifiability,
  author = {Ng, Ignavier and Zheng, Yujia and Dong, Xinshuai and Zhang, Kun},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {On the Identifiability of Sparse ICA without Assuming Non-Gaussianity},
  year = {2023}
}

```

## Requirements
- Python 3.6+
- `numpy`
- `scipy`

## Running the Methods
- To run an example for both decomposition-based and likelihood-based methods, run the following:
```
python main.py
```

## Acknowledgments
- Parts of the methods and optimization procedure are adapted from  [notears](https://github.com/xunzheng/notears/blob/master/notears/utils.py) and [notears-convergence](https://github.com/ignavierng/notears-convergence).