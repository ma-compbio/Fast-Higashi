# Fast-Higashi: Ultrafast and interpretable single-cell 3D genome analysis
[https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00395-7](https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00395-7)

Fast-Higashi is an interpretable model that takes single-cell Hi-C (scHi-C) contact maps as input and jointly infers cell embeddings as well as meta-interactions.
![figs/fig1.png](https://github.com/ma-compbio/Fast-Higashi/blob/main/figs/fig1.png)
# Installation

We now have Fast-Higashi on conda!

Do 
`conda install -c ruochiz fasthigashi`
or
`mamba install -c ruochiz fasthigashi`

After that install the latest pytorch with corresponding CUDA support. Check https://pytorch.org for details. Note that fasthigashi won't check if you have pytorch installed. So, the user would have to install the correct pytorch version individually.

```{bash}
git clone https://github.com/ma-compbio/Fast-Higashi/
cd Fast-Higashi
python setup.py install
```

It is recommended to have pytorch installed (with CUDA support when applicable) before installing higashi.

# Documentation
The input format would be exactly the same as the Higashi software. 
Detailed documentation will be updated here at the [Higashi wiki](https://github.com/ma-compbio/Higashi/wiki/Fast-Higashi-Usage)

# Tutorial
- [Lee et al. (sn-m3c-seq on PFC)](https://github.com/ma-compbio/Fast-Higashi/blob/main/PFC%20tutorial.ipynb)

# Cite

Cite our paper by

```
@article {Zhang2022fast,
	author = {Zhang, Ruochi and Zhou, Tianming and Ma, Jian},
	title = {Ultrafast and interpretable single-cell 3D genome analysis with Fast-Higashi},
	year = {2022},
	doi = {10.1016/j.cels.2022.09.004},
	journal={Cell systems},
  	volume={13},
  	number={10},
  	pages={798--807},
  	year={2022},
  	publisher={Elsevier}
}
```

![figs/Overview.png](https://github.com/ma-compbio/Fast-Higashi/blob/main/figs/higashi_cellsystems.png)



# Contact

Please contact zhangruo@broadinstitute.org or raise an issue in the github repo with any questions about installation or usage. 
