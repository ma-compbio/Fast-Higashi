# Fast-Higashi: Ultrafast and interpretable single-cell 3D genome analysis
https://www.biorxiv.org/content/10.1101/2022.04.18.488683v1

Fast-Higashi is an interpretable model that takes single-cell Hi-C (scHi-C) contact maps as input and jointly infers cell embeddings as well as meta-interactions.
![figs/fig1.png](https://github.com/ma-compbio/Fast-Higashi/blob/main/figs/fig1.png)
# Installation

We now have Fast-Higashi on conda as well!

`conda install -c ruochiz fasthigashi` (only for linux)

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
@article {Zhang2022.04.18.488683,
	author = {Zhang, Ruochi and Zhou, Tianming and Ma, Jian},
	title = {Ultrafast and interpretable single-cell 3D genome analysis with Fast-Higashi},
	elocation-id = {2022.04.18.488683},
	year = {2022},
	doi = {10.1101/2022.04.18.488683},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/04/19/2022.04.18.488683},
	eprint = {https://www.biorxiv.org/content/early/2022/04/19/2022.04.18.488683.full.pdf},
	journal = {bioRxiv}
}
```

![figs/Overview.png](https://github.com/ma-compbio/Fast-Higashi/blob/main/figs/higashi_title.png)



# Contact

Please contact ruochiz@andrew.cmu.edu or raise an issue in the github repo with any questions about installation or usage. 
