# MAPERT: Map Encoder Representations from Transformers
## Introduction
PyTorch code for the SpLU-RoboNLP 2021 paper ["Learning to Read Maps: Understanding Natural Language Instructions from Unseen Maps"](https://aclanthology.org/2021.splurobonlp-1.2.pdf). 



### ROSMI

#### Fine-tuning

1. Download the ROSMI data from the official [GitHub repo](https://github.com/marioskatsak/rosmi-dataset).



REPO ROOT
 |
 |-- data                  
 |    |-- rosmi
 
 | 
 |-- snap
 |-- src
```

Please also kindly contact us if anything is missing!




## Faster R-CNN Feature Extraction


We use the Faster R-CNN feature extractor demonstrated in ["Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering", CVPR 2018](https://arxiv.org/abs/1707.07998)
and its released code at [Bottom-Up-Attention github repo](https://github.com/peteanderson80/bottom-up-attention).







## Reference

```bibtex
@inproceedings{katsakioris-etal-2021-learning,
    title = "Learning to Read Maps: Understanding Natural Language Instructions from Unseen Maps",
    author = "Katsakioris, Miltiadis Marios  and
      Konstas, Ioannis  and
      Mignotte, Pierre Yves  and
      Hastie, Helen",
    booktitle = "Proceedings of Second International Combined Workshop on Spatial Language Understanding and Grounded Communication for Robotics",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.splurobonlp-1.2",
    doi = "10.18653/v1/2021.splurobonlp-1.2",
    pages = "11--21",
    abstract = "Robust situated dialog requires the ability to process instructions based on spatial information, which may or may not be available. We propose a model, based on LXMERT, that can extract spatial information from text instructions and attend to landmarks on OpenStreetMap (OSM) referred to in a natural language instruction. Whilst, OSM is a valuable resource, as with any open-sourced data, there is noise and variation in the names referred to on the map, as well as, variation in natural language instructions, hence the need for data-driven methods over rule-based systems. This paper demonstrates that the gold GPS location can be accurately predicted from the natural language instruction and metadata with 72{\%} accuracy for previously seen maps and 64{\%} for unseen maps.",
}
```



License
-------

Distributed under the [Creative Commons 4.0 Attribution-ShareAlike license
(CC4.0-BY-SA)](https://creativecommons.org/licenses/by-sa/4.0/).



## Acknowledgement
This research received funding by Seebyte, Datalab and MASTS-S. This work was also supported by the EPSRC funded ORCA Hub (EP/R026173/1, 2017-2021).

