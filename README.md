# deadwood_detection

This study is carried out as part of a 2nd year agronomy engineering school internship (equivalent to M1).<br />
Work in progress...<br />
![Current workflow](https://github.com/manon-col/deadwood_detection/edit/main/workflow.jpg?raw=true) <br /><br />
The following python packages are needed. For TensorFlow setup, please see documentation.<br />
```
pip install laspy
pip install pandas
pip install hdbscan
pip install scikit-learn
```
The following R packages are also needed.
```
install.packages("lidR")
# install.packages("devtools")
devtools::install_github("lmterryn/ITSMe", build_vignettes = TRUE)
```

## Documentation <br />
<br />

laspy, Python library for lidar LAS/LAZ IO: ``https://laspy.readthedocs.io/en/latest/index.html``<br />
scikit-learn, Machine-Learning in Python: ``https://scikit-learn.org/stable/``<br />
The hdbscan clustering library: ``https://hdbscan.readthedocs.io/en/latest/index.html``<br />
TensorFlow setup with pip: ``https://www.tensorflow.org/install/pip?hl=en``<br />
Convolutional Neural Network with TensorFlow: ``https://www.tensorflow.org/tutorials/images/cnn?hl=fr``<br />
NNCLR model: ``https://keras.io/examples/vision/nnclr/``<br />
TreeQSM: ``https://github.com/InverseTampere/TreeQSM``<br />
ITSMe: ``https://github.com/lmterryn/ITSMe``<br /><br />

## References <br />
<br />

Computree Core Team. 2017. Computree platform. Office National des Forêts, RDI Department. 
http://rdinnovation.onf.fr/computree. <br /><br />
[Plugin base]
Computree Core Team. 2017. Plugin Base for Computree. Office National des Forêts, RDI Department. 
http://rdinnovation.onf.fr/projects/computree. <br /><br />
[Plugin ONF]
Piboule Alexandre. 2017. Plugin ONF for Computree. Office National des Forêts, RDI Department. 
http://rdinnovation.onf.fr/projects/plugin-onf/wiki. <br /><br />
[Plugin SimpleTree]
Hackenberg Jan, Spiecker Heinrich, Calders Kim, Disney Mathias, Raumonen Pasi. 2015. SimpleTree - an efficient open source tool to build tree models from TLS clouds. Multidisciplinary Digital Publishing Institute. 
Rusu Radu Bogdan, Cousins Steve. 2011. 3d is here: Point cloud library (pcl). IEEE.
