# PK-neuralnet
A Python application for training/testing and exporting to NACARTE (SpaceClaim) of a Neural Network

## Installation
To install the package, download the project and type on your terminal: <br>
`make install`

## Functionalities
* Create a database
`databasepknn --config /path/to/config.yaml` <br>
* Train a PK-neuralnet
`trainpknn --config /path/to/config.yaml` <br>
* Prediction with a trained PK-neuralnet
`predictpknn --config /path/to/config.yaml` <br>
* Export a trained PK-neuralnet to NACARTE
`savepknn --config /path/to/config.yaml` <br>
