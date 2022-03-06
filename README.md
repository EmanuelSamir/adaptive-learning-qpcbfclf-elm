# Online Adaptive Compensation for Model Uncertainty Using Extreme Learning Machine-based Control Barrier Functions

### Requirements

Tested on Ubuntu 18.04

1. Python 3.6 
2. Download and uncompress [Carla 0.9.10](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.tar.gz) 
3. Install package requirements 

~~~~bash
pip install -h requirements.txt
~~~~

4. Dowload [extra maps](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.10.tar.gz) and locate it in CARLA_0.9.10/import, then run the following line to install it

   ~~~
   /folder/to/CarlaSimulator/ImportAssets.sh
   ~~~

### Setup for CARLA simulation

1. Run Carla Simulator

   ~~~~~~bash
   /folder/to/CarlaSimulator/CARLA_0.9.10/CarlaUE4.sh
   ~~~~~~

   Also set a low quality level if required

   ~~~~~~bash
   /folder/to/CarlaSimulator/CARLA_0.9.10/CarlaUE4.sh -quality-level=Low
   ~~~~~~

2. Set map from Town06 (the one used in this work)  with the following command

   ~~~
   /folder/to/CarlaSimulator/CARLA_0.9.10/PythonAPI/util/config.py  --map Town06" 
   ~~~



### Usage

There are 5 main files, three used for numerical simulation 

~~~bash
src
*
├── main_dummy.py
├── main_elm.py
├── main_nn.py
*
~~~



and two for CARLA simulation

~~~bash
src
*
├── main_dummy_carla.py
├── main_elm_carla.py
*
~~~

Run any of the codes using the command

~~~
python main_*.py
~~~

