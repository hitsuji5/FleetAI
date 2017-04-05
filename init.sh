#!/usr/bin/env bash
wget https://s3.amazonaws.com/misteroda0220/FleetAI/triptime_predictor.pkl -P data/pickle
wget https://s3.amazonaws.com/misteroda0220/FleetAI/nyc_network_graph.pkl -P data/pickle
wget https://s3.amazonaws.com/misteroda0220/FleetAI/zones.csv -P data/table
wget https://s3.amazonaws.com/misteroda0220/FleetAI/trips/trips_2016-05.csv -P data/nyc_taxi

pip install -r requirements.txt
mkdir data/results
#nohup python train_dqn.py > out.log 2> err.log < /dev/null &
