#!/bin/bash
set -e

ntotal_parts=1

for ((ipart=0; ipart < $ntotal_parts; ipart++)) do
    python sd_maps/load_sdmap_graph.py --collection data_dict_subset_A_train --city_names train_city --total_parts $ntotal_parts --part $ipart
done