#!/bin/bash

experiments=(
    # Grid Search CNN
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B3_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B4_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_Adam_B5_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B3_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B4_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/N_SGD_B5_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B3_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B4_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_Adam_B5_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B3_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B4_M3_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M2_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M2_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M2_I8_C4_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M3_F2_C2_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M3_F2_C3_D3.yml
    # # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M3_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/cnn/U_SGD_B5_M3_I8_C4_D3.yml
    
    # Grid Search Umonne
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M3_I8_C2_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M3_F2_C2_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M3_F2_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B3_M2_F2_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M3_F2_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B5_M3_F2_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M3_F2_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B3_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B5_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B4_M3_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M3_I8_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M3_I8_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B4_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B4_M2_F2_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B3_M2_F2_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M3_F2_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M3_I8_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B4_M2_F2_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M2_F2_C4_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B4_M2_F2_C2_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M3_F2_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B5_M2_I8_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M2_I8_C4_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B3_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B4_M3_I8_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B4_M3_I8_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B4_M3_F2_C2_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B5_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M3_F2_C2_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B3_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B4_M3_F2_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B3_M2_F2_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B3_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B3_M2_F2_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B5_M3_F2_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B5_M3_I8_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B4_M3_F2_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M2_I8_C4_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B5_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M2_I8_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B4_M3_I8_C4_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B4_M3_F2_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B5_M3_F2_C4_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B3_M2_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B5_M2_F2_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B5_M3_I8_C3_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_SGD_B5_M3_I8_C2_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B5_M3_I8_C4_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_ADAM_B5_M2_I8_C4_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/U_ADAM_B4_M2_F2_C3_D4.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B3_M2_I8_C2_D3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/N_SGD_B5_M2_I8_C4_D4.yml

    # Grid search umonne v2
    /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D4_S1.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D4_S2.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D4_S3.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D5_S1.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D5_S2.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D5_S3.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D4_S1.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D4_S2.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D4_S3.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D5_S1.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D5_S2.yml
    /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D5_S3.yml

    # New Baseline
    # /home/ir-borq1/experiments/regression/umonne/umonne_sst_co.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_mst_co.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_sst_pe.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_lst_co.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_mst_pe.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_lst_pe.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_sst_co.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_mst_co.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_sst_pe.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_lst_co.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_mst_pe.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_lst_pe.yml
)

# ----------------Comands--------------------------
echo "Enqueue multiples train_model.sh jobs:"

for experiment in "${experiments[@]}"; do
    echo "Experment config file: $experiment"
    sbatch --export=experiment=$experiment submit_train.sh
done
