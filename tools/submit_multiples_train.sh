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
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D4_S1.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D4_S2.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D4_S3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D5_S1.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D5_S2.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGD_B5_M3_I8_C3_D5_S3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D4_S1.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D4_S2.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D4_S3.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D5_S1.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D5_S2.yml
    # /home/ir-borq1/experiments/grid_search/umonne/V_SGDNM_B5_M3_I8_C3_D5_S3.yml

    # Smoothing
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l21000_cut.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l21000_full.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l2100_cut.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l2100_full.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l210_cut.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l210_full.yml
    
    #/home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l210_cut_adam.yml
    #/home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l210_full_adam.yml

    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l25_full_adam.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_l25_full_sgd.yml
    
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_zero_cut_sgd.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_zero_full_sgd.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_zero_cut_adam.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_zero_full_adam.yml

    #/home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_e1000_full.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_e1000_cut.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_e100_cut.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_e100_full.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_e10_cut.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_e10_full.yml

    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_ne1000_cut.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_ne1000_full.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_ne100_cut.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_ne100_full.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_ne10_cut.yml
    # /home/ir-borq1/experiments/regression/smoothing_experiments/umonne_lst_ne10_full.yml

    # New Baseline
    # CUT
    # /home/ir-borq1/experiments/regression/umonne/umonne_sst_co.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_mst_co.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_lst_co.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_sst_co.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_mst_co.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_lst_co.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_sst_pe.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_mst_pe.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_lst_pe.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_sst_pe.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_mst_pe.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_lst_pe.yml

    # FULL
    /home/ir-borq1/experiments/regression/umonne/umonne_sst_pe_full.yml
    /home/ir-borq1/experiments/regression/umonne/umonne_mst_pe_full.yml
    # /home/ir-borq1/experiments/regression/umonne/umonne_lst_pe_full.yml
    /home/ir-borq1/experiments/regression/cnn/cnn_sst_pe_full.yml
    /home/ir-borq1/experiments/regression/cnn/cnn_mst_pe_full.yml
    # /home/ir-borq1/experiments/regression/cnn/cnn_lst_pe_full.yml
    /home/ir-borq1/experiments/regression/umonne/umonne_sst_pe_big_full.yml
    /home/ir-borq1/experiments/regression/umonne/umonne_mst_pe_big_full.yml
    /home/ir-borq1/experiments/regression/umonne/umonne_lst_pe_big_full.yml
)

# ----------------Comands--------------------------
echo "Enqueue multiples train_model.sh jobs:"

for experiment in "${experiments[@]}"; do
    echo "Experment config file: $experiment"
    sbatch --export=experiment=$experiment submit_train.sh
done
