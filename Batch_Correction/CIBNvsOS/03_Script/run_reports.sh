#!/bin/bash

jupyter nbconvert --output-dir="../05_Output/01_Reports" --to html --execute 1a_assessing_pca.ipynb
jupyter nbconvert --output-dir="../05_Output/01_Reports" --to html --execute 1b_missing_values.ipynb
jupyter nbconvert --output-dir="../05_Output/01_Reports" --to html --execute 2a_mixed_linear_model_correction.ipynb
jupyter nbconvert --output-dir="../05_Output/01_Reports" --to html --execute 2b_pca_after_correction.ipynb
jupyter nbconvert --output-dir="../05_Output/01_Reports" --to html --execute 3a_limma.ipynb