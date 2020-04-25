#!/bin/bash
cd ../../notebooks/lr_tests
export REGION="s0a"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="n13c"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="s13a"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="n13a"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="n13b"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="n13d"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="n13e"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
export REGION="s13b"
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=60000000 --execute Match_LoTSS_params.ipynb
mv Match_LoTSS_params.nbconvert.ipynb Match_LoTSS_params.nbconvert-${REGION}.ipynb
