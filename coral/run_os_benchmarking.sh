# qsub -q "gpu.q" -pe smp 1 -l gpu_mem=40G -l h_rt=336:00:00 -cwd -j yes -o logs/os_benchmarking.log run_os_benchmarking.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
module load cuda
export PYTHONPATH=~/Papers/Onc-PN/coral/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$SGE_GPU

echo "GPU allocation: ${SGE_GPU}"

gemma_model="/wynton/protected/project/shared_models/gemma/gemma-7b-it/"
annot_data_dir='../data/annotated/'
fdata='data_onc_pn_ie.csv'
fout='output_onc_pn_ie.csv'
dir_data='../data/'
dir_out='../output/'

echo "Creating inference data"
python -u dataprocessor/create_inference_data.py \
-annot_data_dir $annot_data_dir \
-fdata $fdata \
-dir_data $dir_data

echo "Running inference. Current model: ${gemma_model}"
python -u benchmarking/open_source_benchmarking.py \
-fdata $fdata \
-fout $fout \
-dir_data $dir_data \
-dir_out $dir_out \
-model_name_or_path "$gemma_model" \
-batch_size 2

echo "Evaluating"
python -u benchmarking/evaluate_model.py -fdata $fdata -fout $fout -dir_data $dir_data -dir_out $dir_out