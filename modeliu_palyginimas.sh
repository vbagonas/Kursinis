#!/bin/bash
#SBATCH -p main
#SBATCH -n8
#SBATCH --cpus-per-task=1
#SBATCH -J embed_dump
#SBATCH --time=06:00:00
#SBATCH -o embed_dump_%j.out
#SBATCH -e embed_dump_%j.err

cd ~/modeliu_lyginimas/
source /scratch/lustre/home/miva8802/modeliu_lyginimas/clone_env/bin/activate

python3 modeliu_palyginimas_su_klonu_uzduotimi.py
