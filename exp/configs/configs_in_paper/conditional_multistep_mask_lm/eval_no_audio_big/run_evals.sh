CUR_DIR=$(pwd)
FILES_IN_DIR=$(ls $CUR_DIR | grep .yaml)

cd ../../../../launch_scripts

for file in $FILES_IN_DIR;
do
    FILE_PATH=$CUR_DIR/$file
    echo "launching evaluation for $FILE_PATH"
    sbatch --export=CONFIG=$FILE_PATH ./run_eval_cpu.sh
done