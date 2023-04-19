#!/usr/bin/env bash

{
script_start=$(date +%s)
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate $1

notebooks=$(find ../notebooks/. -maxdepth 1 -name '*.ipynb')
n_notebooks=$(find ../notebooks/. -maxdepth 1 -name '*.ipynb' | wc -l | sed 's/^[ \t]*//;s/[ \t]*$//')

echo "*******************"
echo "Executing Notebooks"
echo -e "*******************\n"

echo -e "Script Started At: $(date '+%Y-%m-%d %H:%M:%S')\n"

n=1
for i in $notebooks 
do
  start=$(date +%s)

  echo "---"
  echo "Executing Notebook $n of $n_notebooks"
  echo "Running Notebook: $i" | sed 's/..\/notebooks\/.\///g'
  jupyter nbconvert --inplace --execute $i &>/dev/null
  n=$((n+1))

  end=$(date +%s)
  runtime=$((end-start))
  printf 'Run Time: %02dh:%02dm:%02ds\n' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60))
  echo -e "---\n"
done

script_end=$(date +%s)
script_runtime=$((script_end-script_start))


echo "---"
printf 'Script Total Run Time: %02dh:%02dm:%02ds\n' $((script_runtime/3600)) $((script_runtime%3600/60)) $((script_runtime%60))
echo -e "---\n"


echo "********"
echo "Finished"
echo "********"
} | tee notebook_execution_log.txt
