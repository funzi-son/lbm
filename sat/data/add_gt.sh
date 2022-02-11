ddir=$1
gt=$2
for file in "$ddir"/*.*
do
  echo $file;
  echo "${file/.cnf/_sat=$gt.cnf}";
  mv "$file" "${file/.cnf/_sat=$gt.cnf}"
done