NUM_THREADS=16
EXP_NAME=MOUSE2HUMAN
PROT_FILE=uniprot-proteome_UP000000589.fasta

cd ../../05_Output/$EXP_NAME

#mkdir results_blast_xml

#./fasta-splitter.pl --n-parts 20 --out-dir splitted sequences.fasta

makeblastdb -dbtype prot -in ../../01_Reference/${EXP_NAME}/${PROT_FILE} -out ${EXP_NAME}_refprot

blastp -query ./${EXP_NAME}_sequences.fasta -db ${EXP_NAME}_refprot -out ${EXP_NAME}_results_blast2ref.xml -outfmt 5 -num_threads $NUM_THREADS 
