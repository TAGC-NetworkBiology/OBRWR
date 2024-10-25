NUM_THREADS=16

cd ../05_Output

#mkdir results_blast_xml

#./fasta-splitter.pl --n-parts 20 --out-dir splitted sequences.fasta

makeblastdb -dbtype prot -in ../01_Reference/uniprot-proteome-UP000002254.fasta -out CANLF_refprot

blastp -query ./sequences.fasta -db CANLF_refprot -out results_blast2ref.xml -outfmt 5 -num_threads $NUM_THREADS -
