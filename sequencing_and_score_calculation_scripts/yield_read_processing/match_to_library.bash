usearch -usearch_global IQ_seq.fasta -db Gp2_part_1.fasta -strand plus -id 1.0 -matched IQ_part1.fasta
usearch -usearch_global IQ_part1.fasta -db Gp2_part_2.fasta -strand plus -id 1.0 -matched IQ_part2.fasta
usearch -usearch_global IQ_part2.fasta -db Gp2_part_3.fasta -strand plus -id 1.0 -matched IQ_match.fasta

usearch -usearch_global SH_seq.fasta -db Gp2_part_1.fasta -strand plus -id 1.0 -matched SH_part1.fasta
usearch -usearch_global SH_part1.fasta -db Gp2_part_2.fasta -strand plus -id 1.0 -matched SH_part2.fasta
usearch -usearch_global SH_part2.fasta -db Gp2_part_3.fasta -strand plus -id 1.0 -matched SH_match.fasta
