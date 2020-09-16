from input_deck import inputs


C=[inputs(nb_loops=100000,
         nb_steps=5,
         mutation_type='dynamic',
         nb_mutations=10,
         nb_snapshots=20,
         Nb_sequences=10000,
         yield2optimize='Developability'),

    inputs(nb_loops=100000,
         nb_steps=5,
         mutation_type='dynamic',
         nb_mutations=10,
         nb_snapshots=20,
         Nb_sequences=10000,
         yield2optimize='SH_Average_bc'),

    inputs(nb_loops=100000,
         nb_steps=5,
         mutation_type='dynamic',
         nb_mutations=10,
         nb_snapshots=20,
         Nb_sequences=10000,
         yield2optimize='IQ_Average_bc'),






   ]