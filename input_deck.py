


'''
    THIS IS THE INPUT DECK :
    Number of loops --> number of monte carlo walks to do
    Number of steps --> number of steps to take in a random walk, a single step can have multiple mutations
    Number of snapshots --> how often to save the data.
    Mutation type -->as of right now two options , 'static' or 'dynamic'.
    Number of mutations --> if static the a constant for number of mutations,
     if dynamic then the number of mutations to start with.
'''


# specify inputs right here
class inputs():
    def __init__(self,nb_loops,nb_steps,mutation_type='static',nb_mutations=1,nb_snapshots=10,Nb_sequences=1000):
        #just make all the rest of the parameters besides nb_sequences
        # have the ability to be initilized to different values
        self.nb_loops=nb_loops
        self.nb_steps=nb_steps
        self.nb_snapshots=nb_snapshots
        self.mutation_type= mutation_type
        self.nb_mutations=nb_mutations
        self.Nb_sequences=Nb_sequences


class names():
    def __init__(self):
        self.nb_mutation_fn='nb_mutations'
        self.pp_fn='percent_pos_average'
        self.nb_and_pp_fn='nb_mutations_and_percent_pos'
        self.min_yield_fn='min_yield'
        self.loops_done_fn='loops_done'
        self.times_fn='times'
        self.cys='cys'