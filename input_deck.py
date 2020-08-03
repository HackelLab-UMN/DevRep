


'''THIS IS THE INPUT DECK :
    Number of loops --> number of monte carlo walks to do
    Number of steps --> number of steps to take in a random walk, a single step can have multiple mutations
    Number of snapshots --> how often to save the data.
    Mutation type -->as of right now two options , 'static' or 'dynamic'.
    Number of mutations --> if static the a constant for number of mutations,
     if dynamic then the number of mutations to start with.
'''

# specify inputs right here
class inputs():
    def __init__(self):
        #just make all the rest of the parameters besides nb_sequences
        # have the ability to be initilized to different values
        self.nb_loops=10000
        self.nb_steps=5
        self.nb_snapshots=10
        self.mutation_type= 'dynamic'
        self.nb_mutations=10
        self.Nb_sequences=10000
