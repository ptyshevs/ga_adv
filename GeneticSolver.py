import numpy as np

class GeneticSolver:
    def __init__(self, image_size, population_size=100, n_generations=1000, retain_best=0.8, retain_random=0.05, mutate_chance=0.05,
                 verbosity=0, verbose_step=50, random_state=None, warm_start=False, early_stopping=True, patience=20):
        """
        :param population_size: number of individual candidate solutions
        :param n_generations: number of generations
        :param retain_best: percentage of best candidates to select into the next generation
        :param retain_random: probability of selecting sub-optimal candidate into the next generation
        :param mutate_chance: candidate mutation chance
        :param verbosity: level of verbosity (0 - quiet, 1 - evolution information, 2 - spamming like it's 2003)
        :param verbosity_step: number of generations to process before showing the best score
        :param random_state: if specified, initializes seed with this value
        :param warm_start: if True, initial population generation step is omitted, allowing for continuing training
        :param early_stopping: if True, evolution will stop if top-10 candidates are not changing for several generations
        :param patience: number of generations to wait for best solution change when <early_stopping>
        """
        self.image_size = image_size
        self.population_size = population_size
        self.n_generations = n_generations
        self.retain_best = retain_best
        self.retain_random = retain_random
        self.mutate_chance = mutate_chance
        self.verbosity = verbosity
        self.verbosity_step = verbose_step
        self.random_state = random_state
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.patience = patience

        self._population = None

    def solve(self, fitness_fn, n_generations=-1, verbose_step=None):
        """
        :param fitness_fn: function to optimize w.r.t.
        :param n_generations: number of evolution generations. Overrides initialization value if specified
        :return: best gene from the population pool. You can still have access to population and the corresponding scores afterwards
        """
        if verbose_step is None:
            verbose_step = self.verbose_step
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self._population is None or self.warm_start:
            self._population = self._generate_population(self.image_size)
    
        if n_generations != -1:
            self.n_generations = n_generations
    
        scores = np.zeros(len(self._population))
        prev_scores = np.zeros(len(self._population))
        cnt_no_change_in_scores = 0

        for generation in range(self.n_generations):
            self._population, scores = self.evolve(fitness_fn)
            if np.isclose(prev_scores[:10], scores[:10]).all():
                cnt_no_change_in_scores += 1
            else:
                cnt_no_change_in_scores = 0
                prev_scores = scores
            
            if self.verbosity:
                if generation == 0:
                    print("Generation #: best score")
                elif generation == self.n_generations - 1:
                    print("Generation ",generation,": ",scores[0])
                elif generation % verbose_step == 0:
                    print("Generation ",generation,": ",scores[0])
                    
            if np.isclose(scores[:10], 1).any() or (self.early_stopping and cnt_no_change_in_scores >= self.patience):
                if self.verbosity:
                    print("Early stopping on generation ",generation, " with best score ", scores[0])
                break
    
        return self._population[0], scores[0]

    def _generate_population(self, image_size):
        """
        Generating initial population of individual solutions
        :return: initial population as an array
        """
        return np.random.rand(self.population_size, *image_size)

    def evolve(self, fitness_fn):
        """
        Evolution step
        :return: new generation of the same size along with scores of the best retained individuals
        """
        scores = np.array(self.score_population(self._population, fitness_fn))
        
        retain_len = int(len(scores) * self.retain_best)
        sorted_indices = np.argsort(scores)[::-1]
        self._population = [self._population[idx] for idx in sorted_indices]
        best_scores = scores[sorted_indices][:retain_len]
        if self.verbosity > 1:
            print("best scores:", best_scores)
        parents = self._population[:retain_len]
        leftovers = self._population[retain_len:]

        cnt_degenerate = 0
        for gene in leftovers:
            if np.random.rand() < self.retain_random:
                cnt_degenerate += 1
                parents.append(gene)
        if self.verbosity > 1:
            print("# of degenerates left: ", cnt_degenerate)

        cnt_mutations = 0
        for gene in parents[1:]:  # mutate everyone expecting for the best candidate
            if np.random.rand() < self.mutate_chance:
                self.mutate(gene, self.image_size)
                cnt_mutations += 1
        if self.verbosity > 1:
            print("# of mutations: ", cnt_mutations)

        places_left = self.population_size - retain_len
        children = []
        while len(children) < places_left:
            mom_idx, dad_idx = np.random.randint(0, retain_len - 1, 2)
            if mom_idx != dad_idx:
                child1, child2 = self.crossover(parents[mom_idx], parents[dad_idx], self.image_size)
                children.append(child1)
                if len(children) < places_left:
                    children.append(child2)
        if self.verbosity > 1:
            print("# of children: ", len(children))
        parents.extend(children)
        return parents, best_scores

    @classmethod
    def crossover(cls, mom, dad, image_size):
        """
        Take two parents, return two children, interchanging half of the allels of each parent randomly
        """
        # select_mask = np.random.randint(0, 2, size=(20, 20), dtype='bool')
        select_mask = np.random.binomial(1, 0.5, size=image_size).astype('bool')
        child1, child2 = np.copy(mom), np.copy(dad)
        child1[select_mask] = dad[select_mask]
        child2[select_mask] = mom[select_mask]
        return child1, child2

    @classmethod
    def mutate(cls, field, image_size):
        """
        Inplace mutation of the provided field
        """
        a = np.random.binomial(1, 0.1, size=image_size).astype('bool')
        field[a] = np.clip(field[a] + np.random.randn(*field[a].shape) * 0.1, 0, 1)
        return field

    @classmethod
    def score_population(cls, population, fitness_function):
        """
        Apply fitness function for each gene in a population
        :param population: list of candidate solutions (images)
        :return: list/1d-array of scores for each solution
        """
        if type(population) is list:
            population = np.array(population)
        return fitness_function(population)
