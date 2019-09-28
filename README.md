# Adversarial Attack Using Genetic Algorithm

This is a repo that accompanies my article about generating adverasarial examples. It consists of:

* `GeneticSolver.py` - general framework for genetic approach rewritten for MNIST dataset
* `ImageGeneticSolver.py` - Same as GeneticSolver, but saves the best candidate so far in verbose mode
* `ga_adv.ipynb` - main notebook with the results described in article
* `ga_adv_robust.ipynb` - bonus - exploration of the idea of multi-task learning. If the attacker knows that you consider the example to be fake, he can still use this information during optimization
* `mnist.pnz` - MNIST dataset for training