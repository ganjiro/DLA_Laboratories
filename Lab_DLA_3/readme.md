# Laboratory 3: Deep Reinforcement Learning
This is a repository for the lab 3 of the course deep learning application.

## Exercise 1
In this exercise I refactored and fixed the Francesco Fantechi's implementation of the environment "NavigationGoal" and his q-learning algorith implementation. Those changes have been pushed to the original repository, to allow everyone to work with a standard implementation of the env.

## Exercise 2
To further improve the results made with q-learning I have implemented a version of Double q-learning (van Hasselt, 2010), this technique allows to stabilize the training using an unbiased estimator to predict the q-values of next state for the Bellman residual minimization. 

Here is a graph of the average return of the policy during the training: 

![validation.png](img%2Fvalidation.png)

As we can see the training is still a bit noisy, but it converges. Subsequently, the policy was tested on 1000 runs of the environment reaching the goal 96% of the times, hitting an obstacle 2.5% and not reaching the goal in time 1.5%. 