# Laboratory 2: Large Language Models
This is a repository for the lab 2 of the course deep learning application

## Exercise 1
In the first exercise I have trained a small autoregressive GPT to generate text in the style of Dante Alighieri.
I have reduced the number of iteration of the initial model because of an overfitting problem.

Here is an example of generated text:


        Quivi fummil diserchio, e tutto sgrande,
        volse che 'ntese contra me altri mentimo,
        che Luchitero a mori` che 'l per mi linge!
      
        E beno altre re lungo ferb:
        quel che de' l'altro fei, a pur l'osella?
        Ere: <<Al ventere, a' colui e`?

We can see that the style remembers a lot of Dante Alighieri, but the grammar is not completely correct and it does not make any sense. 

## Exercise 2
In this exercise I had to familiarize with the Hugging Face transformer library.

Here are some text I generated from GPT2 starting with a Fallout citation "War never changes" with different temperatures:
- Temperature 0:

        War never changes.
        
        The only thing that changes is the way the game is played.
        
        The game is played by a single player.

        The game is played by a single player.
        
        The game is played by a single player.
        
        The game is played by a single player.
        
        The
- Temperature 0.4:

        War never changes. The problem is that the game has changed so much that it's hard to keep up with the pace of the game.

        The first game is a huge disappointment. The first game was a big disappointment. The first game was a big disappointment. The first game was a big disappointment. The first game was

- Temperature 0.9:

        War never changes, and neither will it ever. The future of the Internet is, hopefully, all about what users can do with it.

        I'm going to leave you with an open letter written by a lot of other bloggers and activists that has been making my life miserable while writing about the future of the Internet and the

We can notice how the model starts to generate always the same sentence when it has low temperature, on the other hand with higher temperature starts to create something that makes more sense, but still not fully satisfying.   

## Exercise 3
In this exercise I had to train a retrival model starting with a pretrained LLM.

To represent a sentence I decided to use mean pooling, averaging all token embeddings.
I built a simple MLP to project the representation of the query and the corpus into a 128 dim space, then I compute the dot similarity in that space. 
I trained the MLP with the cross entropy loss considering each corpus as class, the similarity between each corpus and the query as logit and the corpus relative to a query as positive label.
The results are analyzed in terms of mean reciprocal rank.

This is the trend of the training loss and the test MRR during the training:

![loss.png](img%2Floss.png)
![mrr.png](img%2Fmrr.png)


