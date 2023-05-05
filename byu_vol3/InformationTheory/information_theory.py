"""
Information Theory Lab

Name
Section
Date
"""

import numpy as np
import wordle


# Problem 1
def get_guess_result(true_word, guess):
    """
    Returns an array containing the result of a guess, with the return values as follows:
        2 - correct location of the letter
        1 - incorrect location but present in word
        0 - not present in word
    For example, if the true word is "boxed" and the provided guess is "excel", the 
    function should return [0,1,0,2,0].
    
    Arguments:
        true_word (string) - the secret word
        guess (string) - the guess being made
    Returns:
        result (array of integers) - the result of the guess, as described above
    """
    # no magic numbers
    length = 5
    # initialize return list
    colors = [None] * length
    # turn the true word into a list
    secret = list(true_word)

    # turn the guessing word into a dictionary with the letters and their positions
    letters = {}
    for l, letter in enumerate(guess):
        if letter not in letters:
            letters[letter] = []
        letters[letter].append(l)

    index = -1
    for letter in letters:
        update = 0

        # check the color of every possible position of each letter
        while len(letters[letter]) > 0:
            color = 0
            failed = False
            for pos in letters[letter]:
                # check if the letter is even in the true word
                try:
                    index = secret.index(guess[pos])

                # if it is not, break and move on, indicating failure to prevent updating the secrete word
                except ValueError:
                    color = 0
                    update = pos
                    failed = True
                    break

                # if there is an exact match, indicate and move on
                if guess[pos] == secret[pos]:
                    color = 2
                    update = pos
                    break

                # if there is not an exact match, indicate and check other positions
                else:
                    if 1 > color:
                        color = 1
                        update = pos

            # remove the letter from the secret word
            if not failed:
                secret[index] = "_"

            # update the return list and the guess dictionary
            colors[update] = color
            letters[letter].remove(update)

    return colors


# Problem 2
def load_words(filen):
    """
    Loads all the words from the given file, ensuring that they
    are formatted correctly.
    """
    with open(filen, 'r') as file:
        # Get all 5-letter words
        words = [line.strip() for line in file.readlines() if len(line.strip()) == 5]
    return words


def get_all_guess_results(possible_words, allowed_words):
    """
    Calculates the result of making every guess for every possible secret word
    
    Arguments:
        possible_words (list of strings)
            A list of all possible secret words
        allowed_words (list of strings)
            A list of all allowed guesses
    Returns:
        ((n,m,5) ndarray) - the results of each guess for each secret word,
            where n is the number
            of allowed guesses and m is number of possible secret words.
    """
    return np.array([[get_guess_result(x, y) for x in possible_words] for y in allowed_words])


# Problem 3
def compute_highest_entropy(all_guess_results, allowed_words):
    """
    Compute the entropy of each guess.
    
    Arguments:
        all_guess_results ((n,m,5) ndarray) - the output of the function
            from Problem 2, containing the results of each 
            guess for each secret word, where n is the the number
            of allowed guesses and m is number of possible secret words.
        allowed_words (list of strings) - list of the allowed guesses
    Returns:
        (string) The highest-entropy guess
        (int) Index of the highest-entropy guess
    """
    # compute the condensed results (2x2 array)
    condensed = np.dot(all_guess_results, np.power(3, np.arange(5)))

    # iterate over every possible guess
    # count the possible results to find the probability of each result
    # compute the sum of the negative logs of the probabilities to find the entropy of that guess
    entropies = - np.array([np.sum(
        counts / np.sum(counts) * np.log2(counts / np.sum(counts))
    )
        for _, counts in [np.unique(row, return_counts=True)
                          for row in condensed]
    ])

    # return the index of the guess with the highest entropy and the word at that index
    index = np.argmax(entropies)
    return index, allowed_words[index]


# Problem 4
def filter_words(all_guess_results, possible_words, guess_idx, result):
    """
    Create a function that filters the list of possible words after making a guess.
    Since we already computed the result of all guesses for all possible words in 
    Problem 2, we will use this array instead of recomputing the results.
    
    Return a filtered list of possible words that are still possible after
    knowing the result of a guess. Also return a filtered version of the array
    of all guess results that only contains the results for the secret words 
    still possible after making the guess. This array will be used to compute 
    the entropies for making the next guess.
    
    Arguments:
        all_guess_results (3-D ndarray)
            The output of Problem 2, containing the result of making
            any allowed guess for any possible secret word
        possible_words (list of str)
            The list of possible secret words
        guess_idx (int)
            The index of the guess that was made in the list of allowed guesses.
        result (tuple of int)
            The result of the guess
    Returns:
        (list of str) The filtered list of possible secret words
        (3-D ndarray) The filtered array of guess results
    """
    raise NotImplementedError("Problem 4 incomplete.")


# Problem 5
def play_game_naive(game, all_guess_results, possible_words, allowed_words, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of making guesses at random.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)

    raise NotImplementedError("Problem 5 incomplete.")


# Problem 6
def play_game_entropy(game, all_guess_results, possible_words, allowed_words, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of guessing the maximum-entropy guess.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)

    raise NotImplementedError("Problem 6 incomplete.")


# Problem 7
def compare_algorithms(all_guess_results, possible_words, allowed_words, n=20):
    """
    Compare the algorithms created in Problems 5 and 6. Play n games with each
    algorithm. Return the mean number of guesses the algorithms from
    problems 5 and 6 needed to guess the secret word, in that order.
    
    
    Arguments:
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        n (int)
            Number of games to run
    Returns:
        (float) - average number of guesses needed by naive algorithm
        (float) - average number of guesses needed by entropy algorithm
    """
    raise NotImplementedError("Problem 7 incomplete.")


if __name__ == "__main__":
    pass
    # # test problem 1
    # true_words = ["boxed", "apple", "banan", "happy", "eeeee", "eeeee", "asdfg", "asdfg", "aback", "aahed"]
    # guesses =    ["excel", "orang", "banan", "hobby", "abcde", "ebcbe", "sdfga", "qwert", "aahed", "aback"]
    # # true_words = ["eeeee"]
    # # guesses = ["abcde"]
    # expected_results = [
    #     [0, 1, 0, 2, 0],
    #     [0, 0, 1, 0, 0],
    #     [2, 2, 2, 2, 2],
    #     [2, 0, 0, 0, 2],
    #     [0, 0, 0, 0, 2],
    #     [2, 0, 0, 0, 2],
    #     [1, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0],
    #     [2, 1, 0, 0, 0],
    #     [2, 0, 1, 0, 0],
    # ]
    #
    # # Test the function using a for loop
    # for i in range(len(true_words)):
    #     true_word = true_words[i]
    #     guess = guesses[i]
    #     expected_result = expected_results[i]
    #     guessed = get_guess_result(true_word, guess)
    #     success = guessed == expected_result
    #
    #     print(f"Test {i}:", success)
    #     if not success:
    #         print("\tTruth:\t", true_word)
    #         print("\tGuess:\t", guess)
    #         print("\tReturned:\t", guessed)
    #         print("\tExpected:\t",expected_result)

    # # test problem 2
    # secrets = load_words("possible_words.txt")
    # allowed = load_words("allowed_words.txt")
    #
    # np.save("full_words.npy", get_all_guess_results(secrets, allowed))

    # # test problem 3
    # words = np.load("full_words.npy")
    # test_words = words.copy()
    #
    # results = compute_highest_entropy(test_words, load_words("allowed_words.txt"))
    # print(results)

    # test problem 4
