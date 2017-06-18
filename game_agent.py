"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

m = {}
moves = {}

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

class longestPathTimeout(SearchTimeout):
    def __init__(self, message):
        self.message = message

class openAreaTimeout(SearchTimeout):
    def __init__(self, message):
        self.message = message

class legalMovesTimeout(SearchTimeout):
    def __init__(self, message):
        self.message = message

def init_moves_by_position(game,loc) :
    """Generate the list of possible moves for an L-shaped motion (like a
    knight in chess restricted by the board).
    """

    global moves
    r, c = loc
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    moves[loc] = [(r + dr, c + dc) for dr, dc in directions if ((r + dr) in range(game.width) and (c + dc) in range(game.height))  ]
    return moves[loc]

def get_moves_for_position(game, loc):

    if loc == game.NOT_MOVED:
        return game.get_blank_spaces()

    moves_for_position =moves.get(loc)
    if not moves_for_position:
        moves_for_position = init_moves_by_position(game,loc)


    valid_moves = [m for m in moves_for_position if game._board_state[m[0] + m[1] * game.height] == game.BLANK]
    random.shuffle(valid_moves)
    return valid_moves

def init_number_moves_by_position(game):
    global m
    for w in range(game.width):
        for h in range(game.height):
            if (w,h) in [(0,0), (0,game.width-1),(game.height-1,0),(game.height-1,game.width-1)]:
                m[(w , h)] = 2
            elif w == 0 or h == 0 or w == game.width-1 or h == game.height-1  :
                m[(w , h)] = 3.6
            elif w == 1 or h == 1 or w == game.width-2 or h == game.height-2  :
                m[(w , h)] = 6
            else:
                m[(w , h)] = 8


def longest_path( game, depth, player) :
    """count max number of moves you could make if the board froze right now, assume other player doesn't move"""

    legal_moves = game.get_legal_moves(player)

    depth += 1
    if depth > 20 or not legal_moves:
        return depth

    v = max([(longest_path(game.forecast_move(m), depth, player)) for m in legal_moves])
    return v

def get_open_area(game, player):

    """Return all empty spaces in the area sorrounding the player"""
    initial_position = game.get_player_location(player)
    to_examine = [initial_position]
    closed = [initial_position]
    accessible = []

    while (to_examine.__len__() > 0):
        space = to_examine.pop()
        """save empty spaces next to this space"""
        sorrounding_spaces = [m for m in get_moves_for_position(game,space) ]
        for successor in  sorrounding_spaces:
           """skip the spaces already seen"""
           if successor in closed:
               continue

           """mark this space as seen"""
           closed.append(successor)

           """if it is empty mark it as accessible and set it up for expansion"""
           if  successor in game.get_blank_spaces():
               accessible.append(successor)
               to_examine.append(successor)



    return accessible

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player) :
        return float("-inf")

    if game.is_winner(player)  :
        return float("inf")

    """apply different strategy in the initial, middle or near end of game"""
    blank_spaces = len(game.get_blank_spaces())
    total_spaces = game.width * game.height
    remaining_game = blank_spaces / total_spaces

    own_legal_moves = get_moves_for_position(game, game.get_player_location(player))
    opp_legal_moves = get_moves_for_position(game, game.get_player_location(game.get_opponent(player)))
    own_moves = len(own_legal_moves)
    opp_moves = len(opp_legal_moves)
    diff_legal_moves = float(own_moves - opp_moves)

    """initial game: """
    if 0.4 < remaining_game  :

        #if player.time_left < player.TIMER_THRESHOLD:
        #    raise legalMovesTimeout("legalMovesTimeout")
        #    print("ignored timeout")
        return diff_legal_moves

    """middle game"""
    if 0.2 < remaining_game :

        my_open_area = get_open_area(game,player)
        opponent_open_area = get_open_area(game,game.get_opponent(player))
        my_area_len = len(my_open_area)
        opp_area_len = len(opponent_open_area)

        #if player.time_left < player.TIMER_THRESHOLD:
        #    raise openAreaTimeout("openAreaTimeout at blank spaces " + str(blank_spaces) + "and total spaces " + str(total_spaces))
        #    print("ignored timeout")

        if my_area_len == 0 :
            return float("-inf")

        if opp_area_len == 0 :
            return float("inf")

        """ if each player is in a separate area,  the player with the bigger area wins"""
        intersection =  [space for space in my_open_area if space in opponent_open_area]
        if len(intersection) == 0:
            if my_area_len == opp_area_len:
                if game.active_player == player:
                    return float("-inf")
                else:
                    return float("inf")
            return float("inf") if my_area_len > opp_area_len else  float("-inf")


        return diff_legal_moves

    """near the end of game"""
    if  0 <= remaining_game :
        #print("E " + str(percent_time_remaining(player)))
        depth = 0
        my_longest_path  =  0 if not own_moves else max([(longest_path(game.forecast_move(m), depth, player)) for m in own_legal_moves])
        opp_longest_path  =  0 if not opp_moves else max([(longest_path(game.forecast_move(m), depth, game.get_opponent(player))) for m in opp_legal_moves])
        #if player.time_left()  < player.TIMER_THRESHOLD:
        #    raise longestPathTimeout("longestPathTimeout")
        #    print("ignored timeout")
        return my_longest_path - opp_longest_path

    return diff_legal_moves

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2 * opp_moves)

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    #own_moves = len(get_moves_for_position(game, game.get_player_location(player)))
    #opp_moves = len(get_moves_for_position(game, game.get_player_location(game.get_opponent(player))))
    blank_spaces = len(game.get_blank_spaces())

    return float((own_moves - opp_moves)/(blank_spaces +1))


def custom_score_4(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if not m:
        init_number_moves_by_position(game)

    diff = m[game.get_player_location(player)]  -  m[game.get_player_location(game.get_opponent(player))]

    blank_spaces = len(game.get_blank_spaces())
    total_spaces = game.width * game.height
    remaining_game = 100* blank_spaces / total_spaces

    return float(remaining_game - diff)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=1, score_fn=custom_score, timeout=10.0):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout as err:
            #print("Search Timeout: {0} ".format(err))
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            print("ignored timeout")

        legal_moves = game.get_legal_moves(game.active_player)

        if not legal_moves:
            return (-1, -1)

        _, move = max([(self.min_value(game.forecast_move(m), depth), m) for m in legal_moves])

        return move


    def max_value(self, game, depth) :

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            print("ignored timeout")

        legal_moves = game.get_legal_moves(game.active_player)

        depth -= 1
        if depth == 0 or not legal_moves:
            v =  self.score(game, self )
            return v


        v,m = max([(self.min_value(game.forecast_move(m), depth), m) for m in legal_moves])
        return v

    def min_value(self, game, depth):
        """Helper function of min_player

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : hashable
            One of the objects registered by the game object as a valid player.
            (i.e., `player` should be either game.__player_1__ or
            game.__player_2__).

        Returns
        ----------
        float
            The min value of the current game state
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            print("ignored timeout")

        legal_moves = game.get_legal_moves(game.active_player)

        depth -= 1
        if depth == 0 or not legal_moves:
            v =  self.score(game, self )
            return v


        v,m = min([(self.max_value(game.forecast_move(m), depth), m) for m in legal_moves])

        return v


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        depth = 1


        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while True :

                m = self.alphabeta(game,depth)
                if m:
                   best_move = m
                depth += 1
        except SearchTimeout:
            return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration

        return best_move


    def alphabeta(self, game, depth):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            print("ignored timeout")

        legal_moves = game.get_legal_moves(game.active_player)

        if not legal_moves:
            return (-1, -1)


        alpha = float("-inf")
        beta = float("inf")
        move = None
        for m in legal_moves:
            value = self.min_value(game.forecast_move(m),depth, alpha, beta)
            if value > alpha:
                  alpha = value
                  move = m

        return  move

    def max_value(self, game, depth, alpha, beta) :

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            print("ignored timeout")

        legal_moves = game.get_legal_moves(game.active_player)

        depth -= 1

        if not legal_moves or depth == 0:
            v =  self.score(game, self )
            return v

        v = float("-inf")

        for m in legal_moves:
            v =max(v, self.min_value(game.forecast_move(m), depth, alpha=alpha, beta=beta) )
            if v >= beta:
                return v
            alpha = max(alpha,v)


        return v

    def min_value(self, game, depth, alpha, beta):
        """Helper function of min_player

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : hashable
            One of the objects registered by the game object as a valid player.
            (i.e., `player` should be either game.__player_1__ or
            game.__player_2__).

        Returns
        ----------
        float
            The min value of the current game state
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            print("ignored timeout")

        legal_moves = game.get_legal_moves(game.active_player)


        depth -= 1

        if not legal_moves or depth == 0:
            v =  self.score(game, self )
            return v

        "---> legal moves for min: "  + str(legal_moves)
        v = float("inf")

        for m in legal_moves:
            v = min(v, self.max_value(game.forecast_move(m), depth, alpha=alpha, beta=beta) )
            if v <= alpha:
                return v
            beta = min(beta,v)

        return v
