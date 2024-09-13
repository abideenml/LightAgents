import os
import sys

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))



from examples.mcts.mct import MCT
from examples.mcts.utilities import seed_answers


question = "Among the $900$ residents of Aimeville, there are $195$ who own a diamond ring, $367$ who own a set of golf clubs, and $562$ who own a garden spade. In addition, each of the $900$ residents owns a bag of candy hearts. There are $437$ residents who own exactly two of these things, and $234$ residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things."
monte_carlo_simulation = MCT(question,seed_answers=seed_answers,iterations=30)
best_answer= monte_carlo_simulation.search()
print(best_answer)