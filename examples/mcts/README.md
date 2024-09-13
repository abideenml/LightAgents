## Monte Carlo Tree Simulation


### Overview
Monte Carlo Tree Simulation (MCTS) is a highly effective algorithm for decision-making, widely used in game theory and artificial intelligence. By leveraging random sampling, MCTS explores multiple potential outcomes, guiding agents to make optimal decisions based on probabilistic reasoning.

In this implementation, we start with an initial "seed" answer, which is evaluated and critiqued by a large language model (LLM). The LLM refines the response iteratively, enhancing the quality of the solution through repeated simulations and feedback, ultimately arriving at the best possible outcome.

### Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/itsAdee/Light-Agents
   cd Light-Agents
   ```

2. Install dependencies using Poetry:
   ```sh
   poetry install
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your Together API key:
   ```sh
   TOGETHER_API_KEY=your_api_key_here
   ```

### Running the Script

1. Activate the Poetry shell:
   ```sh
   poetry shell
   ```

2. Run the script:
   ```sh
   python examples/mcts/main.py
   ```

