# Snake AI

A snake game that is controlled by a neural network and trained using a genetic algorithm.

## Technologies Used

- Python 3.11.3
- Pygame 2.4.0
- PyTorch 2.0.1

## Setup and Installation

1. Clone this repository to your local machine.
2. Install the necessary packages:
```bash
pip install pygame pytorch
```
3. Navigate to the repository's directory and run the main file to start the game:
```bash
python main.py
```

## How It Works

The neural network guides the snake's movements. It takes in 14 inputs: 12 for nearby blocks (0=empty, 0.5=snake body, 1=wall) and 2 for the food's relative position (-1=left/up, 1=right/down, 0=same level). The output is the probability of the snake moving in each of the four directions. The game starts with a population of 500 snakes, each with its own neural network. At the end of each generation, the 10 best-performing snakes are selected via a fitness function and are used to reproduce the next generation.

## License

This project is licensed under the terms of the MIT license.
