import myappkit

import AI
import Apple
import Grid
import Snake
import SnakeGame

app = myappkit.App(400, 480, 'Snake AI')
app.activities['home'].bgColor = (210, 210, 190)

gameplay = SnakeGame.GameplayActivity()
gameplay.setGrid(Grid.Grid(dim=(14, 14), padding_size=3, padding_value=-2,
                           block_size=20, origin=(442, 50 - 60), padding_hidden=True))


def loadup_snakes():
    for i in range(gameplay.population):
        apple = Apple.Apple()
        apple.setSpawnRandomness(True)
        snek = Snake.Snake(apple, 2)
        snek.setSteps(1)

        ai = AI.GAModel([14, 10, 12, 10, 4])
        snek.linkAI(ai, active=True)

        gameplay.add_snake(snek)


def new_game():
    loadup_snakes()
    app.resize(1300, 800)
    app.setCurrentActivity('gameplay')


def resume():
    gameplay.load_model()
    loadup_snakes()
    gameplay.update_mutation_rate()
    gameplay.mutate(chance=0.9)

    app.resize(1300, 800)
    app.setCurrentActivity('gameplay')


def enableEff():
    gameplay.efficiency_mode = True
    gameplay.frame_rate = 1000


def disableEff():
    gameplay.efficiency_mode = False
    gameplay.frame_rate = 60


new_game_button = myappkit.Button(rect=(0.5 * 400 - 100, 100, 200, 100),
                                  default_color=(255, 0, 0),
                                  hover_color=(200, 0, 0),
                                  focus_color=(100, 0, 0),
                                  text='New Game',
                                  text_color=(255, 255, 255),
                                  text_size=30,
                                  onClick=new_game)

resume_button = myappkit.Button(rect=(0.5 * 400 - 100, 280, 200, 100),
                                default_color=(255, 0, 0),
                                hover_color=(200, 0, 0),
                                focus_color=(100, 0, 0),
                                text='Resume',
                                text_color=(255, 255, 255),
                                text_size=30,
                                onClick=resume)

efficency_button = myappkit.Button(rect=(10, 280, 250, 50),
                                   default_color=(255, 0, 0),
                                   hover_color=(200, 0, 0),
                                   focus_color=(100, 0, 0),
                                   text='Efficiency Mode',
                                   text_color=(255, 255, 255),
                                   text_size=22,
                                   click_action='focus',
                                   onClick=enableEff,
                                   onUnfocus=disableEff)

nightMode_button = myappkit.Button(rect=(501, 0, 200, 50),
                                   border_radii=(0, 0, 2, 0),
                                   default_color=(190, 190, 170),
                                   hover_color=(50, 50, 50),
                                   focus_color=(20, 30, 20),
                                   text='Night Mode',
                                   text_color=(100, 100, 100),
                                   text_size=22,
                                   click_action='focus')


def pause():
    gameplay.paused = True
    gameplay.frame_rate = 10


def unpause():
    gameplay.paused = False
    gameplay.frame_rate = 60


pause_button = myappkit.Button(rect=(701, 0, 200, 50),
                               border_radii=(0, 0, 2, 2),
                               default_color=(190, 190, 170),
                               hover_color=(50, 50, 50),
                               focus_color=(20, 30, 20),
                               text='Pause',
                               text_color=(100, 100, 100),
                               text_size=22,
                               click_action='focus',
                               onClick=pause,
                               onUnfocus=unpause)


def enableNightMode():
    gameplay.bgColor = (0, 10, 0)
    gameplay.color = (100, 100, 100)  # font color
    gameplay.grid.color = (10, 20, 10)
    gameplay.grid.lines_color = (20, 30, 20)
    nightMode_button.width = 2
    AI.GAModel.shown = False
    Apple.Apple.color = (150, 0, 0)
    Snake.Snake.color = (0, 0, 100)


def disableNightMode():
    gameplay.bgColor = (210, 210, 190)
    gameplay.color = (0, 0, 0)  # font color
    gameplay.grid.color = (200, 200, 180)
    gameplay.grid.lines_color = (190, 190, 170)
    nightMode_button.width = 0
    AI.GAModel.shown = True
    Apple.Apple.color = (255, 0, 0)
    Snake.Snake.color = (0, 0, 130)


nightMode_button.onClick = enableNightMode
nightMode_button.onUnfocus = disableNightMode

app.activities['home'].addItem(new_game_button)
app.activities['home'].addItem(resume_button)
# gameplay.addItem(efficency_button)
gameplay.addItem(nightMode_button)
gameplay.addItem(pause_button)

app.addActivity('gameplay', gameplay)

app.run()
