import pygame, sys, random
import numpy as np


class Pong:

    def __init__(self):

        pygame.init()
        self.screen_width = 1280
        self.screen_height = 800

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("My Pong Game!")

        self.clock = pygame.time.Clock()

        self.ball = pygame.Rect(0, 0, 30, 30)
        self.ball.center = (self.screen_width / 2, self.screen_height / 2)

        self.cpu = pygame.Rect(0, 0, 20, 100)
        self.cpu.centery = self.screen_height / 2

        self.player = pygame.Rect(0, 0, 20, 100)
        self.player.midright = (self.screen_width, self.screen_height / 2)

        self.ball_speed_x = 6
        self.ball_speed_y = 6
        self.player_speed = 0
        self.cpu_speed = 6

        self.cpu_points, self.player_points = 0, 0

        self.frameStep(1)

    def possibleActions(self):
        return [0, 2, 3]

    def frameStep(self, action=0):
        # Action 0 means quit
        if action == -1:
            pygame.quit()
            sys.exit()
        pygame.event.pump()

        # go up
        if action == 2:
            self.player_speed = -6
        # go down
        if action == 3:
            self.player_speed = 6

        if action == 0:
            self.player_speed = 0

        # Change the positions of the game objects
        reward, terminal = self.animate_ball()
        self.animate_player()
        self.animate_cpu()

        self.screen.fill('black')
        self.draw()
        pygame.display.update()

        imageData = pygame.surfarray.array3d(pygame.display.get_surface())
        # Return the image data reward and if the game has ended
        return imageData, reward, terminal

    def draw(self):
        # Draw the score
        score_font = pygame.font.Font(None, 100)
        cpu_score_surface = score_font.render(str(self.cpu_points), True, "white")
        player_score_surface = score_font.render(str(self.player_points), True, "white")
        self.screen.blit(cpu_score_surface, (self.screen_width / 4, 20))
        self.screen.blit(player_score_surface, (3 * self.screen_width / 4, 20))

        # Draw the game objects
        pygame.draw.aaline(self.screen, 'white', (self.screen_width / 2, 0),
                           (self.screen_width / 2, self.screen_height))
        pygame.draw.ellipse(self.screen, 'white', self.ball)
        pygame.draw.rect(self.screen, 'white', self.cpu)
        pygame.draw.rect(self.screen, 'white', self.player)

    def reset_ball(self):
        self.ball.x = self.screen_width / 2 - 10
        self.ball.y = random.randint(10, 100)
        self.ball_speed_x *= random.choice([-1, 1])
        self.ball_speed_y *= random.choice([-1, 1])

    def point_won(self, winner):
        if winner == "cpu":
            self.cpu_points += 1
        if winner == "player":
            self.player_points += 1

        self.reset_ball()

    def animate_ball(self):
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        if self.ball.bottom >= self.screen_height or self.ball.top <= 0:
            self.ball_speed_y *= -1

        if self.ball.right >= self.screen_width:
            self.point_won("cpu")
            return -1, True

        if self.ball.left <= 0:
            self.point_won("player")
            return 1, True

        if self.ball.colliderect(self.player):
            self.ball_speed_x *= -1
            self.ball.x = self.player.x - 45
            return 0.1, False

        if self.ball.colliderect(self.cpu):
            self.ball_speed_x *= -1
            self.ball.x = self.cpu.x + 1 + self.cpu.width

        return 0, False

    def animate_player(self):
        self.player.y += self.player_speed

        if self.player.top <= 0:
            self.player.top = 0

        if self.player.bottom >= self.screen_height:
            self.player.bottom = self.screen_height

    def animate_cpu(self):
        self.cpu.y += self.cpu_speed
        odds = random.randint(0, 0)
        if self.ball.centery <= self.cpu.centery and odds == 0:
            self.cpu_speed = -6
        if self.ball.centery >= self.cpu.centery and odds == 0:
            self.cpu_speed = 6

        if self.cpu.top <= 0:
            self.cpu.top = 0
        if self.cpu.bottom >= self.screen_height:
            self.cpu.bottom = self.screen_height

