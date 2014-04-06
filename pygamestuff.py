#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      lukea_000
#
# Created:     02/11/2013
# Copyright:   (c) lukea_000 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import sys, pygame
from pygame.locals import *

pygame.init()
size = width, height = 1700, 700
white = 255, 255, 255
screen = pygame.display.set_mode(size)

class Crosshair(object):
    def __init__(self, speed = [1, 1], quadratic = True):
        self.quadratic = quadratic
        self.speed = speed
        self.cross = pygame.image.load('gaussianBlur.png')#pygame.image.load('bmpcrosshair.bmp')
        self.crossrect = self.cross.get_rect()
##        print self.crossrect.center, "is the center"
##        print self.crossrect, "is the rect"
##        print self.crossrect.top, "is the top"
##        print self.crossrect.left, "is the left"
        self.result = []
        self.delay = 20
        self.userWantsToQuit = False
        self.draw()

    def draw(self):
        self.remove()
        #could maybe edit the crossrect directly for smoother motions
        #The Rect object has several virtual attributes which can be used to move and align the Rect:
        #top, left, bottom, right
        #topleft, bottomleft, topright, bottomright
        #midtop, midleft, midbottom, midright
        #center, centerx, centery
        #size, width, height
        #w,h
        screen.blit(self.cross, self.crossrect)
        pygame.display.flip()

    def drawCrossAt(self, coords):
        self.crossrect.center = coords
        self.draw()

    def move(self):
        self.crossrect = self.crossrect.move(self.speed)
        if self.crossrect.left < 0 or self.crossrect.right > width:
            self.speed[0] = -self.speed[0]
        if self.crossrect.top < 0 or self.crossrect.bottom > height:
            self.speed[1] = -self.speed[1]

    def record(self, x, y):
        cx, cy = self.crossrect.centerx, self.crossrect.centery
        lis = [x, y, cx, cy]
        if self.quadratic == True:
            lis.append([cx * cx, cx * cy, cy * cy])
        self.result.append(lis)

    def record(self, inputTuple):
        self.result.append(list(inputTuple)+[self.crossrect.centerx,self.crossrect.centery])

    def write(self):
        fo = open("1700wxoffsetyoffsetxy.csv", "w")
        for line in self.result:
            print line
            result = ""
            for number in line:
                result += str(number) + str(',')
            fo.write(result + "\n")
        fo.close()

    #collects data, returns true if done looping
    def loop(self):
        self.move()
        pygame.time.delay(self.delay)
        self.draw()

    def remove(self):
        screen.fill(white)
        pygame.display.flip()

    def clearEvents(self):
        pygame.event.clear()

    # Blocks the thread while waiting for a click.
    def getClick(self):
        needClick = True
        while needClick:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    self.crossrect.center = pos
                    self.draw()
                    needClick = False
                else:
                    continue

    # Returns True, saves position, and draws the crosshairs if a click has occurred.
    # Returns False if not.
    def pollForClick(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                self.crossrect.center = pos
                self.draw()
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.userWantsToQuit = True
        return False

    def close(self):
        pygame.display.quit()


##ch = Crosshair()
##for i in range(10):
##    pygame.time.delay(100)
##    ch.getClick()


#while 1:
#    pressed = pygame.mouse.get_pressed()
#    if any(pressed):
#        break
#    for event in pygame.event.get():
#        if event.type in (QUIT, pygame.KEYDOWN):
#            break

#    crosshair.draw()
#    pygame.time.delay(10)#miliseconds
#    xoffset = 0
#    yoffset = 0
#    crosshair.record(xoffset, yoffset)
#    crosshair.move()



