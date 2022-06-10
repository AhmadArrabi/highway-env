import cv2
_x=None
_y=None
def getPixels(event, x, y,flags, params):
     if event == cv2.EVENT_LBUTTONDOWN:
        global _x,_y
        _x = x
        _y = y
        #print(x, ' ', y)
def returnPixels():
   return (_x,_y)