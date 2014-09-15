import math
import cv2 #Import functions from OpenCV
import platform
from numpy  import *
##read images from file
##create ON/OFF and OFF/ON RGCs that fit onto image and store in list (different ways to create the RGCs)
##create list to store results of each image
##for each image
##    create a list to store the ganglion cell firing rates
##    for each ganglion cell
##        add firing rate to the list
##    add the list to the list of results and add the class at the end
##
##feed the list into the neural network as training data
##test on some other data
##report results


def main():
    print(platform.architecture())
    #cv.NamedWindow('a_window', cv.CV_WINDOW_AUTOSIZE)
    #precision = 2.0 # should be odd if odd size of ganglion cells otherwise doesn't matter AND MUST HAVE A DECIMAL AT THE END
    #print(precision)
    sizeOfGanglionCell=3.0 # MUST BE DIVISIBLE BY 3 AND MUST HAVE A DECIMAL AT THE END
    length = 3.0
    width = 1.0
    precision = 10.0
    theangle = 0.0
    for x in range(0,12):
        print("def createOrientationA"+str(x)+"L"+str(int(length))+"():")
        innerSquare = square(length,width,theangle,precision,inner=True)
        printCell(innerSquare,precision,"inner")
        outerSquare = square(length,width,theangle,precision,inner=False)
        printCell(outerSquare,precision,"outer")
        maxInnerSquare = getMax(innerSquare)
        maxOuterSquare = getMax(outerSquare)
        #print("maxInner:"+str(maxInnerSquare))
        #print("maxOuter:"+str(maxOuterSquare))
        ratio = maxInnerSquare/maxOuterSquare
        print("    ratio = "+str(ratio))
        print("    normalization = "+str(maxInnerSquare))
        print("    return (inner,outer,ratio,normalization)")
        #print(maxOuterSquare*ratio)
        theangle += (pi/2)/6
##    innerCircle = circle((sizeOfGanglionCell/3)/2,precision,True)
##    print(len(innerCircle))
##    print(len(innerCircle[0]))
##    biggerCircle = circle(sizeOfGanglionCell/2,precision,False)
##    printCircle(innerCircle,precision)
##    holify(biggerCircle,innerCircle)
##    printCircle(biggerCircle,precision)
##    
##    print("ON Centered RGC")
##    Y = 9+4
##    X  = 9+4
##    image = [[0]*Y for x in range(X)]
##    for y in range(3,6):
##        for x in range(3,6):
##            image[x][y]=255
##    for i in range(len(image)):
##        for j in range(len(image[0])):
##            print("{0:.3f}, ".format(image[i][j])),
##        print''
##    ganglionCellPos = (0,0)
##    maxn = getMax(biggerCircle)
##    
##    minn = getMin(innerCircle)
##    ratio = (minn/maxn)*-1
##    print("Big Circle Max: "+ repr(maxn))
##    print("Small Circle Max: " + repr(minn))
##    print("Ratio: "+repr(ratio))
##    firingRate = getFiringRate(image,ganglionCellPos,innerCircle,biggerCircle,ratio)
##    print("Firing Rate of RGC in image: "+str(firingRate/minn))
    #end of program
    cv2.waitKey(0)

#determine firing rate of a RGC
def getFiringRate(image,ganglionCellPos,innerCircle,biggerCircle,ratio):
    firingRateCenter = 0.0
    firingRateSurround = 0.0
    for i in range(len(biggerCircle)):
        for j in range(len(biggerCircle[0])):
            firingRateSurround += biggerCircle[i][j] * image[i+ganglionCellPos[0]][j+ganglionCellPos[1]]
            if i>=len(innerCircle) and i < len(innerCircle)*2 and j >=len(innerCircle) and j < len(innerCircle)*2:
                firingRateCenter += innerCircle[i-len(innerCircle)][j-len(innerCircle)] * image[i+ganglionCellPos[0]][j+ganglionCellPos[1]]
    #ratio = firingRateCenter/firingRateSurround
    #print(ratio)
    #print(firingRateSurround)
    firingRateSurround *= ratio
    #print(firingRateSurround)  
    #print(firingRateCenter)
    return firingRateCenter+firingRateSurround

def getRatio(biggerCircle,innerCircle):
    firingRateCenter = 0.0
    for i in range(len(innerCircle)):
        for j in range(len(innerCircle[0])):
            firingRateCenter += innerCircle[i][j]*255
    firingRateSurround = 0.0
    for i in range(len(biggerCircle)):
        for j in range(len(biggerCircle[0])):
            firingRateSurround += biggerCircle[i][j]*255
    ratio = firingRateCenter/firingRateSurround
    return ratio

def getMax(biggerCircle):
    firingRateSurround = 0.0
    for i in range(len(biggerCircle)):
        for j in range(len(biggerCircle[0])):
            firingRateSurround += biggerCircle[i][j]*255
    return firingRateSurround

def getMin(innerCircle):
    firingRateCenter = 0.0
    for i in range(len(innerCircle)):
        for j in range(len(innerCircle[0])):
            firingRateCenter += innerCircle[i][j]*255
    return firingRateCenter

#reduce values in center
def holify(biggerCircle,innerCircle):
    print("holify")
    for i in range(len(biggerCircle)):
        for j in range(len(biggerCircle[0])):
            if i>=len(innerCircle) and i < len(innerCircle)*2 and j >=len(innerCircle) and j < len(innerCircle)*2:
                biggerCircle[i][j] = biggerCircle[i][j] - innerCircle[i-len(innerCircle)][j-len(innerCircle)]

#print(quadrants)
def printCircle(quadrants,precision):
    #print("percentages")
    print("([ \\")
    for i in range(len(quadrants)):
        print("["),
        for j in range(len(quadrants[0])):
            if j+1 == len(quadrants[0]):
                print("{0:.3f}".format(quadrants[i][j])),
                print("], \\\n"),
            else:
                print("{0:.3f}, ".format(quadrants[i][j])),
    print("])")

#print(quadrants)
def printCell(quadrants,precision,inner):
    print("    "+str(inner)+" = array([ \\")
    for i in range(len(quadrants)):
        print("["),
        for j in range(len(quadrants[0])):
            if j+1 == len(quadrants[0]):
                print("{0:.3f}".format(quadrants[i][j])),
                print("], \\\n"),
            else:
                print("{0:.3f}, ".format(quadrants[i][j])),
    print("])")

def square(length,width,angle,precision,inner):
    
    pixelLength = length*precision
    pixelWidth = width*precision
    fudge = 0
    point1 = ((pixelLength/2)-(pixelWidth/2),-fudge)
    point2 = ((pixelLength/2)-(pixelWidth/2),pixelLength+fudge)
    point3 = ((pixelLength/2)+(pixelWidth/2),pixelLength+fudge)
    point4 = ((pixelLength/2)+(pixelWidth/2),-fudge)
    points = [point1,point2,point3,point4]

    points = rotate(points,pixelLength,angle)
    mask = zeros(shape=(pixelLength,pixelLength))
    quadrants = [[0.0]*int(length) for x in range(int(length))]
    for pixelY in range(int(pixelLength)):
        for pixelX in range(int(pixelLength)):
            if inner==True:
                if inRect(pixelX+0.5,pixelY+0.5,points):
                    if ((pixelX+0.5)-(pixelLength/2))**2 + ((pixelY+0.5)-(pixelLength/2))**2 <= (pixelLength/2)**2:
                        mask[pixelY,pixelX] = random.uniform(40, 230)
                        quadrantCoord = quadrant(pixelX,pixelY,pixelLength,length)
                        row,col = quadrantCoord
                        quadrants[row][col] += 1
            else:
                if not inRect(pixelX+0.5,pixelY+0.5,points):
                    if ((pixelX+0.5)-(pixelLength/2))**2 + ((pixelY+0.5)-(pixelLength/2))**2 <= (pixelLength/2)**2:
                        mask[pixelY,pixelX] = random.uniform(40, 230)
                        quadrantCoord = quadrant(pixelX,pixelY,pixelLength,length)
                        row,col = quadrantCoord
                        quadrants[row][col] += 1
    cv2.imshow("window"+str(angle)+str(inner), mask)
    return quadrants

def rotate(points,pixelLength,angle):
    ox = pixelLength/2
    oy = pixelLength/2
    newPoints = []
    for point in points:
        newX = cos(angle) * (point[0]-ox) - sin(angle) * (point[1]-oy)+ox
        newY = sin(angle) * (point[0]-ox) + cos(angle) * (point[1]-oy)+oy
        newPoint = (newX,newY)
        newPoints.append(newPoint)
    return newPoints
def inRect(pixelX,pixelY,points):
    inRect = True
    for i in range(0,len(points)):
        if i+1 == len(points):
            X1,Y1 = points[i]
            X2,Y2 = points[0]
        else:
            X1,Y1 = points[i]
            X2,Y2 = points[i+1]
        A = -1*(Y2 - Y1)
        B = X2 - X1
        C = -1*(A * X1 + B * Y1)
        D = A * pixelX + B * pixelY + C
        #print(D)
        if D > 0: # if D > 0 the point is on the left side
            inRect = False
    return inRect
        
def circle(quadrantRadius,precision,isOn):
    quadrantDiameter = quadrantRadius*2
    pixelRadius = quadrantRadius*precision
    pixelDiameter = pixelRadius*2
    quadrantsSize = quadrantDiameter**2
    
    img = zeros(shape=(pixelDiameter,pixelDiameter))
    
    quadrants = [[0.0]*int(quadrantDiameter) for x in range(int(quadrantDiameter))]
    for pixelY in range(int(pixelDiameter)):
        for pixelX in range(int(pixelDiameter)):
            #calculate if pixel is colliding with circle
            #check if which quadrant it is in
            #add to the correct quadrant
            if ((pixelX+0.5)-pixelRadius)**2 + ((pixelY+0.5)-pixelRadius)**2 <= pixelRadius**2:
                img[pixelY,pixelX] = random.uniform(40, 230)
                quadrantCoord = quadrant(pixelX,pixelY,pixelDiameter,quadrantDiameter)
                row,col = quadrantCoord
                quadrants[row][col]+=1
    #for i in range(len(quadrants)):
    #    for j in range(len(quadrants[0])):
            #print(quadrants[i][j])
            #print(quadrants[i][j]/(precision**2))
            #quadrants[i][j]=quadrants[i][j]/(precision**2)
    #        if isOn != True:
    #           pass
                #print(quadrants[i][j])
                #quadrants[i][j] *= -1
                #print(quadrants[i][j])
    cv2.imshow("window"+str(random.uniform(0,100)), img)
    return quadrants
        
def quadrant(pixelX,pixelY,pixelDiameter,quadrantDiameter):
    x1,y1=0,0
    stepSize = int(pixelDiameter/quadrantDiameter)
    x2,y2 = stepSize,stepSize
    quadrantNumberRow = 0
    quadrantNumberCol = 0
    while(True):
        if pixelX >= x1 and pixelX < x2 and pixelY >= y1 and pixelY < y2:
            return (quadrantNumberRow,quadrantNumberCol)
        
        if x2 % int(pixelDiameter) == 0:
            x1 = 0
            x2 = stepSize
            y1 += stepSize
            y2 += stepSize
            quadrantNumberCol = 0
            quadrantNumberRow += 1
        else:
            x1 += stepSize
            x2 += stepSize
            quadrantNumberCol += 1
        
main()


##    #create an image
##    #add some shape to image
##    #for each ganglion cell
##    #   check the light at each 
##    #print max firing rate of ON centered GRF, light in center
##    #1*1*255+4*0.5*255+4*0.25*255
##    firingRateCenter = 0
##    for i in range(len(innerCircle)):
##        for j in range(len(innerCircle[0])):
##            firingRateCenter += (innerCircle[i][j]/(precision**2))*255
##    print("Firing rate of fully stimulated center:{0:.3f}".format(firingRateCenter))
##    
##    #print min firing rate of ON centered GRF, light on surround
##    firingRateSurround = 0
##    for i in range(len(biggerCircle)):
##        for j in range(len(biggerCircle[0])):
##            firingRateSurround += (biggerCircle[i][j]/(precision**2))*255
##    ratio = firingRateCenter/firingRateSurround
##    firingRateSurround *= ratio
##    print("Firing rate of fully stimulated surround:{0:.3f}".format(firingRateSurround))
##    
##    
##    #print firing rate when light on both center and surround
##    print("Both center and surround are stimulated:{0:.3f}".format(firingRateCenter-firingRateSurround))
##    #print firing rate when no light on both center and surround
##    print("Neither center or surround are stimulated:{0:.3f}".format(0))
