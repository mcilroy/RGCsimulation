import cv2
from numpy  import *
from cellSizes import *
import loaderNumpy
import letter_recog
import time
import gc
#to do:
def main():
    cellPixelSize=3
    print("cellPixelSize:"+str(cellPixelSize))
    isRGC = True
    print("isRGC:"+str(isRGC))
    overlap = .1
    cutoff = 0.15
#############################################################################################################
    #training_Size = 6977 #actual: 60000
    #testing_Size = 24045 #actual: 10000
    training_Size = 6977 #actual: 6977
    testing_Size = 24045 #actual: 24045
    images,labels = loaderNumpy.load_MITFaces(dataset="training",path='D:\\computerVisionDatasets',asbytes=True,selection=slice(0, training_Size))
    testing_images,testing_labels = loaderNumpy.load_MITFaces(dataset="testing",path='D:\\computerVisionDatasets',asbytes=True,selection=slice(0, testing_Size))
    #images, labels = loaderNumpy.load_mnist(dataset='training',path='python-mnist-master\\mnist',asbytes=True,selection=slice(0, training_Size))
    #testing_images, testing_labels = loaderNumpy.load_mnist(dataset='testing',path='python-mnist-master\\mnist',asbytes=True,selection=slice(0, testing_Size))
    blah = images[7].astype(uint8, copy=True)
    #print(blah)
    print(blah.dtype)
    print("done loading training data")
    print("training size: "+str(training_Size))
    print("testing size: "+str(testing_Size))
    imageRows = images[0].shape[0]
    imageCols = images[0].shape[1]
    print("image size height: "+str(imageRows))
    print("image size width: "+str(imageCols))
    space = (0,0,imageRows,imageCols)
    testImage(blah,cutoff,cellPixelSize,space,overlap,isRGC)
##    for y in range(0,int(imageRows)):
##        for x in range(int(imageCols)):
##            images[0][y,x] /= 255.
##    cv2.imshow("original", images[0])
    runModel(imageRows,imageCols,training_Size,testing_Size,images,labels,testing_images,testing_labels,cellPixelSize,isRGC,overlap)
    #runModelHebbian(imageRows,imageCols,training_Size,testing_Size,images,labels,testing_images,testing_labels,cellPixelSize,overlap)
    #runModelUnalteredData(imageRows,imageCols,training_Size,testing_Size,images,labels,testing_images,testing_labels)
#############################################################################################################

##    cutoff = 0.025
    testImgFilename1 = "C:\\Users\\STU\\Google Drive\\courses\\CompNeuro\\project\\grid with onoff.png"
##    testImgFilename2 = "C:\\Users\\STU\\Google Drive\\courses\\CompNeuro\\project\\Aaron_Eckhart_0001.jpg"
##    testImgFilename3 = "C:\\Users\\STU\\Google Drive\\courses\\CompNeuro\\project\\femaleface.jpg"
##
    img1 = loadFile(testImgFilename1,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    print(img1.dtype)
    #print(img1)
    imageRows = img1.shape[0]
    imageCols = img1.shape[1]
    print("image size height: "+str(imageRows))
    print("image size width: "+str(imageCols))
    space = (0,0,imageRows,imageCols)
    #testImage(img1,cutoff,cellPixelSize,space,overlap,isRGC)
##    
##    img2 = loadFile(testImgFilename2,cv2.CV_LOAD_IMAGE_GRAYSCALE)
##    imageRows = img2.shape[0]
##    imageCols = img2.shape[1]
##    space = (0,0,imageRows,imageCols)
##    print("image size height: "+str(imageRows))
##    print("image size width: "+str(imageCols))
##    testImage(img2,cutoff,cellPixelSize,space,overlap,isRGC)
##    
##    img3 = loadFile(testImgFilename3,cv2.CV_LOAD_IMAGE_GRAYSCALE)
##    imageRows = img3.shape[0]
##    imageCols = img3.shape[1]
##    space = (0,0,imageRows,imageCols)
##    print("image size height: "+str(imageRows))
##    print("image size width: "+str(imageCols))
##    testImage(img3,cutoff,cellPixelSize,space,overlap,isRGC)
#############################################################################################################
    print("done")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testImage(image,cutoff,cellPixelSize,space,overlap,isRGC=True):
##    for y in range(0,int(imageRows)):
##        for x in range(len(image[0])):
##            image[y,x] /= 255.
    #cv2.imshow("original", image)
    cells = initializeCells(cellPixelSize,space,isRGC,overlap)
    if isRGC:
        imgColor = drawCircles(image,cells,cutoff)
    else:
        imgColor = drawOrientationLines(image,cells,cutoff)
        ##    for y in range(0,int(imageRows*.5)):
##        for x in range(len(images[0][0])):
##            imgColor[y,x] /= 255.
##            imgColor[y,x][1] /= 255.
##            imgColor[y,x][2] /= 255.
    cv2.imshow("size"+str(cellPixelSize)+" cutoff"+str(cutoff)+ "#:"+str(random.randint(0, 10000)), imgColor)

    
def initializeCells(cellPixelSize,space,isRGC=True,overlap=0.1):
    print("initialize cells...")
    cells=[]
    if isRGC:
        rgcSpec = RGCSpec(cellPixelSize)
        initializeRGCsUniformly(cells,space,rgcSpec,overlap)
    else:
        orientationSpec = []
        for x in range(0,12):
            rgcSpec = RGCSpec(cellPixelSize,True,x)
            orientationSpec.append(rgcSpec)
        initializeOrientationUniformly(cells,0,0,space,orientationSpec,overlap)
    print("num cells:"+str(len(cells)))
    return cells

def runModel(imageRows,imageCols,training_Size,testing_Size,images,labels,testing_images,testing_labels,cellPixelSize=3,isRGC=True,overlap=0.1):
    t1 = time.time()
    space=(0,0,imageRows,imageCols) 
    cells = initializeCells(cellPixelSize,space,isRGC,overlap)
    print("Get Firing Rates for training data...")
    transTraining = zeros((training_Size,len(cells)),dtype=float32)
    for i,image in enumerate(images):
        for j,argc in enumerate(cells):
            fr = argc.GetFiringRate(image)
            if fr > 0.4 :
                transTraining[i,j] = 1.0
            else:
                transTraining[i,j] = 0.0
    t2 = time.time()
    print("Time to calculate training Firing Rates: "+str((t2-t1)/60))

    t1 = time.time()
    print 'training model...'
    Model = letter_recog.SVM
    model = Model()
    print("Model: "+str(Model.__name__))
    model.train(transTraining,labels)
    t2 = time.time()
    print("Time to train model: "+str((t2-t1)/60))

    t1 = time.time()
    train_rate = mean(model.predict(transTraining) == labels)
    t2 = time.time()
    print("Time to test model on training data: "+str((t2-t1)/60))
    
    #delete stuff
    images = zeros(shape=(1,1))
    labels = zeros(shape=(1,1))
    transTraining = zeros(shape=(1,1))

    t1 = time.time()
    print("Get Firing Rates for testing data...")
    transTesting = zeros((testing_Size,len(cells)),dtype=float32)
    for i,image in enumerate(testing_images):
        for j,argc in enumerate(cells):
            fr = argc.GetFiringRate(image)
            transTesting[i,j] = fr
    t2 = time.time()
    print("Time to calculate testing Firing Rates: "+str((t2-t1)/60))
    
    t1 = time.time()
    print 'testing model...'
    test_rate  = mean(model.predict(transTesting) == testing_labels)
    print 'train rate: %f  test rate: %f' % (train_rate*100, test_rate*100)
    t2 = time.time()
    print("Time to test model with testing data: "+str((t2-t1)/60))

def runModelHebbian(imageRows,imageCols,training_Size,testing_Size,images,labels,testing_images,testing_labels,cellPixelSize=3,overlap=0.1):
    isRGC = False
    t1 = time.time()
    space=(0,0,imageRows,imageCols) 
    cells = initializeCells(cellPixelSize,space,isRGC,overlap)
    print("Get Firing Rates for training data...")
    transTraining = zeros((training_Size,12),dtype=float32)
    for i,image in enumerate(images):
        for j,argc in enumerate(cells):
            fr = argc.GetFiringRate(image)
            theangle = 0.0
            for x in range(0,12):
                if argc.angle == theangle :
                    transTraining[i,x] += fr
                    break
                theangle += (pi/2)/6
    t2 = time.time()
    print("Time to calculate training Firing Rates: "+str((t2-t1)/60))

    t1 = time.time()
    print 'training model...'
    Model = letter_recog.SVM
    model = Model()
    print("Model: "+str(Model.__name__))
    model.train(transTraining,labels)
    t2 = time.time()
    print("Time to train model: "+str((t2-t1)/60))

    t1 = time.time()
    train_rate = mean(model.predict(transTraining) == labels)
    t2 = time.time()
    print("Time to test model on training data: "+str((t2-t1)/60))
    
    #delete stuff
    images = zeros(shape=(1,1))
    labels = zeros(shape=(1,1))
    transTraining = zeros(shape=(1,1))

    t1 = time.time()
    print("Get Firing Rates for testing data...")
    transTesting = zeros((testing_Size,12),dtype=float32)
    for i,image in enumerate(testing_images):
        for j,argc in enumerate(cells):
            fr = argc.GetFiringRate(image)
            theangle = 0.0
            for x in range(0,12):
                if argc.angle == theangle :
                    transTesting[i,x] += fr
                    break
                theangle += (pi/2)/6
    t2 = time.time()
    print("Time to calculate testing Firing Rates: "+str((t2-t1)/60))
    
    t1 = time.time()
    print 'testing model...'
    test_rate  = mean(model.predict(transTesting) == testing_labels)
    print 'train rate: %f  test rate: %f' % (train_rate*100, test_rate*100)
    t2 = time.time()
    print("Time to test model with testing data: "+str((t2-t1)/60))

def runModelUnalteredData(imageRows,imageCols,training_Size,testing_Size,images,labels,testing_images,testing_labels):
    t1 = time.time()
    print("NumPixels:"+str(imageRows*imageCols))
    normalTraining = zeros((training_Size,imageRows*imageCols),dtype=float32)
    for i in range(0,len(images)):
        normalTraining[i] = images[i].ravel()
    normalTesting = zeros((testing_Size,imageRows*imageCols),dtype=float32)
    for i in range(0,len(testing_images)):
        normalTesting[i] = testing_images[i].ravel()
    t2 = time.time()
    print("Time to convert array of 2d arrays to array of 1d arrays: "+str((t2-t1)/60))
    
    t1 = time.time()
    print 'training model on unaltered data...'
    Model = letter_recog.SVM
    model = Model()
    print("Model: "+str(Model.__name__))
    model.train(normalTraining,labels)
    print 'testing model on unaltered data...'
    train_rate = mean(model.predict(normalTraining) == labels)
    test_rate  = mean(model.predict(normalTesting) == testing_labels)
    print 'unaltered data. train rate: %f  test rate: %f' % (train_rate*100, test_rate*100)
    t2 = time.time()
    print("Time to train and test model with normal data: "+str((t2-t1)/60))

def initializeOrientationUniformly(rgcs,startRows,startCols,space,orientationSpec,overlap):
    overlap=int(overlap*orientationSpec[0].surround.shape[0])
    if overlap<=0:
        overlap = 1
    print("overlap: "+str(overlap))
    startRows,startCols,imageRows,imageCols = int(space[0]),int(space[1]),int(space[2]),int(space[3])
    for row in range(startRows,imageRows-orientationSpec[0].surround.shape[0],overlap):
        for col in range(startCols,imageCols-orientationSpec[0].surround.shape[1],overlap):
            for x in range(0,len(orientationSpec)):
                rgc = RGC(orientationSpec[x].surround.shape[0]/2.0,row,col,CenterType.ON,orientationSpec[x])
                rgcs.append(rgc)
                rgc = RGC(orientationSpec[x].surround.shape[0]/2.0,row,col,CenterType.OFF,orientationSpec[x])
                rgcs.append(rgc)

def drawOrientationLines(image,rgcs,cutoff):
    imgColor = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for argc in rgcs:
        fr = argc.GetFiringRate(image,isOrientation=True)
        drawOrientationLine(imgColor,argc,fr,cutoff)
    return imgColor

def drawOrientationLine(imgColor,argc,fr,cutoff):
    if fr < cutoff and fr > -cutoff:
        return
    if(fr < 0):
        frScaled = fr*-1
        if argc.centerType == CenterType.ON:
            thecolor = (0,0,frScaled*255)
        else:
            thecolor = (frScaled*255,0,frScaled*255)
    else:
        frScaled = fr
        if argc.centerType == CenterType.ON:
            thecolor = (0,frScaled*255,0)
        else:
            thecolor = (0,frScaled*255,frScaled*255)
    pensize = 1 #default
    ox,oy = argc.col+argc.radius,argc.row+argc.radius
    x1,y1 = argc.col+argc.radius,argc.row
    x2,y2 = argc.col+argc.radius,argc.row+(2*argc.radius)
    points = [(x1,y1),(x2,y2)]
    #if argc.angle != 0:
    #    print("points")
    #    print(points)
    newPoints = []
    for point in points:
        newX = cos(argc.angle*((pi/2)/6)) * (point[0]-ox) - sin(argc.angle*((pi/2)/6)) * (point[1]-oy)+ox
        newY = sin(argc.angle*((pi/2)/6)) * (point[0]-ox) + cos(argc.angle*((pi/2)/6)) * (point[1]-oy)+oy
        newPoint = (newX,newY)
        newPoints.append(newPoint)
    #if argc.angle != 0:
    #    print(newPoints)
    cv2.line(imgColor,(int(newPoints[0][0]),int(newPoints[0][1])),(int(newPoints[1][0]),int(newPoints[1][1])),thecolor,pensize)
    #cv2.line(imgColor,(int(argc.col+argc.radius),int(argc.row),int(argc.col+(2*argc.radius)),int(,thecolor,pensize)
    
def drawLines(image,orients,cutoff):
    imgColor = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for orient in orients:
        frs = 0.0
        for argc in orient.rgcs:
            frs += argc.GetFiringRate(image)
            #drawCircle(imgColor,argc,1,cutoff)

        #print(frs)    
        #frs = frs / (normalization*3)
        #print(frs)
        for blah in orient.rgcs:
            pass
            #print(blah.row,blah.col)
        #print("len of orient:"+str(len(orient.rgcs)))
        drawLine(imgColor,orient.rgcs,frs/3,cutoff)
    return imgColor

def drawCircles(image,rgcs,cutoff):
##    print(image)
    imgColor = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #for y in range(0,int(len(imgColor))):
    #    for x in range(int(len(imgColor))):
    #        print(imgColor[y,x]),
    #    print()
    for argc in rgcs:
        fr = argc.GetFiringRate(image)
        drawCircle(imgColor,argc,fr,cutoff)
    return imgColor

def drawLine(imgColor,argc,fr,cutoff):
    if fr < cutoff and fr > -cutoff:
        return
    if(fr < 0):
        frScaled = fr*-1
        if argc[0].centerType == CenterType.ON:
            thecolor = (0,0,frScaled*255)
        else:
            thecolor = (frScaled*255,0,frScaled*255)
    else:
        frScaled = fr
        if argc[0].centerType == CenterType.ON:
            thecolor = (0,frScaled*255,0)
        else:
            thecolor = (0,frScaled*255,frScaled*255)
    pensize = 1 #default
    #pensize = argc.center.shape[0]
    cv2.line(imgColor, (int(argc[0].col+argc[0].radius),int(argc[0].row+argc[0].radius)),(int(argc[2].col+argc[0].radius),int(argc[2].row+argc[0].radius)),
             thecolor, pensize)

def drawCircle(imgColor,argc,fr,cutoff):
    if fr < cutoff and fr > -cutoff:
        return
    if(fr < 0):
        frScaled = fr*-1
        if argc.centerType == CenterType.ON:
            thecolor = (0,0,frScaled*255)
        else:
            thecolor = (frScaled*255,0,frScaled*255)
    else:
        frScaled = fr
        if argc.centerType == CenterType.ON:
            thecolor = (0,frScaled*255,0)
        else:
            thecolor = (0,frScaled*255,frScaled*255)
    pensize = 1 #default
    #pensize = argc.center.shape[0]
    cv2.circle(imgColor,(int(argc.col+argc.radius),int(argc.row+argc.radius)),int(argc.radius),thecolor,pensize)

def initializeRGCsInALine(rgcs,startRows,startCols,endRows,endCols,rgcSpec):
    row, col = startRows,startCols
    step = 1
    while(row <= endRows or col <= endCols):
        rgc = RGC(rgcSpec.surround.shape[0]/2.0,row,col,CenterType.ON,rgcSpec)
        rgcs.append(rgc)
        rgc = RGC(rgcSpec.surround.shape[0]/2.0,row,col,CenterType.OFF,rgcSpec)
        rgcs.append(rgc)
        if(row <= endRows):
            row += step
        if(col <= endCols):
            col += step

def createRGCOrientationsRandomly(numCells,startRows,startCols,endRows,endCols,rgcSpec):
    orients =[]
    for i in range(numCells):
        row,col = getRandomPos(startRows,startCols,endRows,endCols,rgcSpec.surround.shape[0])
        vert = SimpleOrientationCell("vert")
        hori = SimpleOrientationCell("hori")
        backslash = SimpleOrientationCell("backslash")
        forSlash = SimpleOrientationCell("forSlash")
        orientationSize = rgcSpec.surround.shape[0]/3
        kernel = orientationSize // 2
        for y in range(-kernel,kernel+1):
            for x in range(-kernel,kernel+1):
                rgc = RGC(rgcSpec.surround.shape[0]/2.0,row+(rgcSpec.surround.shape[0]*y),col+(rgcSpec.surround.shape[0]*x),CenterType.ON,rgcSpec)
                if x == 0:
                   vert.rgcs.append(rgc)
                if y == 0:
                    hori.rgcs.append(rgc)
                if (x == 1 and y ==-1) or (x == 0 and y ==0) or (x == -1 and y ==1):
                    forSlash.rgcs.append(rgc)
                if (x == y):
                    backslash.rgcs.append(rgc)
        orients.append(vert)
        orients.append(hori)
        orients.append(forSlash)
        orients.append(backslash)
    return orients

def createRGCOrientationUniformly(startRows,startCols,endRows,endCols,rgcSpec,overlap,numOrientations):
    orients=[]
    

def initializeRGCsRandomly(rgcs,numCells,startRows,startCols,imageRows,imageCols,rgcSpec):
    for x in range(numCells):
        row,col = getRandomPos(startRows,startCols,imageRows,imageCols,rgcSpec.surround.shape[0])
        rgc = RGC(rgcSpec.surround.shape[0]/2.0,row,col,CenterType.ON,rgcSpec)
        rgcs.append(rgc)
        row,col = getRandomPos(startRows,startCols,imageRows,imageCols,rgcSpec.surround.shape[0])
        rgc = RGC(rgcSpec.surround.shape[0]/2.0,row,col,CenterType.OFF,rgcSpec)
        rgcs.append(rgc)

def initializeRGCsUniformly(rgcs,space,rgcSpec,overlap):
    overlap=int(overlap*rgcSpec.surround.shape[0])
    if overlap<=0:
        overlap = 1
    print("Overlap:"+str(overlap))
    startRows,startCols,imageRows,imageCols = int(space[0]),int(space[1]),int(space[2]),int(space[3])
    for row in range(startRows,imageRows-rgcSpec.surround.shape[0],overlap):
        for col in range(startCols,imageCols-rgcSpec.surround.shape[1],overlap):
            rgc = RGC(rgcSpec.surround.shape[0]/2.0,row,col,CenterType.ON,rgcSpec)
            rgcs.append(rgc)
            rgc = RGC(rgcSpec.surround.shape[0]/2.0,row,col,CenterType.OFF,rgcSpec)
            rgcs.append(rgc)
    
def getRandomPos(startRows,startCols,rows,cols,sizeOfCell):
    row = random.randint(startRows, (rows)-sizeOfCell)
    col = random.randint(startCols, (cols)-sizeOfCell)
    return (row,col)

def loadFile(filename,colorType):
        img = cv2.imread(filename,colorType)
        return img

class CenterType:
    ON=0
    OFF=1

class RGC:

    def __init__(self,radius,row,col,centerType,rgcSpec):
        self.radius=radius
        self.col=col
        self.row=row
        self.ratio=rgcSpec.ratio
        self.normalization=rgcSpec.normalization
        self.center=rgcSpec.center
        self.surround=rgcSpec.surround
        self.centerType=centerType
        self.angle = rgcSpec.angle
        
    def GetFiringRate(self,image,isOrientation=False):
        firingRateCenter = 0
        firingRateSurround = 0
        for i in range(len(self.surround)):
            for j in range(len(self.surround[0])):
                if self.centerType == CenterType.ON:
                    firingRateSurround -= self.surround[i][j] * image[i+self.row][j+self.col]
                else:
                    firingRateSurround += self.surround[i][j] * image[i+self.row][j+self.col]
                if not isOrientation:
                    if i>=len(self.center) and i < len(self.center)*2 and j >=len(self.center) and j < len(self.center)*2:
                        if self.centerType == CenterType.ON:
                            firingRateCenter += self.center[i-len(self.center)][j-len(self.center)] * image[i+self.row][j+self.col]
                        else:
                            firingRateCenter -= self.center[i-len(self.center)][j-len(self.center)] * image[i+self.row][j+self.col]
                else:
                        if self.centerType == CenterType.ON:
                            firingRateCenter += self.center[i][j] * image[i+self.row][j+self.col]
                        else:
                            firingRateCenter -= self.center[i][j] * image[i+self.row][j+self.col]
        firingRateSurround *= self.ratio
        return (firingRateCenter+firingRateSurround)/self.normalization

class SimpleOrientationCell():
    rgcs = []
    startCol,startRow = 0,0
    endCol,endRow=0,0
    name = ""
    def __init__(self,name):
        self.name=name
        self.rgcs = []
        startCol,startRow = 0,0
        endCol,endRow =0,0

class Result:
    rgcInput=[]
    actualOutput=[]
    def __init__(self):
        self.rgcInput=[]
        self.actualOutput = []
    def __init__(self,rgcInput,actualOutput):
        self.rgcInput=rgcInput
        self.actualOutput = actualOutput



class RGCSpec():
    ratio =0
    normalization=0
    center=array([])
    surround=array([])
    angle = 0
    def __init__(self,rgcPixelSize,isOrientation=False,angle=0):
        if rgcPixelSize == 3:
            if isOrientation:
                if angle==0:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA0L3()
                elif angle == 1:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA1L3()
                elif angle == 2:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA2L3()
                elif angle == 3:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA3L3()
                elif angle == 4:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA4L3()
                elif angle == 5:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA5L3()
                elif angle == 6:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA6L3()
                elif angle == 7:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA7L3()
                elif angle == 8:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA8L3()
                elif angle == 9:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA9L3()
                elif angle == 10:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA10L3()
                elif angle == 11:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA11L3()
                self.angle = angle
            else:
                self.center,self.surround,self.ratio,self.normalization = createRGCSize3()
        elif rgcPixelSize == 6:
            self.center,self.surround,self.ratio,self.normalization = createRGCSize6()
        elif rgcPixelSize == 9:
            if isOrientation:
                if angle==0:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA0L9()
                elif angle == 1:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA1L9()
                elif angle == 2:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA2L9()
                elif angle == 3:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA3L9()
                elif angle == 4:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA4L9()
                elif angle == 5:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA5L9()
                elif angle == 6:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA6L9()
                elif angle == 7:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA7L9()
                elif angle == 8:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA8L9()
                elif angle == 9:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA9L9()
                elif angle == 10:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA10L9()
                elif angle == 11:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA11L9()
                self.angle = angle
            else:
                self.center,self.surround,self.ratio,self.normalization = createRGCSize9()
        elif rgcPixelSize == 12:
            self.center,self.surround,self.ratio,self.normalization = createRGCSize12()
        elif rgcPixelSize == 15:
            if isOrientation:
                if angle==0:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA0L15()
                elif angle == 1:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA1L15()
                elif angle == 2:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA2L15()
                elif angle == 3:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA3L15()
                elif angle == 4:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA4L15()
                elif angle == 5:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA5L15()
                elif angle == 6:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA6L15()
                elif angle == 7:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA7L15()
                elif angle == 8:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA8L15()
                elif angle == 9:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA9L15()
                elif angle == 10:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA10L15()
                elif angle == 11:
                    self.center,self.surround,self.ratio,self.normalization = createOrientationA11L15()
                self.angle = angle
            else:
                self.center,self.surround,self.ratio,self.normalization = createRGCSize15()

main()

    #############
    #testImgFilename = "C:\\Users\\STU\\Google Drive\\courses\\CompNeuro\\project\\grid with onoff.png"
    #testImgFilename = "C:\\Users\\STU\\Google Drive\\courses\\CompNeuro\\project\\Aaron_Eckhart_0001.jpg"
    #testImgFilename = "C:\\Users\\STU\\Google Drive\\courses\\CompNeuro\\project\\femaleface.jpg"
    #img = loadFile(testImgFilename,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #cv2.imshow("window1", img) 
    #for i in range(1,11,2):
    #    img = cv2.GaussianBlur(img,(i,i),0)
    
    #imgColor = cv2.imread(testImgFilename)
    #img = cv2.cvtColor(imgColor,cv2.COLOR_BGR2GRAY)
    #img = images[0]
    #images = []
    #images.append(img)
    #numCells = 9000
    #numOrientations = 1000
    #initializeRGCsRandomly(rgcs,numCells,imageRows*.25,imageCols*.25,imageRows*.75,imageCols*.75,rgcSpec)
    #initializeRGCsInALine(rgcs,imageRows*.25,imageCols*.25,imageRows*.75,imageCols*.75,rgcSpec)

    
    #imgColor= drawCircles(images,rgcs,results,cutoff)
    
    #orients = createRGCOrientationsRandomly(numOrientations,imageRows*.25,imageCols*.25,imageRows*.75,imageCols*.75,rgcSpec)
    #imgColor = drawLines(images,orients,cutoff)     
    
    
    #cv2.imshow("window2", img)
    
    #initializeOrientationUniformly(rgcs,imageRows*.25,imageCols*.25,imageRows*.75,imageCols*.75,orientationSpec,overlap=0.4)
    #imgColor = drawOrientationLines(images,rgcs,cutoff)
    #cv2.imshow("size"+str(rgcPixelSize)+" cutoff"+str(cutoff), imgColor)
    
    #cv2.moveWindow("size"+str(rgcPixelSize)+" cutoff"+str(cutoff), 0, 0)
