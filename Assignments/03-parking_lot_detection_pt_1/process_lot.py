'''This program takes a image of parking lot and 
take the corresponding xml file converts to json. json file
is processed to find the x and y points and finds all the cars 
in a parking lot. And then saves all the cropped cars in a folder 
called spaces and thier histogram is calculated ans stored as csv file..'''

#importing
import cv2 as cv
import numpy as np
#from xmljson import parker, Parker
from xmljson import yahoo
from xml.etree.ElementTree import fromstring
from json import dumps,loads
from matplotlib import pyplot as plt
from scipy.spatial import distance

#taking a single parking lot image from UFPR04 folder.        
img = cv.imread("C:\\Users\\theju\\.spyder-py3\\UFPR04\\UFPR04\\Sunny\\2013-01-29\\a.jpg")
#identifying the single car in parking lot by dimensions.

img = cv.rectangle(img, (900,550), (1000,430), (0,255,0), 4)
#cropping the image of the car.
crop_img = img[430:550,900:1000]

#plotting the image          
plt.imshow(img)
plt.show()
plt.imshow(crop_img)
plt.show()

# images are displayed in a window.
#cv.imshow('Draw01',img)
#cv.imshow("cropped", crop_img)
#cv.waitKey(20)
#cv.destroyAllWindows() 

#This function takes the parking lot image's corresponding xml file
# and converting to a json.
def xmltojson(file_name,out_file=None):
    fp = open(file_name,'r')
    xmldata = fp.read()
    jsond = dumps(yahoo.data(fromstring(xmldata)))
    jsond = loads(jsond)
    spaces = jsond['parking']['space']

    #json file is printed.
    if not out_file is None:
        f = open(out_file,'w')
        f.write(dumps(spaces,indent=4, separators=(',', ': ')))
        f.close()

    for space in spaces:
        print(space['contour'])
        for point in space['contour']['point']:
            print(point)

#giving file names for storing the sliced images.
def filenames(space):
    fname=[]
    file = 'sliced image-'
    file = file + space['id']
    sname = 'spaces/' + file + '.jpg'
    fname.append(sname)
    return fname
#giving file names for sotring histograms csv files.
def histname(space):
    hists=[]
    file='hist-'
    file=file+space['id']
    hname = 'histograms/' + file + '.csv'
    hists.append(hname)
    return hists
    

#function to read the json file and load it.  
def load_spaces(definition_file):
    f = open(definition_file,'r')
    spaces = loads(f.read())
    return spaces

#function to extract the x,y points in a json file.
def extract_points(space):
    points = []
    for i in range(4):
        x = int(space['contour']['point'][i]['x'])
        y = int(space['contour']['point'][i]['y'])
        # appending those points in a list.
        points.append((x,y))
    return points

#function for drawing lines around the cars in the parking lot image.
def draw_parking_space(points,img):
    #different colors are choosen.
    colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0)]
    for i in range(4):
        x1 = points[i][0]
        print(x1)
        y1 = points[i][1]
        print(y1)
        x2 = points[(i+1)%4][0]
        print(x2)
        y2 = points[(i+1)%4][1]
        print(y2)
        cv.line(img, (x1,y1),(x2,y2),colors[i], 4)
       
        # the location of points of lowest and highest values in x and y.
        x1 = points[0][1]
        print ('x1',x1)
        y1 = points[1][0]
        print('y1',y1)
        x2 = points[2][1]
        print('x2',x2)
        y2 = points[3][0]
        print('y2',y2)
        # image is cropped using these ponits.
        crop_img = img[x1:x2,y2:y1]
        plt.imsave(cropped_images[0],crop_img)
        
        # finding the image's histogram.
        
        hsv = cv.cvtColor(crop_img,cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        np.savetxt(hist_images[0], hist, delimiter=',')
        #print(hist)

#function for parllelogram using the points x,y.        
def make_parallelogram(p,type=0):
    """
    Types: 0 = smallest area , 1 = largest area , 2 = avg area
    """
    for i in range(4):
        a = p[i]
        b = p[(i+1) % 4]
        # distance is calculated.
        dst = distance.euclidean(a,b)
        print(dst)
    print()


#main() function.
if __name__=='__main__':
    # xml file path is given.
    filename = 'C:\\Users\\theju\\.spyder-py3\\UFPR04\\UFPR04\\Sunny\\2013-01-29\\pklot.xml'
    #xml to json coneversion and stored in a file pklot_example-json.
    xmltojson(filename,'pklot_example-json.json',)
    #json file given to process the spaces in it.
    definition_file = 'C:\\Users\\theju\\.spyder-py3\\pklot_example-json.json'
    #image file path is given.
    image_file = "C:\\Users\\theju\\.spyder-py3\\UFPR04\\UFPR04\\Sunny\\2013-01-29\\a.jpg"
    spaces = load_spaces(definition_file)
    #image file is stored in img.
    img = cv.imread(image_file)
    for space in spaces:
        #calling the functions.
        cropped_images = filenames(space)
        hist_images=histname(space)
        points = extract_points(space)
        make_parallelogram(points)
        draw_parking_space(points,img)
    # Typical opencv methods to show images in a window.
    #cv.imshow('Draw02',img)
    #stores an image.
    #cv.imwrite('1.jpg',img)
    #cv.waitKey(20)
    

