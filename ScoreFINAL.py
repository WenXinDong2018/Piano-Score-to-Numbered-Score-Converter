import cv2
import numpy as np
import matplotlib.pyplot as plt
# make an array that store clef locations and elimate repeating locations

class Converter:

    def __init__(self, scorePath):

        self.score = cv2.imread(scorePath)
        self.score_BW = cv2.cvtColor(self.score, cv2.COLOR_BGR2GRAY)
        self.s_w,self.s_h = self.score_BW.shape[::-1]

    def thresholding(self, img, minV, maxV):
        gaus = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
        return gaus
    
    def fillHoles(self, img, size):
        kernel = np.ones((size,size), np.uint8)
        img = cv2.erode(img, kernel, iterations = 1)
        img = cv2.dilate(img, kernel, iterations = 1)
        return img
        

    def filterContourByArea(self, contours,minA, maxA):
        index = len(contours)-1
        while index>=0:
            area = cv2.contourArea(contours[index])
            if area<minA or area>maxA: contours.pop(index)
            index-=1
        return contours
        
    def findRect(self, img):  #split src img into lines. 
        _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #filter contour by area 
        minA = (self.s_w * 0.7) * (self.s_h * 0.2) *0.5
        maxA = self.s_w * self.s_h * 0.9
        contours = self.filterContourByArea(contours, minA, maxA)
        #for each contour, find a bounding rectangle
        rectangles = {}
        for cnt in contours:
            #trim off clef and sign signature
            x,y,w,h = cv2.boundingRect(cnt)
            clefW = int(w*0.08)
            x += clefW  
            w -= clefW
            rectangles[y] = [x,y,w,h//2]
            rectangles[y+1] = [x,y+h//2,w,h//2]
        sortedRectangles = list(dict(sorted(rectangles.items())).values())
        return sortedRectangles

    def structuralElementMorphology(self, img, structure):
        dst = cv2.bitwise_not(img)
        structureElement = cv2.getStructuringElement(cv2.MORPH_RECT, structure)
        dst = cv2.erode(dst, structureElement)
        dst = cv2.dilate(dst, structureElement)
        dst = cv2.bitwise_not(dst)
        return dst
    
    def boundRectangles(self, contours, img, padding):              #bound rect around contour
        rectangles = {}
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            rectangles[x] = [x,y,w,h]
            cv2.rectangle(img,(x-padding,y-padding),(x+w+padding,y+h+padding),(0,255,255),2)
        sortedRectangles = list(dict(sorted(rectangles.items())).values()) 
        return sortedRectangles
            
        
    def extractNotes(self, topLine,trebleBool):

        rows, cols = topLine.shape
        topLine_Original = cv2.cvtColor(topLine,cv2.COLOR_GRAY2RGB)
        horizontal = self.structuralElementMorphology(topLine, (cols // 30, 1))

        vertical = self.structuralElementMorphology(topLine, (1, rows // 25))
        vertical = self.structuralElementMorphology(vertical, (1, rows // 25))

        horizontal_C = cv2.cvtColor(horizontal,cv2.COLOR_GRAY2RGB)
        vertical_C = cv2.cvtColor(vertical,cv2.COLOR_GRAY2RGB)

        #contour out the notes
        _, contourNotes, _ = cv2.findContours(vertical, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contourNotes = self.filterContourByArea(contourNotes, rows*cols//4000, rows*cols//30) #rows*cols//4000, rows*cols//30
        
        #find bounding rectangles for the notes
        noteRects = self.boundRectangles(contourNotes, vertical_C, 4) #returns rect location on src img

        #get rid of the verticle sticks of the notes.
        const = 5
        vertical_without_sticks = self.structuralElementMorphology(vertical, (const, const)) #5,5
        
        #for each note, contour circles.
        circles = []
        for rect in noteRects:
            x,y,w,h = rect
            roi = vertical_without_sticks[0:-1, x:x+w]   
            #padding
            if x-3>0 and w+x+3 < cols:
                roi = vertical_without_sticks[0:-1, x-3:x+w+3]
            
            _, circleContours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            circleContours = self.filterContourByArea(circleContours, 10, 200 )#10,200
            cv2.drawContours(vertical_C[0:-1, x:x+w], circleContours, -1, (255,255,0), 2)  
            
            centerSet = {}
            for cnt in circleContours:
                center,w_h,angle = cv2.minAreaRect(cnt)
                #filter out false positives. 
                
                if w_h[0]< w_h[1]*0.5 or w_h[0]>w_h[1]*2:
                    continue
                
                center = (round(center[0]), round(center[1]))
                
                cv2.circle(vertical_C[0:-1, x:x+w],center,2,(0,255,0),-1)      
                centerSet[center[0]] = center
            
            sortedCircles = list(dict(sorted(centerSet.items())).values()) 
            circles.append(sortedCircles)


        #for each rectangle, contour out five lines
        
        horizontal = cv2.bitwise_not(horizontal)
        lines= []
        prevLines = None
        for rect in noteRects:
            x,y,w,h = rect
            roi = horizontal[:-1, x:x+w]
            _, lineContours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            lineContours = self.filterContourByArea(lineContours, 0, 400)
            
            lineSet = []
           
            for cnt in lineContours:
                rx,ry,rw,rh = cv2.boundingRect(cnt)
                midH = ry + rh//2
                line = [(rx,midH), (rx+rw, midH)]
                lineSet.append(midH)
                cv2.line(vertical_C[0:-1, x:x+w],line[0],line[1],(0,0,255),2)
            lineSet.sort()
            lines.append(lineSet)

        #until now: got circle list and line list inside each rectangle
        #for each circle, compare position against lines, find note
        #find position of lines
        notes = []
        for i in range(len(noteRects)):
            x,y,w,h = noteRects[i]
            rectLines = lines[i]  #[1,2,3,4,5]
            rectCircles = circles[i]  #[(3,1),(3,2),(5,4), (6,8)]
            if rectCircles == []:
                continue
            if len(rectLines)==5:
                prevLines = rectLines

            elif prevLines ==None:
                continue

            elif len(rectLines) ==4:
                if rectLines[0] + 5 > prevLines[0]: #first line is present
                    for i in range(1,4):
                        if rectLines[i] + 5 > prevLines[i] and rectLines[i] - 5 < prevLines[i] :    #current line is present
                            continue
                        else:
                            rectLines.insert(i,prevLines[i])
                            continue
                    if len(rectLines)==4:
                        rectLines.append(rectLines[3]*2 - rectLines[2])
                else:
                    rectLines.insert(0,prevLines[0])
                    

                if len(rectLines)==4:
                    rectLines = prevLines
                    
                prevLines = rectLines
                
            elif len(rectLines)==6:
                if rectLines[0] +2 < prevLines[0]:
                    rectLines = rectLines[1:]
                    prevLines = rectLines
                elif rectLines[5] - 2 > prevLines[4]:
                    rectLines = rectLines[:5]
                    prevLines = rectLines
                else:
                    rectLines = prevLines      
                
            else:
                rectLines = prevLines
              
            #find position of cicles
            prevNoteXPos = -4
            lineSeparation = (rectLines[4] - rectLines[0])/8
            for circle in rectCircles:
            
                #determine line on which note lies  
                location = 0
                if trebleBool:
                    location = round((rectLines[4] - circle[1])/lineSeparation) + 3  #treble clef
                else:
                    location = round((rectLines[4] - circle[1])/lineSeparation) -9   #bass clef
                
                octaveN = location // 7
                noteN = location % 7

                if noteN == 0:
                    noteN = 7
                    
                prevNoteXPos = circle[0]
                #write text in the appropriate space
                textlocation = circle[1] + int(lineSeparation*3.5)
                if textlocation > rows*0.9:
                    textlocation = circle[1] - int(lineSeparation*3.5)
                cv2.putText(topLine_Original,str(noteN),(x+circle[0],textlocation), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2,cv2.LINE_AA)
                cv2.putText(vertical_C,str(noteN),(x+circle[0],textlocation), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2,cv2.LINE_AA)

                if octaveN==1:
                    cv2.circle(topLine_Original,(x+circle[0]+int(lineSeparation*1.5),textlocation-int(lineSeparation*3.5)),2,(0,0,255),-1)
                elif octaveN==-1:
                    cv2.circle(topLine_Original,(x+circle[0]+ int(lineSeparation*1.5),textlocation + int(lineSeparation* 1.5) ),2,(0,0,255),-1)
                                    
        return topLine_Original

    def outImg(self, rectangles, filename):
        newScoreBW1 = cv2.cvtColor(self.score_BW,cv2.COLOR_GRAY2RGB)
        
        for i in range(len(rectangles)):
            x,y,w,h = rectangles[i]
            roi = self.score_BW[y:y+h, x:x+w]
            padBool = False
            pad = 3
            if y-pad>0 and x-pad >0 and x+w+pad< self.s_w and y+h+pad< self.s_h: 
                roi = self.score_BW[y-pad:y+h+pad, x-pad:x+w+pad]
                padBool = True
            trebleBool = False
            if (i%2==0):
                trebleBool = True
            topLine_Original = self.extractNotes(roi, trebleBool)

            if padBool:
                newScoreBW1[y-pad:y+h+pad, x-pad:x+w+pad] = topLine_Original
            else:    
                newScoreBW1[y:y+h, x:x+w] = topLine_Original

        #cv2.imshow("newScoreBW1", newScoreBW1)
        cv2.imwrite(filename, newScoreBW1)

    def main(self):
        #step1: threshold gray image
        self.score_BW = self.thresholding(self.score_BW, 100, 255)
        #step1.5: optional, fill empty holes. 
        #self.score_BW = self.fillHoles(self.score_BW,3)  

        #step2: find rectangles
        rectangles = self.findRect(self.score_BW)
        pad = 3

        #step3:  output img we have until now
        self.outImg(rectangles, "output.jpg")

        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
converter = Converter("test.jpeg")
converter.main()

