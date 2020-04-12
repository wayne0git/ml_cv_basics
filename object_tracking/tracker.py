# Ref : https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class Tracker():
    """Object tracker given detected bounding boxes.

    Attributes:
        nextObjectID (int): Next unique object ID.
        objects (OrderedDict): key - Object ID. value - Centroid. (x, y)
        disappeared (OrderedDict):  key - Object ID. 
                                    value - Number of consecutive frames it has been marked as disappeared
        maxDisappeared (int): Number of maximum consecutive disappeared frames allowed.

    """
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # Register new object ID
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Deregister existed object ID when disappeared > maxDisappeared
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        """Update tracker given detected bounding boxes
        Args:           
            rects (list): List of bounding boxes. (x1, y1, x2, y2)
        """
        if len(rects) == 0:
            # Mark existing tracked objects as disappeared
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1

    			# Deregister when disappeared > maxDisappeared
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects
        else:
            # Compute centroids
            inputCentroids = np.zeros((len(rects), 2), dtype="int")

            for (i, (startX, startY, endX, endY)) in enumerate(rects):
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                inputCentroids[i] = (cX, cY)

            # Register when no existed objects
            if len(self.objects) == 0:
                for i in range(0, len(inputCentroids)):
                    self.register(inputCentroids[i])
            # Data association
            else:
                # Data of existed objects
                objectIDs = list(self.objects.keys())
                objectCentroids = list(self.objects.values())
     
    			# Compute centroid distance
                D = dist.cdist(np.array(objectCentroids), inputCentroids)
     
    			# Find the best match between existed objects and new boxes
                # Use sort to make sure match with smallest distance is handled first
                rows = D.min(axis=1).argsort()  # Index for existed object ID
                cols = D.argmin(axis=1)[rows]   # Index for bounding boxes

    			# Keep track of used object ID/box pair 
                # In order to determine if we need to update, register, or deregister
                usedRows = set()
                usedCols = set()
     
    			# Update tracker 
                for (row, col) in zip(rows, cols):
    				# Ignore examined value
                    if row in usedRows or col in usedCols:
                        continue
     
    				# Update existed object
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.disappeared[objectID] = 0
     
    				# Update checked index
                    usedRows.add(row)
                    usedCols.add(col)

    			# Check unused object ID/box
                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)

                # Mark unused existed objects as disappeared
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # Deregister when disappeared > maxDisappeared
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                
                # Register unused input centroid as new objects
                for col in unusedCols:
                    self.register(inputCentroids[col])
 
            return self.objects