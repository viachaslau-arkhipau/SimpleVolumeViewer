import numpy as np

class DataTypeError(Exception):
    pass

class AxesError(Exception):
    pass

class EmptyVolumeError(Exception):
    pass

class DirectionsNameError(Exception):
    pass

class IndexLimitsNameError(Exception):
    pass


class VolumeStorage:

    StandartAxisDirections = {
        'left':      np.array([ 1.,  0.,  0.]),
        'right':     np.array([-1.,  0.,  0.]),
        'posterior': np.array([ 0.,  1.,  0.]),
        'anterior':  np.array([ 0., -1.,  0.]),
        'superior':  np.array([ 0.,  0.,  1.]),
        'inferior':  np.array([ 0.,  0., -1.])
    }
    
    StandartAxisNames = {
        'left':      'frontal',
        'right':     'frontal',
        'posterior': 'saggital',
        'anterior':  'saggital',
        'superior':  'vertical',
        'inferior':  'vertical'
    }
    
    class VolumeView:
        
        def __init__(self, storage, viewDirection, upDirection, verticalFlip = True):
            
            if not (viewDirection in VolumeStorage.StandartAxisNames.keys() and 
                    upDirection in VolumeStorage.StandartAxisNames.keys()):
                raise DirectionsNameError("The valid direction names are \'left\' or \'right\' " + 
                                              "(for frontal axis), \'superior\' or \'inferior\' " + 
                                              "(for saggital axis) and \'posterior\' or \'anterior\' " + 
                                              "(for vertical axis) only. Ex.: \'left-posterior-superior\'.")
                
            if VolumeStorage.StandartAxisNames[viewDirection] == \
                    VolumeStorage.StandartAxisNames[upDirection]:
                raise DirectionsNameError("Expected view and up directions along the different axes, but " +
                                         "detected both ones along " + 
                                          VolumeStorage.StandartAxisNames[viewDirection] + " axis.")
                
                
            self.viewDirection = VolumeStorage.StandartAxisDirections[viewDirection.lower()]
            self.upDirection   = VolumeStorage.StandartAxisDirections[upDirection.lower()]
            
            self.verticalFlip = verticalFlip
            
            self.storage = storage
            
            # AXES
            self.viewAxisIndex  = np.argmax(np.abs(np.sum(self.storage.axes * self.viewDirection, 1)))
            self.upAxisIndex    = np.argmax(np.abs(np.sum(self.storage.axes * self.upDirection, 1)))
            self.rightAxisIndex = [i for i in range(3) if not i in [self.viewAxisIndex, self.upAxisIndex]][0]
            
            self.rightDirection = np.cross(self.viewDirection, self.upDirection)
                
            self.tranposeFlag = self.upAxisIndex > self.rightAxisIndex
                
            if self.verticalFlip:
                self.upDirection = -self.upDirection
                
            vStep = -1 if np.sum(self.upDirection * self.storage.axes[self.upAxisIndex]) < 0 else 1
            hStep = -1 if np.sum(self.rightDirection * self.storage.axes[self.rightAxisIndex]) < 0 else 1
                
            self.slice = np.s_[::vStep, ::hStep]

            self.viewName = viewDirection
            self.upName = upDirection
            
            
            # LIMITS AND INDEX
            self.minIndex = 0
            self.maxIndex = self.storage.data.shape[self.viewAxisIndex] - 1
            self.setCurrentIndex(self.maxIndex // 2)
            
        def transformImage(self, img):
            return (img.T if self.tranposeFlag else img)[self.slice].copy() 
            
        def getIndexLimits(self):
            return self.minIndex, self.maxIndex
        
        def setCurrentIndex(self, ind):
            if ind < self.minIndex or ind >= self.maxIndex:
                raise IndexLimitsNameError("The index value should be correct index for data element along " +
                                          "axis #" + str(self.viewAxisIndex) + " (" + str(self.minIndex) +
                                          " >= ind < " + str(self.maxIndex) + ").")
                
            self.currentIndex = ind

        def getDirectionNames(self):
            return {'view': self.viewName, 'up': self.upName}
        
        def getCurrentIndex(self):
            return self.currentIndex
        
        def hasNext(self):
            return self.currentIndex < self.maxIndex
        
        def setIndexNext(self):
            if self.hasNext():
                self.currentIndex += 1
                
            else:
                raise IndexLimitsNameError("There is no correct next value for index.")
                
        
        def hasPrev(self):
            return self.currentIndex > self.minIndex
        
        def setIndexPrev(self):
            if self.hasPrev():
                self.currentIndex -= 1
                
            else:
                raise IndexLimitsNameError("There is no correct previous value for index.")
                
                
        def getSlice(self, ind):
            if ind < self.minIndex or ind > self.maxIndex:
                raise IndexLimitsNameError("The index value should be correct index for data element along " +
                                          "axis #" + str(self.viewAxisIndex) + " (" + str(self.minIndex) +
                                          " >= ind <=" + str(self.maxIndex) + ").")
                
            return self.transformImage(self.storage.data.take(indices = ind, axis = self.viewAxisIndex))
        
        def getCurrentSlice(self):
            return self.getSlice(self.currentIndex)
            
        def getNextSlice(self):
            self.setIndexNext()
            return self.getSlice(self.currentIndex)
            
        def getPrevSlice(self):
            self.setIndexPrev()
            return self.getSlice(self.currentIndex)
        
        def getMIP(self):
            return self.transformImage(np.max(self.storage.data, self.viewAxisIndex))
            
    
    def __init__(self, data, directions = 'left-posterior-superior', 
                 initStandarView = True, verticalFlip = True):
        
        # DATA
        if not type(data) == np.ndarray:
            raise DataTypeError("Incorrect volume data type. Expected <class 'numpy.ndarray'>, but " +
                                str(type(data)) + " was detected.")
            
        if not len(data.shape) == 3:
            raise AxesError("Incorrect number of axes. Expected 3, but " +
                                str(len(data.shape)) + " was detected.")
            
        if data.size == 0:
            raise EmptyVolumeError("Volume data should be not empty.")
            
        self.data = data.copy()
        
        # AXES
        dirNames = directions.lower().split('-')
        
        if len(dirNames) == 3:
            
            self.axes = np.zeros((3, 3))
            axisNamesSet = set()
            
            for i in range(3):
                if not dirNames[i] in VolumeStorage.StandartAxisNames.keys():
                    raise DirectionsNameError("The valid direction names are \'left\' or \'right\' " + 
                                              "(for frontal axis), \'superior\' or \'inferior\' " + 
                                              "(for saggital axis) and \'posterior\' or \'anterior\' " + 
                                              "(for vertical axis) only. Ex.: \'left-posterior-superior\'.")
                
                self.axes[i] = VolumeStorage.StandartAxisDirections[dirNames[i]]
                axisNamesSet.add(VolumeStorage.StandartAxisNames[dirNames[i]])
                
                
            if not len(axisNamesSet) == 3:
                raise DirectionsNameError("Expected directions along three axes: " +
                                          "\'left\' or \'right\' " + 
                                              "(for frontal axis), \'superior\' or \'inferior\' " + 
                                              "(for saggital axis) and \'posterior\' or \'anterior\' " + 
                                              "(for vertical axis)" + 
                                          ". Ex.: \'left-posterior-superior\'. " + 
                                         "But detected direction for " + str(len(axisNamesSet)) + " axes: " +
                                         " and ".join(axisNamesSet) + ".")
            
        
        else:
            raise DirectionsNameError("Direction names are expected as three words joined by hyphen, ex.: " + 
                                "\'left-posterior-superior\'.")
            
        # VIEWS
        self.views = {}
        
        if initStandarView:
            
            self.addView('frontal', 
                         viewDirection = 'posterior', 
                         upDirection = 'superior', 
                         verticalFlip = verticalFlip)
            
            self.addView('lateral', 
                         viewDirection = 'right', 
                         upDirection = 'superior', 
                         verticalFlip = verticalFlip)
            
            self.addView('axial', 
                         viewDirection = 'inferior', 
                         upDirection = 'anterior', 
                         verticalFlip = verticalFlip)

    def getViews(self):
        return self.views
        
    def addView(self, name, viewDirection, upDirection, verticalFlip = True):
        self.views[name] = VolumeStorage.VolumeView(storage = self, 
                                                    viewDirection = viewDirection, 
                                                    upDirection = upDirection, 
                                                    verticalFlip = verticalFlip)
