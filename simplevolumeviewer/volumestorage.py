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

class SpacingValueError(Exception):
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
    
    class Interpolator:
    
        approxMatrix = np.array([[-1.,  3., -3.,  1.],
                                 [ 3., -6.,  3.,  0.],
                                 [-3.,  0.,  3.,  0.],
                                 [ 1.,  4.,  1.,  0.]
                                ]) / 6.

        interpMatrix = np.array([[-1.,  3., -3.,  1.],
                                 [ 2., -5.,  4., -1.],
                                 [-1.,  0.,  1.,  0.],
                                 [ 0.,  2.,  0.,  0.]
                                ]) / 2.

        linearMatrix = np.array([[0,  0.,  0., 0.],
                                 [0,  0.,  0., 0.],
                                 [0, -1.,  1., 0.],
                                 [0,  1.,  0., 0.]
                                ])

        @staticmethod
        def prepareInterpolation(baseNumb, scaleCoef):

            X = np.arange(baseNumb)
            Y = np.arange(int(baseNumb * scaleCoef)) / scaleCoef
            Y = Y[Y <= baseNumb - 1]
            d = baseNumb - 1 - Y[-1]
            Y += d / 2

            # indices
            ind = np.zeros((Y.shape[0], 4))

            ind[:, 1] = Y.astype('int')
            ind[:, 0] = ind[:, 1] - 1
            ind[:, 2] = ind[:, 1] + 1
            ind[:, 3] = ind[:, 1] + 2

            ind = np.clip(ind, 0, baseNumb - 1).astype('int')

            # interpolation parameters
            T = np.zeros((Y.shape[0], 4))

            T[:, 3] = 1.
            T[:, 2] = Y - ind[:, 1]
            T[:, 1] = T[:, 2] ** 2
            T[:, 0] = T[:, 2] ** 3

            return ind.reshape(-1), T

        def __init__(self, vSize, vSpacing, hSize, hSpacing):

            if vSpacing > hSpacing:

                self.horizontalScale = False
                self.ind, self.T = VolumeStorage.Interpolator.prepareInterpolation(vSize, vSpacing / hSpacing)

            else:

                self.horizontalScale = True
                self.ind, self.T = VolumeStorage.Interpolator.prepareInterpolation(hSize, hSpacing / vSpacing)
                
        def processImage(self, img, interpModel):
            
            if self.horizontalScale:
                I = img.T[self.ind]
                
            else:
                I = img[self.ind]
                
            I = I.reshape(-1, 4, I.shape[1])
                
            if interpModel == 'L' or interpModel == 'linear':
                M = self.T @ VolumeStorage.Interpolator.linearMatrix
                
            elif interpModel == 'A' or interpModel == 'approximation':
                M = self.T @ VolumeStorage.Interpolator.approxMatrix
                
            elif interpModel == 'I' or interpModel == 'interpolation':
                M = self.T @ VolumeStorage.Interpolator.interpMatrix
                
            result = np.sum((M.reshape(-1) * I.reshape(-1, I.shape[2]).T).T.reshape(-1, 4, I.shape[2]), 1)
            
            if self.horizontalScale:
                result = result.T
            
            return result
            
    
    class VolumeView:
        
        def __init__(self, storage, viewDirection, upDirection, verticalFlip = True, interpModel = 'none'):
            
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
            
            # SPACING
            vSize    = self.storage.data.shape[self.upAxisIndex]
            vSpacing = self.storage.spacing[self.upAxisIndex]
            hSize    = self.storage.data.shape[self.rightAxisIndex]
            hSpacing = self.storage.spacing[self.rightAxisIndex]
                
            # interpolation
            if not vSpacing == hSpacing:
                self.interpoalateFlag = True
                self.interpoalator = VolumeStorage.Interpolator(vSize = vSize, 
                                                                vSpacing = vSpacing, 
                                                                hSize = hSize, 
                                                                hSpacing = hSpacing)
            else:
                self.interpoalateFlag = False
                
            self.interpModel = interpModel
                
        def setInterpolationModel(self, interpModel):
            self.interpModel = interpModel
            
        def getInterpolationModel(self):
            return self.interpModel
            
        def transformImage(self, img):
            if self.interpoalateFlag and not self.interpModel == 'none':
                return self.interpoalator.processImage((img.T if self.tranposeFlag else img)[self.slice], 
                                                        self.interpModel) 
                
            else:
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
            
    
    def __init__(self, data, directions = 'left-posterior-superior', spacing = np.ones((3)),
                 initStandarView = True, verticalFlip = True, interpModel = 'none'):
        
        # DATA
        if not type(data) == np.ndarray:
            raise DataTypeError("Incorrect volume data type. Expected <class 'numpy.ndarray'>, but " +
                                str(type(data)) + " was detected.")
            
        if not len(data.shape) == 3:
            raise AxesError("Incorrect number of data axes. Expected 3, but " +
                                str(len(data.shape)) + " was detected.")
            
        if data.size == 0:
            raise SpacingValueError("Volume data should be not empty.")
            
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
            
        #SPACING
        if not type(spacing) == np.ndarray:
            raise DataTypeError("Incorrect spacing data type. Expected <class 'numpy.ndarray'>, but " +
                                str(type(spacing)) + " was detected.")
            
        if not (len(spacing.shape) == 1 and spacing.shape[0] == 3): 
            raise AxesError("Incorrect spacing geometry. Expected 3 elements along 1 axis, but " +
                                str(len(spacing.shape)) + " axes with (" + ", ".join(str(a) for a in spacing.shape) +
                                                                                ") lenghts was detected.")
            
        if np.min(spacing) <= 0:
            raise DataTypeError("Spacing value should be positive number.")
            
        self.spacing = spacing.copy()
        
        # VIEWS
        self.views = {}
        
        if initStandarView:
            
            self.addView('frontal', 
                         viewDirection = 'posterior', 
                         upDirection = 'superior', 
                         verticalFlip = verticalFlip,
                         interpModel = interpModel)
            
            self.addView('lateral', 
                         viewDirection = 'right', 
                         upDirection = 'superior', 
                         verticalFlip = verticalFlip,
                         interpModel = interpModel)
            
            self.addView('axial', 
                         viewDirection = 'inferior', 
                         upDirection = 'anterior', 
                         verticalFlip = verticalFlip,
                         interpModel = interpModel)
          

    def getViews(self):
        return self.views
        
    def addView(self, name, viewDirection, upDirection, verticalFlip = True, interpModel = 'none'):
        self.views[name] = VolumeStorage.VolumeView(storage = self, 
                                                    viewDirection = viewDirection, 
                                                    upDirection = upDirection, 
                                                    verticalFlip = verticalFlip,
                                                    interpModel = interpModel)
