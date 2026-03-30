import javafx.application.Platform
import qupath.lib.roi.ROIs
import qupath.lib.objects.PathObjects

def x = 37376
def y = 29696
def size = 512

Platform.runLater {

    // Move to tile center
    getCurrentViewer().setCenterPixelLocation(x + size/2, y + size/2)

    // Create ROI
    def roi = ROIs.createRectangleROI(x, y, size, size, null)

    // Convert ROI → PathObject (THIS FIXES YOUR ERROR)
    def annotation = PathObjects.createAnnotationObject(roi)

    // Add to viewer
    addObject(annotation)
}