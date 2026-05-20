/**
 * Apply the saved pixel classifier "nucs_1" and create Nuc annotation objects.
 * Filters: minimum object size 3 um^2, minimum hole size 1 um^2.
 */

import static qupath.lib.gui.scripting.QPEx.*

def configValue = { String name, value -> binding.hasVariable(name) ? binding.getVariable(name) : value }

String classifierName = configValue('classifierName', 'nucs_1')
double minAreaUm2 = configValue('minAreaUm2', 3.0)
double minHoleAreaUm2 = configValue('minHoleAreaUm2', 1.0)
String nucObjectClassName = configValue('nucObjectClassName', 'Nuc')
boolean saveImageData = configValue('saveImageData', true)

if (getCurrentImageData() == null)
    throw new IllegalStateException('No image is open.')

if (getCurrentImageData().getHierarchy() == null)
    throw new IllegalStateException('No hierarchy is available for the current image.')

// Remove any existing Nuc annotations before running the classifier.
def hierarchy = getCurrentImageData().getHierarchy()
def existingNuc = new LinkedHashSet()
existingNuc.addAll(hierarchy.getAnnotationObjects().findAll { obj ->
    obj?.getPathClass()?.getName() == nucObjectClassName
})
existingNuc.addAll(hierarchy.getDetectionObjects().findAll { obj ->
    obj?.getPathClass()?.getName() == nucObjectClassName
})
if (!existingNuc.isEmpty()) {
    hierarchy.removeObjects(new ArrayList<>(existingNuc), true)
    println "Removed ${existingNuc.size()} existing '${nucObjectClassName}' object(s)"
}

def existingAnnotations = new LinkedHashSet(hierarchy.getAnnotationObjects())
createAnnotationsFromPixelClassifier(classifierName, minAreaUm2, minHoleAreaUm2)

def nucClass = getPathClass(nucObjectClassName)
def newAnnotations = getCurrentImageData().getHierarchy().getAnnotationObjects().findAll { !existingAnnotations.contains(it) }
newAnnotations.each { annotation ->
    if (annotation.getPathClass() == null)
        annotation.setPathClass(nucClass)
}

if (saveImageData) {
    def projectEntry = getProjectEntry()
    if (projectEntry != null) {
        projectEntry.saveImageData(getCurrentImageData())
    } else if (getCurrentImageData().getLastSavedPath() != null) {
        qupath.lib.io.PathIO.writeImageData(new File(getCurrentImageData().getLastSavedPath()), getCurrentImageData())
    } else {
        println 'Warning: no project entry or last saved path available; image data was not saved.'
    }
}

println "Applied pixel classifier '${classifierName}' and created ${newAnnotations.size()} '${nucObjectClassName}' annotation(s) (min size ${minAreaUm2} um^2, min hole size ${minHoleAreaUm2} um^2)."
