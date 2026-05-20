/**
 * Export annotations of selected classes to GeoJSON.
 * For whole-project runs, one GeoJSON file is written per image using the image name.
 */

import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.io.PathIO.GeoJsonExportOptions
import qupath.lib.objects.PathObject

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
def runTarget = 'Whole Project'   // 'Current Image' or 'Whole Project'
def annotationClassNames = ['Cholangyocytes']
def outputDirectoryPath = 'X:/roy/snRNAseq_retention/analysis/Qupath/cholangyocytes/results/annotations'
def exportHierarchy = false       // true exports child objects too; false exports only the selected annotations

def wantedClasses = (annotationClassNames ?: []).collect { it.toString().trim() }.findAll { it }

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
def sanitizeFileStem = { String text ->
    (text ?: 'image').replaceAll('[\\\\/:*?<>|]', '_').trim()
}

def selectAnnotations = { Collection<PathObject> objects ->
    if (wantedClasses.isEmpty())
        return objects.findAll { obj -> obj != null && obj.isAnnotation() }

    objects.findAll { obj ->
        obj != null && obj.isAnnotation() && obj.getPathClass() != null && wantedClasses.contains(obj.getPathClass().getName())
    }
}

def exportObjects = { List<PathObject> objectsToExport, File outputFile ->
    if (objectsToExport.isEmpty()) {
        println 'No matching annotations found; nothing to export.'
        return false
    }
    outputFile.parentFile?.mkdirs()
    exportObjectsToGeoJson(objectsToExport, outputFile.absolutePath, GeoJsonExportOptions.PRETTY_JSON, GeoJsonExportOptions.FEATURE_COLLECTION)
    println "Exported ${objectsToExport.size()} annotation(s) to ${outputFile.absolutePath}"
    return true
}

def getImageStem = { imageData, entry ->
    // Prefer entry.getImageName() which gives clean image name without server-specific suffixes
    def imageName = entry?.getImageName() ?: imageData?.getServer()?.getMetadata()?.getName() ?: 'image'
    // Strip only server suffixes (e.g., :timestamp), keep file extension (.tif)
    imageName = imageName.replaceAll(':[^/]*$', '')
    sanitizeFileStem(imageName)
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
def outputDir = new File(outputDirectoryPath)
outputDir.mkdirs()

if (runTarget == 'Current Image') {
    def imageData = getCurrentImageData()
    if (imageData == null)
        throw new IllegalStateException('No image is open.')

    def entry = getProjectEntry()
    def stem = getImageStem(imageData, entry)
    def objects = exportHierarchy ? imageData.getHierarchy().getAnnotationObjects() : imageData.getHierarchy().getAnnotationObjects()
    def filtered = selectAnnotations(objects)
    def outputFile = new File(outputDir, stem + '.geojson')
    exportObjects(filtered, outputFile)
    println "[SAVING] ${outputFile.absolutePath}"
} else if (runTarget == 'Whole Project') {
    def project = getProject()
    if (project == null)
        throw new IllegalStateException('No project is open.')

    for (entry in project.getImageList()) {
        println "Processing ${entry.getImageName()}"
        def imageData = entry.readImageData()
        def annotations = imageData.getHierarchy().getAnnotationObjects()
        def filtered = selectAnnotations(annotations)
        if (filtered.isEmpty()) {
            println '  -> no matching annotations'
            continue
        }

        def stem = getImageStem(imageData, entry)
        def outputFile = new File(outputDir, stem + '.geojson')
        exportObjects(filtered, outputFile)
        println "[SAVING] ${outputFile.absolutePath}"
    }
} else {
    throw new IllegalArgumentException("Invalid runTarget: ${runTarget}")
}
