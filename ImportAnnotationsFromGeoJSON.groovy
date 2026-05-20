/**
 * Import annotations from GeoJSON files back into the project.
 * Files are matched automatically by image name: <image name>.geojson
 */

import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.io.PathIO
import qupath.lib.objects.PathObject

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
def inputDirectoryPath = 'X:/roy/snRNAseq_retention/analysis/Qupath/cholangyocytes/results/annotations'
def allowedClassNames = []   // Leave empty to import everything from the GeoJSON files
def clearExistingAnnotations = false

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
def sanitizeFileStem = { String text ->
    (text ?: 'image').replaceAll(/[\\/:*?<>|]/, '_').trim()
}

def getImageStem = { imageData, entry ->
    // Prefer entry.getImageName() which gives clean image name without server-specific suffixes
    def imageName = entry?.getImageName() ?: imageData?.getServer()?.getMetadata()?.getName() ?: 'image'
    // Strip only server suffixes (e.g., :timestamp), keep file extension (.tif)
    imageName = imageName.replaceAll(':[^/]*$', '')
    sanitizeFileStem(imageName)
}

def filterObjects = { Collection<PathObject> objects ->
    def wanted = (allowedClassNames ?: []).collect { it.toString().trim() }.findAll { it }
    if (wanted.isEmpty())
        return objects.findAll { it != null && it.isAnnotation() }

    objects.findAll { obj ->
        obj != null && obj.isAnnotation() && obj.getPathClass() != null && wanted.contains(obj.getPathClass().getName())
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
def project = getProject()
if (project == null)
    throw new IllegalStateException('No project is open.')

def inputDir = new File(inputDirectoryPath)
if (!inputDir.isDirectory())
    throw new IllegalArgumentException("Input directory does not exist: ${inputDirectoryPath}")

def geoJsonFiles = inputDir.listFiles { file ->
    file != null && file.isFile() && file.name.toLowerCase().endsWith('.geojson')
}?.toList() ?: []

if (geoJsonFiles.isEmpty())
    throw new IllegalArgumentException("No GeoJSON files found in ${inputDirectoryPath}")

def filesByStem = [:]
geoJsonFiles.each { file ->
    def stem = file.name.substring(0, file.name.lastIndexOf('.'))
    filesByStem[stem] = file
}

for (entry in project.getImageList()) {
    def imageData = entry.readImageData()
    def hierarchy = imageData.getHierarchy()
    if (hierarchy == null) {
        println "Skipping ${entry.getImageName()} - no hierarchy"
        continue
    }

    def stem = getImageStem(imageData, entry)
    def file = filesByStem[stem]
    if (file == null) {
        println "Skipping ${entry.getImageName()} - no matching GeoJSON file"
        continue
    }

    println "Importing ${file.name} -> ${entry.getImageName()}"
    def objects = PathIO.readObjects(file)
    if (objects == null || objects.isEmpty()) {
        println '  -> no objects found'
        continue
    }

    def importedObjects = filterObjects(objects)
    if (importedObjects.isEmpty()) {
        println '  -> no matching annotations after filtering'
        continue
    }

    if (clearExistingAnnotations) {
        hierarchy.removeObjects(new ArrayList<>(hierarchy.getAnnotationObjects()), true)
    }

    hierarchy.addObjects(importedObjects)
    println "  -> imported ${importedObjects.size()} annotation(s)"
}
