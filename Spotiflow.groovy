/**
 * Spotiflow Detection Template script
 * @author Rémy Dornier
 *
 * This script is a template to detect objects using a Spotiflow model within QuPath.
 * After defining the builder, it will:
 * 1. Find all selected annotations in the current open ImageEntry
 * 2. Export the selected annotations to a temp folder that can be specified with tempDirectory()
 * 3. Run the spotiflow detction using the defined/default model name or path
 * 4. Create the desired objects (i.e. points) with the selected statistics (i.e. spotiflow outputs)
 *
 * NOTE: that this template does not contain all options, but should help get you started
 * See all options by calling spotiflow.helpPredict()
 *
 * NOTE 2: You should change pathObjects get all annotations if you want to run for the project. By default this script
 * will only run on the selected annotations.
 */

import qupath.lib.roi.ROIs
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory
import qupath.imagej.tools.IJTools
import qupath.lib.regions.RegionRequest
import qupath.lib.io.PathIO
import qupath.ext.biop.spotiflow.Spotiflow
import ij.*
import ij.measure.Measurements
import ij.process.ImageStatistics
import java.util.ArrayList
// Logging imports removed - let all output show naturally

// If you have trained a custom model, specify the model directory as a File in setModelDir()
// If you want to use any other pre-trained models, specify its name in setPretrainedModelName()
// -> List of all pre-trained models : https://weigertlab.github.io/spotiflow/pretrained.html
def configValue = { String name, value -> binding.hasVariable(name) ? binding.getVariable(name) : value }

def resolveImageName = { imageData, entry = null ->
        def serverPath = imageData?.getServer()?.getPath()?.toString()
        def serverName = serverPath != null ? new File(serverPath).getName() : null
        serverName ?: entry?.getImageName() ?: imageData?.getServer()?.getMetadata()?.getName() ?: 'image'
}

def channels = configValue('channels', ["Cy5", "TMR"])
def runTarget = configValue('runTarget', 'Current Image')   // 'Current Image' or 'Whole Project'
def detectionScope = configValue('detectionScope', 'Specific annotation classes')  // 'Whole image', 'All annotations', or 'Specific annotation classes'
def annotationClassNames = configValue('annotationClassNames', ['Cholangyocytes'])  // Used only when detectionScope == 'Specific annotation classes'
def distanceAnnotationClassName = configValue('distanceAnnotationClassName', 'Cholangyocytes')  // Distance is computed relative to annotations with this class name
def outputCsvPath = configValue('outputCsvPath', 'X:/roy/snRNAseq_retention/analysis/Qupath/cholangyocytes/results/temp.csv')  // CSV export path for detected spots
def measureChannelIntensity = configValue('measureChannelIntensity', true)  // Measure intensity for all channels at each spot
def save_detections = configValue('save_detections', true)  // If false, detections are removed before saving image data
// Spotiflow output suppression removed - output now shows naturally


Date start = new Date()

def spotiflow = Spotiflow.builder()
//        .tempDirectory(new File("path/to/tmp/folder"))       // OPTIONAL : default is in 'qpProject/spotiflow-temp' folder
//        .setModelDir(new File("path/to/my/model"))           // OPTIONAL : path to your own trained model
//        .setPretrainedModelName("smfish_3d")                 // OPTIONAL : Default is 'general'
//        .setMinDistance(2)                                   // OPTIONAL : Positive integer value
//        .setProbabilityThreshold(0.2)                        // OPTIONAL : Positive value
//        .disableGPU()                                        // OPTIONAL : Force using CPU ; default is automatic (let spotiflow decide)
//        .process3d()                                         // OPTIONAL : process the entire zstack
//        .zPositions(0,5)                                     // OPTIONAL : ONLY works wih process3d(). Select a sub-stack (start and end inclusive)
//        .doSubpixel(true)                                    // OPTIONAL : true to get subpixel resolution ; false to not. Default: let spotiflow choose
//        .setClass("ClassName")                               // OPTIONAL : set the same class for all detections. Default: not assign any classes
        .setClassChannelName()                               // OPTIONAL : create a new class for each channel and assign detection to it. Default: not assign any classes
        .nThreads(20)                                        // OPTIONAL : How much you want to paralellize processing. Default 12
//        .saveBuilder("MyFancyName")                          // OPTIONAL : To save builder parameters as JSON file
//        .saveTempImagesAsOmeZarr()                           // OPTIONAL : ONLY AVAILABLE FOR SPOTIFLOW >= 0.5.8. Save temp images as ome-zarr instead of ome.tiff
//        .clearAllChildObjects()                              // OPTIONAL : Clear all previous detections, whatever their class
//        .createAnnotations()                                 // OPTIONAL : Create annotations instead of detections. WARNING: this can slow up a lot QuPath. Only to use to pre-annotated small patches for later training.
        .clearChildObjectsBelongingToCurrentChannels()       // OPTIONAL : Clear all previous detections which belong to the current selected channels (i.e. with their class set with the name of the channel)
        .channels(*channels)        // REQUIRED : list of channel name(s) to process. At least one channel is required
        .cleanTempDir()                                      // OPTIONAL : Clean all files from the tempDirectory
//        .addParameter("key","value")                         // OPTIONAL : Add more parameter, base on the available ones
        .build()

// ******************** SCRIPT STARTS HERE - you usually don't need to change anything below this line **************************

def geometryFactory = new GeometryFactory()

def validRunTargets = ['Current Image', 'Whole Project']
def validScopes = ['Whole image', 'All annotations', 'Specific annotation classes']

if (!validRunTargets.contains(runTarget)) {
        throw new IllegalArgumentException("Invalid runTarget: ${runTarget}. Use one of ${validRunTargets}")
}
if (!validScopes.contains(detectionScope)) {
        throw new IllegalArgumentException("Invalid detectionScope: ${detectionScope}. Use one of ${validScopes}")
}

def classNames = (annotationClassNames ?: []).collect { it.toString().trim() }.findAll { it }
// Use two forms: a variable-safe lowercase name for measurements/properties,
// and a human-readable header form that preserves the annotation class name.
def distanceVarBase = "distance_to_${distanceAnnotationClassName.replaceAll('\\s+','_').toLowerCase()}"
def distanceHeaderBase = "distance_to_${distanceAnnotationClassName.replaceAll('\\s+','_')}"

def getImageIdentifier = { imageData, entry = null ->
        entry?.getID() ?: imageData.getServer().getPath()
}

// Helper to build pathObjects for a given imageData and selected scope
def buildPathObjectsForImage = { imageData ->
        def server = imageData.getServer()
        def plane = ImagePlane.getDefaultPlane()

        if (detectionScope == 'Whole image') {
                def roi = ROIs.createRectangleROI(0, 0, server.getWidth(), server.getHeight(), plane)
                return [PathObjects.createAnnotationObject(roi)]
        }

        def annotations = imageData.getHierarchy().getAnnotationObjects()

        if (detectionScope == 'All annotations') {
                return annotations
        }

        return annotations.findAll { annotation ->
                def pathClass = annotation.getPathClass()
                pathClass != null && classNames.contains(pathClass.getName())
        }
}

def collectDistanceAnnotationGeometry = { imageData ->
        def distanceAnnotations = imageData.getHierarchy().getAnnotationObjects().findAll { annotation ->
                def pathClass = annotation.getPathClass()
                pathClass != null && pathClass.getName() == distanceAnnotationClassName
        }

        def geometries = distanceAnnotations.collect { it.getROI()?.getGeometry() }.findAll { it != null }
        if (geometries.isEmpty())
                return null

        def combinedGeometry = geometries[0]
        for (int i = 1; i < geometries.size(); i++) {
                combinedGeometry = combinedGeometry.union(geometries[i])
        }
        return combinedGeometry
}

def findAnnotationParent = { pathObject ->
        def parent = pathObject?.getParent()
        while (parent != null && !parent.isAnnotation()) {
                parent = parent.getParent()
        }
        return parent != null && parent.isAnnotation() ? parent : null
}

def signedDistanceToGeometry = { Geometry targetGeometry, Geometry pointGeometry ->
        if (targetGeometry == null || pointGeometry == null)
                return Double.NaN

        def inside = targetGeometry.contains(pointGeometry) || targetGeometry.touches(pointGeometry)
        if (inside) {
                def boundaryDistance = targetGeometry.getBoundary().distance(pointGeometry)
                return -boundaryDistance
        }
        return targetGeometry.distance(pointGeometry)
}

def computeAnnotationChannelStats = { annotation, server, channelNames ->
        def statsMap = [:]
        if (annotation == null || annotation.getROI() == null)
                return statsMap

        try {
                def req = RegionRequest.createInstance(server.getPath(), 1.0, annotation.getROI())
                def pathImg = IJTools.convertToImagePlus(server, req)
                def imp = pathImg.getImage()
                def ijRoi = IJTools.convertToIJRoi(annotation.getROI(), pathImg)

                if (ijRoi == null)
                        return statsMap

                int nChannels = imp.getNChannels() ?: 1
                for (int ch = 1; ch <= nChannels; ch++) {
                        try {
                                def proc = null
                                if (nChannels > 1) {
                                        int stackIndex = imp.getStackIndex(ch, 1, 1)
                                        proc = imp.getStack().getProcessor(stackIndex)
                                } else {
                                        proc = imp.getProcessor()
                                }

                                if (proc == null)
                                        continue

                                proc.setRoi(ijRoi)
                                def stats = ImageStatistics.getStatistics(proc, Measurements.MEAN + Measurements.MIN_MAX, null)
                                def chName = channelNames[ch] ?: "Channel_${ch}"
                                statsMap["annotation_mean_${chName}".toString()] = stats?.mean
                                statsMap["annotation_max_${chName}".toString()] = stats?.max
                                statsMap["annotation_min_${chName}".toString()] = stats?.min
                        } catch (e) {
                                // skip this channel
                        }
                }
        } catch (e) {
                println "Warning: failed to compute annotation channel stats: ${e}"
        }

        return statsMap
}

def appendSpotRows = { imageData, imageName, outRows ->
        def distanceGeometry = collectDistanceAnnotationGeometry(imageData)
        def server = imageData.getServer()
        def pixelCal = server.getMetadata()?.getPixelCalibration()
        def pixelSize = Double.NaN
        if (pixelCal != null) {
                try {
                        pixelSize = pixelCal.getAveragedPixelSizeMicrons()
                } catch (e) {
                        pixelSize = Double.NaN
                }
        }
        if (Double.isNaN(pixelSize)) {
                // fallback: try server helper
                try {
                        pixelSize = server.getAveragedPixelSizeMicrons()
                } catch (e) {
                        pixelSize = 1.0
                }
        }

        // Get channel names from server metadata once (for all spots)
        def channelNames = [:]
        try {
                def meta = server.getMetadata()
                if (meta != null) {
                        def metaChannels = meta.getChannels()
                        if (metaChannels != null && metaChannels.size() > 0) {
                                for (int ci = 0; ci < metaChannels.size(); ci++) {
                                        channelNames[ci + 1] = metaChannels[ci].getName() ?: "Channel_${ci + 1}"
                                }
                        }
                }
        } catch(e) {
                // metadata channels not available
        }

        def annotationStatsCache = [:]

        // collect all detection objects (don't filter by channels here)
        def detections = imageData.getHierarchy().getDetectionObjects()

        detections.each { detection ->
                def roi = detection.getROI()
                if (roi == null) return

                def annotationParent = findAnnotationParent(detection)
                def annotationId = annotationParent?.getID()
                def annotationClassification = annotationParent?.getPathClass()?.getName()
                def annotationChannelStats = [:]

                if (measureChannelIntensity && annotationParent != null) {
                        def cacheKey = annotationId != null ? annotationId : annotationParent
                        if (!annotationStatsCache.containsKey(cacheKey)) {
                                annotationStatsCache[cacheKey] = computeAnnotationChannelStats(annotationParent, server, channelNames)
                        }
                        annotationChannelStats = annotationStatsCache[cacheKey] ?: [:]
                }

                def x_px = roi.getCentroidX()
                def y_px = roi.getCentroidY()
                def x_um = Double.isFinite(x_px) ? x_px * pixelSize : Double.NaN
                def y_um = Double.isFinite(y_px) ? y_px * pixelSize : Double.NaN

                def pointGeometry = roi.getGeometry()?.getCentroid()
                def signedDistancePx = signedDistanceToGeometry(distanceGeometry, pointGeometry)
                def signedDistanceUm = Double.isFinite(signedDistancePx) ? signedDistancePx * pixelSize : Double.NaN

                def channelName = detection.getPathClass()?.getName() ?: ''

                // Optionally measure intensity for all channels at the exact spot pixel
                def channelIntensities = [:]
                if (measureChannelIntensity) {
                        try {
                                def cx = Math.round(x_px) as int
                                def cy = Math.round(y_px) as int
                                def req = RegionRequest.createInstance(server.getPath(), 1.0, cx, cy, 1, 1)
                                def pathImg = IJTools.convertToImagePlus(server, req)
                                def imp = pathImg.getImage()

                                // Measure intensity for each channel
                                int nChannels = imp.getNChannels() ?: 1
                                for (int ch = 1; ch <= nChannels; ch++) {
                                        try {
                                                def proc = null
                                                if (nChannels > 1) {
                                                        int stackIndex = imp.getStackIndex(ch, 1, 1)
                                                        proc = imp.getStack().getProcessor(stackIndex)
                                                } else {
                                                        proc = imp.getProcessor()
                                                }

                                                if (proc != null) {
                                                        def intensity = proc.getPixelValue(0, 0)
                                                        def chName = channelNames[ch] ?: "Channel_${ch}"
                                                        channelIntensities["intensity_${chName}".toString()] = intensity
                                                }
                                        } catch (e) {
                                                // skip this channel
                                        }
                                }
                        } catch (e) {
                                println "Warning: failed to sample channel intensities: ${e}"
                        }
                }

                outRows << [
                        image_name: imageName,
                        channel: channelName,
                        annotation_id: annotationId,
                        annotation_classification: annotationClassification,
                        x: x_px,
                        y: y_px,
                        x_um: x_um,
                        y_um: y_um,
                        (distanceHeaderBase + '_px'): signedDistancePx,
                        (distanceHeaderBase + '_um'): signedDistanceUm
                ] + channelIntensities + annotationChannelStats

                // Attach measurements/properties to the detection so they are available in QuPath
                try {
                        // Preferred API if available (use variable-safe measurement keys)
                        detection.setMeasurementValue(distanceVarBase + '_px', signedDistancePx)
                        detection.setMeasurementValue(distanceVarBase + '_um', signedDistanceUm)
                        channelIntensities.each { key, value ->
                                detection.setMeasurementValue(key, value)
                        }
                } catch (e1) {
                        try {
                                // Try putting into a measurement list
                                def ml = detection.getMeasurementList()
                                ml.put(distanceVarBase + '_px', signedDistancePx)
                                ml.put(distanceVarBase + '_um', signedDistanceUm)
                                channelIntensities.each { key, value ->
                                        ml.put(key, value)
                                }
                        } catch (e2) {
                                try {
                                        // Fallback to properties map
                                        detection.getProperties().put(distanceVarBase + '_px', signedDistancePx)
                                        detection.getProperties().put(distanceVarBase + '_um', signedDistanceUm)
                                        channelIntensities.each { key, value ->
                                                detection.getProperties().put(key, value)
                                        }
                                } catch (e3) {
                                        println "Warning: failed to attach measurements to detection: ${e3}"
                                }
                        }
                }
        }
}

def writeCsv = { rows, csvPath ->
        def outputFile = new File(csvPath)
        def parent = outputFile.parentFile
        if (parent != null)
                parent.mkdirs()

        outputFile.withPrintWriter('UTF-8') { writer ->
                def finalHeaders = ['image_name','channel','annotation_id','annotation_classification','x','y','x_um','y_um', distanceHeaderBase + '_px', distanceHeaderBase + '_um']
                if (!rows.isEmpty()) {
                        def headers = rows.collectMany { it.keySet() }.unique()
                        finalHeaders = finalHeaders.findAll { headers.contains(it) } + (headers - finalHeaders)
                }

                writer.println finalHeaders.join(',')
                rows.each { row ->
                        def values = finalHeaders.collect { header ->
                                def value = row.containsKey(header) ? row.get(header) : null
                                if (value == null)
                                        return ''
                                def text = value.toString()
                                if (text.contains(',') || text.contains('"') || text.contains('\n'))
                                        return '"' + text.replace('"', '""') + '"'
                                return text
                        }
                        writer.println values.join(',')
                }
        }
}

def saveImageDataForImage = { imageData ->
        def projectEntry = getProjectEntry()
        if (projectEntry != null) {
                projectEntry.saveImageData(imageData)
                return
        }

        def lastSavedPath = imageData?.getLastSavedPath()
        if (lastSavedPath != null) {
                PathIO.writeImageData(new File(lastSavedPath), imageData)
                return
        }

        println "Warning: no project entry or last saved path available; image data was not saved."
}

def finalizeImageData = { imageData ->
        def hierarchy = imageData?.getHierarchy()
        if (hierarchy == null)
                return

        if (!save_detections) {
                def detectionsToRemove = new ArrayList<>(hierarchy.getDetectionObjects())
                if (!detectionsToRemove.isEmpty())
                        hierarchy.removeObjects(detectionsToRemove, true)
        }

        saveImageDataForImage(imageData)
}

// Run on current image
def csvRows = []

if (runTarget == 'Current Image') {
        def imageData = getCurrentImageData()
        def pathObjects = buildPathObjectsForImage(imageData)
        if (!pathObjects || pathObjects.isEmpty()) {
                throw new IllegalStateException("No matching parent objects/annotations found for the chosen scope.")
        }
        spotiflow.detectObjects(imageData, getImageIdentifier(imageData, getProjectEntry()), pathObjects)

        def imageName = resolveImageName(imageData, getProjectEntry())
        def rowsBeforeAppend = csvRows.size()
        appendSpotRows(imageData, imageName, csvRows)
        def spotsFound = csvRows.size() - rowsBeforeAppend
        println "[SPOTS FOUND] ${spotsFound} spots in ${imageName}"
        finalizeImageData(imageData)
} else { // Whole project
        def project = getProject()
        for (entry in project.getImageList()) {
                println "Processing: ${entry.getImageName()}"
                def imageData = entry.readImageData()
                def pathObjects = buildPathObjectsForImage(imageData)
                if (!pathObjects || pathObjects.isEmpty()) {
                        println "  -> No matching objects for scope; skipping"
                        continue
                }
                spotiflow.detectObjects(imageData, entry.getID(), pathObjects)

                def rowsBeforeAppend = csvRows.size()
                appendSpotRows(imageData, resolveImageName(imageData, entry), csvRows)
                def spotsFound = csvRows.size() - rowsBeforeAppend
                println "[SPOTS FOUND] ${spotsFound} spots in ${entry.getImageName()}"
                finalizeImageData(imageData)
        }
}

writeCsv(csvRows, outputCsvPath)
println "[SAVING] ${outputCsvPath}"

// You could do some post-processing here, e.g. to remove objects that are too small, but it is usually better to
// do this in a separate script so you can see the results before deleting anything.

Date stop = new Date()
long milliseconds = stop.getTime() - start.getTime()
int seconds = (int) (milliseconds / 1000) % 60 ;
int minutes = (int) ((milliseconds / (1000*60)) % 60);
int hours   = (int) ((milliseconds / (1000*60*60)) % 24);
println "Spotiflow completed in " + hours + "h " + minutes + "m " + seconds + "s"
