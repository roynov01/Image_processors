/**
 * advanced_spotiflow
 * - Parameters at the top
 * - Import annotations from a per-image `.geojson`
 * - Apply the nuc pixel classifier inline
 * - Run Spotiflow detection
 * - Compute distances and channel intensities
 * - Append to CSV and log files
 */

import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.io.PathIO
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.regions.RegionRequest
import qupath.imagej.tools.IJTools
import qupath.ext.biop.spotiflow.Spotiflow
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.Coordinate
import qupath.lib.analysis.features.ObjectMeasurements
import ij.process.ImageStatistics
import ij.measure.Measurements

// ---------------------------
// Parameters (edit these)
// ---------------------------
// Annotation import parameters
def annotationInputDirectoryPath = 'results/annotations'
def annotationClassNamesToLoad = ['Cholangyocytes']
def clearExistingAnnotationsBeforeLoad = true

// Nuclei pixel classifier parameters
def classifierName = 'nucs_1'
def minAreaUm2 = 3.0
def minHoleAreaUm2 = 1.0
def nucObjectClassName = 'Nuc'
def saveImageDataAfterClassifier = true

//Spotiflow parameters
def channels = ['Cy5', 'TMR']
def detectionScope = 'Specific annotation classes' // 'Whole image' | 'All annotations' | 'Specific annotation classes'
def annotationClassNames = ['Cholangyocytes']
def distanceAnnotationClassName = 'Nuc'
def measureChannelIntensity = true
def save_detections = true
def clearDetectionsBeforeSpotiflow = true

// export parameters
def baseResultsDir = 'results'
def appendCsv = true
def appendLog = true
def clearCsvAtRunStart = false
def outputCsvName = 'spots.csv'
def logFileName = 'spotiflow_log.txt'

// ---------------------------
// Paths
// ---------------------------
def projectDir = null
try {
    def projectPath = new File(getProject().getPath().toString())
    projectDir = projectPath.isDirectory() ? projectPath : projectPath.getParentFile()
} catch (e) {
    projectDir = new File('.')
}
if (projectDir == null)
    projectDir = new File('.')
def resultsDirFile = baseResultsDir == null || baseResultsDir.trim().isEmpty() ? projectDir : (new File(baseResultsDir).isAbsolute() ? new File(baseResultsDir) : new File(projectDir, baseResultsDir))
resultsDirFile.mkdirs()

def outputCsvPath = new File(resultsDirFile, outputCsvName).absolutePath
def logFilePath = new File(resultsDirFile, logFileName).absolutePath
def outputCsvFile = new File(outputCsvPath)
def logFile = new File(logFilePath)
if (!appendLog && logFile.exists())
    logFile.delete()

def logWriter = { String msg ->
    println msg
    logFile.append(msg + System.lineSeparator())
}

def logImageHeader = { imageName, imageStem, csvPathValue, logPathValue ->
    def now = new java.text.SimpleDateFormat('yyyy-MM-dd HH:mm:ss.SSS').format(new Date())
    logWriter('--------------------------------------------------')
    logWriter("[IMAGE START] ${now}")
    logWriter("[IMAGE NAME] ${imageName}")
    logWriter("[IMAGE STEM] ${imageStem}")
    logWriter("[CSV] ${csvPathValue}")
    logWriter("[LOG] ${logPathValue}")
    logWriter("[CONFIG] annotationClassNames=${annotationClassNames}; distanceAnnotationClassName=${distanceAnnotationClassName}; measureChannelIntensity=${measureChannelIntensity}; appendCsv=${appendCsv}; appendLog=${appendLog}")
}

def countRowsWithMissingValues = { rows, keys ->
    rows.count { row -> keys.any { key -> row[key] == null || row[key].toString().trim().isEmpty() } }
}

def countRowsWithFiniteDistances = { rows ->
    def isFiniteValue = { v ->
        if (v == null) return false
        if (v instanceof Number) return Double.isFinite(((Number) v).doubleValue())
        try {
            def d = Double.parseDouble(v.toString())
            return Double.isFinite(d)
        } catch (ignored) {
            return false
        }
    }
    rows.count { row ->
        def px = row["distance_to_${distanceAnnotationClassName}_px"]
        def um = row["distance_to_${distanceAnnotationClassName}_um"]
        isFiniteValue(px) || isFiniteValue(um)
    }
}

// ---------------------------
// Helpers
// ---------------------------
def sanitizeFileStem = { String text -> (text ?: 'image').replaceAll('[\\/:*?"<>|]', '_').trim() }

def toGroovyLiteral = { value ->
    if (value == null)
        return 'null'
    if (value instanceof CharSequence)
        return '"' + value.toString().replace('\\', '\\\\').replace('"', '\\"') + '"'
    if (value instanceof Boolean || value instanceof Number)
        return value.toString()
    if (value instanceof Collection)
        return '[' + value.collect { item -> toGroovyLiteral(item) }.join(', ') + ']'
    if (value.getClass().isArray())
        return '[' + value.toList().collect { item -> toGroovyLiteral(item) }.join(', ') + ']'
    return '"' + value.toString().replace('\\', '\\\\').replace('"', '\\"') + '"'
}

def runScriptWithVariables = { File scriptFile, Map scriptVariables, def entryBinding = null ->
    if (!scriptFile.isFile())
        throw new IllegalArgumentException("Script file does not exist: ${scriptFile.absolutePath}")
    def scriptText = scriptFile.getText('UTF-8')
    def injectedLines = scriptVariables.collect { key, value ->
        "binding.setVariable(${toGroovyLiteral(key.toString())}, ${toGroovyLiteral(value)})"
    }
    if (entryBinding != null)
        injectedLines << "binding.setVariable('entry', ${toGroovyLiteral(entryBinding)})"
    if (!injectedLines.isEmpty())
        scriptText = injectedLines.join('\n') + '\n' + scriptText
    def tempFile = File.createTempFile('Wrapped_', '_' + scriptFile.name.replaceAll('[^A-Za-z0-9_]', '_') + '.groovy')
    try {
        tempFile.setText(scriptText, 'UTF-8')
        getQuPath().runScript(tempFile, null)
    } finally {
        tempFile.delete()
    }
}
def getImageStem = { imageData, qupathEntry ->
    def candidate = qupathEntry?.getImageName()
    if (candidate == null || candidate.trim().isEmpty() || !candidate.contains('.')) {
        candidate = imageData?.getServer()?.getMetadata()?.getName()
    }
    if (candidate == null || candidate.trim().isEmpty() || !candidate.contains('.')) {
        def serverPath = imageData?.getServer()?.getPath()?.toString()
        if (serverPath != null) {
            def lastSlash = Math.max(serverPath.lastIndexOf('/'), serverPath.lastIndexOf('\\'))
            candidate = lastSlash >= 0 ? serverPath.substring(lastSlash + 1) : serverPath
        }
    }
        candidate = (candidate ?: 'image').replaceAll(':[^/\\\\]*$', '')
    sanitizeFileStem(candidate)
}

def buildPathObjectsForImage = { imageData ->
    def server = imageData.getServer()
    def plane = ImagePlane.getDefaultPlane()

    if (detectionScope == 'Whole image') {
        def roi = ROIs.createRectangleROI(0, 0, server.getWidth(), server.getHeight(), plane)
        return [PathObjects.createAnnotationObject(roi)]
    }

    def annotations = imageData.getHierarchy().getAnnotationObjects()
    if (detectionScope == 'All annotations')
        return annotations

    def wanted = (annotationClassNames ?: []).collect { it.toString().trim() }
    return annotations.findAll { a -> a != null && a.getPathClass() != null && wanted.contains(a.getPathClass().getName()) }
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
            } catch (ignored) {
            }
        }
    } catch (e) {
        logWriter("Warning: failed to compute annotation channel stats: ${e}")
    }
    return statsMap
}

def collectDistanceAnnotationGeometry = { imageData ->
    def hierarchy = imageData.getHierarchy()
    def distanceObjects = []
    distanceObjects.addAll(hierarchy.getAnnotationObjects().findAll { a -> a?.getPathClass()?.getName() == distanceAnnotationClassName })
    distanceObjects.addAll(hierarchy.getDetectionObjects().findAll { d -> d?.getPathClass()?.getName() == distanceAnnotationClassName })
    def geometries = distanceObjects.collect { it.getROI()?.getGeometry() }.findAll { it != null }
    if (geometries.isEmpty())
        return null
    def combined = geometries[0]
    for (int i = 1; i < geometries.size(); i++)
        combined = combined.union(geometries[i])
    return combined
}

def findContainingAnnotation = { imageData, roi, allowedClassNames = null ->
    if (imageData == null || roi == null)
        return null

    def point = roi.getGeometry()?.getCentroid()
    if (point == null)
        return null

    def annotations = imageData.getHierarchy().getAnnotationObjects()
    def allowed = (allowedClassNames ?: []).collect { it.toString().trim() }.findAll { it }
    return annotations.find { annotation ->
        def pathClassName = annotation?.getPathClass()?.getName()
        if (!allowed.isEmpty() && !allowed.contains(pathClassName))
            return false
        if (pathClassName == distanceAnnotationClassName)
            return false
        def annotationGeometry = annotation?.getROI()?.getGeometry()
        annotationGeometry != null && (annotationGeometry.contains(point) || annotationGeometry.touches(point))
    }
}

def signedDistanceToGeometry = { Geometry target, Geometry point ->
    if (target == null || point == null)
        return Double.NaN
    def inside = target.contains(point) || target.touches(point)
    if (inside)
        return -target.getBoundary().distance(point)
    return target.distance(point)
}

def appendSpotRows = { imageData, imageName, outRows, Geometry distanceGeometry = null ->
    if (distanceGeometry == null)
        distanceGeometry = collectDistanceAnnotationGeometry(imageData)
    def server = imageData.getServer()
    def pixelSize = 1.0
    try {
        def pixelCalibration = server.getPixelCalibration()
        if (pixelCalibration != null) {
            pixelSize = pixelCalibration.getAveragedPixelSizeMicrons()
            logWriter("[PIXEL CALIBRATION] width=${pixelCalibration.getPixelWidthMicrons()} height=${pixelCalibration.getPixelHeightMicrons()} averaged=${pixelSize}")
        } else {
            pixelSize = server.getAveragedPixelSizeMicrons()
            logWriter("[PIXEL CALIBRATION] fallback server averaged=${pixelSize}")
        }
    } catch (ignored) {
        logWriter('Warning: no pixel calibration found; micron values will match pixel values.')
    }

    def channelNames = [:]
    try {
        def meta = server.getMetadata()
        def metaChannels = meta?.getChannels()
        if (metaChannels) {
            for (int ci = 0; ci < metaChannels.size(); ci++)
                channelNames[ci + 1] = metaChannels[ci].getName() ?: "Channel_${ci + 1}"
        }
    } catch (ignored) {
    }

    def annotationStatsCache = [:]
    def detections = imageData.getHierarchy().getDetectionObjects()
    def geometryFactory = new GeometryFactory()
    int pointGeometryFallbackCount = 0
    def debugDistanceSamples = []
    detections.each { d ->
        def roi = d.getROI()
        if (roi == null)
            return

        def parent = findContainingAnnotation(imageData, roi, annotationClassNames)

        def annotationId = parent?.getID()
        def annotationClassification = parent?.getPathClass()?.getName()
        def annotationChannelStats = [:]

        if (measureChannelIntensity && parent != null) {
            def cacheKey = annotationId ?: parent
            if (!annotationStatsCache.containsKey(cacheKey))
                annotationStatsCache[cacheKey] = computeAnnotationChannelStats(parent, server, channelNames)
            annotationChannelStats = annotationStatsCache[cacheKey] ?: [:]
        }

        def x_px = roi.getCentroidX()
        def y_px = roi.getCentroidY()
        def x_um = Double.isFinite(x_px) ? x_px * pixelSize : Double.NaN
        def y_um = Double.isFinite(y_px) ? y_px * pixelSize : Double.NaN
        def pointGeometry = roi.getGeometry()?.getCentroid()
        if (pointGeometry == null && Double.isFinite(x_px) && Double.isFinite(y_px)) {
            pointGeometry = geometryFactory.createPoint(new Coordinate(x_px, y_px))
            pointGeometryFallbackCount++
        }
        def signedDistancePx = signedDistanceToGeometry(distanceGeometry, pointGeometry)
        def signedDistanceUm = Double.isFinite(signedDistancePx) ? signedDistancePx * pixelSize : Double.NaN
        def channelName = d.getPathClass()?.getName() ?: ''

        if (debugDistanceSamples.size() < 5) {
            debugDistanceSamples << "channel=${channelName}; x_px=${x_px}; y_px=${y_px}; fallback=${pointGeometry == null}; dist_px=${signedDistancePx}; dist_um=${signedDistanceUm}"
        }

        // Per-spot intensity sampling (one value per channel) at centroid
        def spotIntensityMap = [:]
        if (measureChannelIntensity && Double.isFinite(x_px) && Double.isFinite(y_px)) {
            try {
                def sampleX = Math.floor(x_px) as int
                def sampleY = Math.floor(y_px) as int
                def sampleRoi = ROIs.createRectangleROI(sampleX, sampleY, 1, 1, ImagePlane.getDefaultPlane())
                def req = RegionRequest.createInstance(server.getPath(), 1.0, sampleRoi)
                def pathImg = IJTools.convertToImagePlus(server, req)
                def imp = pathImg.getImage()
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
                        def val = proc.getPixelValue(0, 0)
                        def chName = channelNames[ch] ?: "Channel_${ch}"
                        def safeChName = chName.toString().replaceAll('[\\/:*?"<>| ]', '_')
                        spotIntensityMap["intensity_${safeChName}".toString()] = val
                    } catch (ignored) {
                    }
                }
            } catch (ignored) {
            }
        }

        outRows << [
                image_name               : imageName,
                channel                  : channelName,
                annotation_id            : annotationId,
                annotation_classification : annotationClassification,
                x                        : x_px,
                y                        : y_px,
                x_um                     : x_um,
                y_um                     : y_um,
                ("distance_to_${distanceAnnotationClassName}" + '_px'): signedDistancePx,
                ("distance_to_${distanceAnnotationClassName}" + '_um'): signedDistanceUm
        ] + annotationChannelStats + spotIntensityMap

        try {
            def distPx = signedDistancePx ?: Double.NaN
            def distUm = signedDistanceUm ?: Double.NaN
            try { d.getMeasurementList().putMeasurement("distance_to_${distanceAnnotationClassName}_px".toString(), distPx) } catch(e) { d.getMeasurementList().put("distance_to_${distanceAnnotationClassName}_px".toString(), distPx) }
            try { d.getMeasurementList().putMeasurement("distance_to_${distanceAnnotationClassName}_um".toString(), distUm) } catch(e) { d.getMeasurementList().put("distance_to_${distanceAnnotationClassName}_um".toString(), distUm) }
            
            // attach intensity measurements to detection
            spotIntensityMap.each { k, v ->
                def numericValue = (v instanceof Number) ? ((Number) v).doubleValue() : null
                if (numericValue != null) {
                    try { d.getMeasurementList().putMeasurement(k.toString(), numericValue) } catch(e) { d.getMeasurementList().put(k.toString(), numericValue) }
                }
            }
        } catch (e) {
            logWriter("[MEASUREMENT ERROR] ${e}")
        }
    }

    logWriter("[DISTANCE POINT FALLBACK] used=${pointGeometryFallbackCount} detections=${detections.size()}")
    if (!debugDistanceSamples.isEmpty())
        logWriter("[DISTANCE SAMPLES] ${debugDistanceSamples.join(' | ')}")
    try {
        fireHierarchyUpdate()
    } catch (ignored) {
    }
}

def logGeometrySummary = { imageData ->
    def hierarchy = imageData.getHierarchy()
    def annotations = hierarchy.getAnnotationObjects()
    def detections = hierarchy.getDetectionObjects()
    def distanceObjects = []
    distanceObjects.addAll(annotations.findAll { a -> a?.getPathClass()?.getName() == distanceAnnotationClassName })
    distanceObjects.addAll(detections.findAll { d -> d?.getPathClass()?.getName() == distanceAnnotationClassName })
    def spotAnnotations = annotations.findAll { a ->
        def name = a?.getPathClass()?.getName()
        name != null && (annotationClassNames ?: []).collect { it.toString().trim() }.contains(name)
    }
    def distanceDetections = detections.findAll { d -> d?.getPathClass()?.getName() == distanceAnnotationClassName }
    logWriter("[ANNOTATION COUNTS] spot=${spotAnnotations.size()} distance=${distanceObjects.size()} distance_annotations=${distanceObjects.size() - distanceDetections.size()} distance_detections=${distanceDetections.size()} total_annotations=${annotations.size()} total_detections=${detections.size()}")
}

def clearDetectionsForCurrentImage = { imageData ->
    def hierarchy = imageData.getHierarchy()
    def detections = new ArrayList<>(hierarchy.getDetectionObjects())
    if (!detections.isEmpty()) {
        hierarchy.removeObjects(detections, true)
        logWriter("[CLEAR DETECTIONS] removed=${detections.size()}")
    } else {
        logWriter('[CLEAR DETECTIONS] removed=0')
    }
}

def writeCsv = { rows, csvPath, boolean appendMode = false ->
    def outputFile = new File(csvPath)
    def parent = outputFile.parentFile
    if (parent != null)
        parent.mkdirs()

    def defaultHeaders = ['image_name', 'channel', 'annotation_id', 'annotation_classification', 'x', 'y', 'x_um', 'y_um', "distance_to_${distanceAnnotationClassName}_px", "distance_to_${distanceAnnotationClassName}_um"]
    def finalHeaders = defaultHeaders
    if (!rows.isEmpty()) {
        def headers = rows.collectMany { it.keySet().collect { it.toString() } }.unique()
        finalHeaders = defaultHeaders.collect { it.toString() }.findAll { headers.contains(it) } + (headers - defaultHeaders.collect { it.toString() })
    }

    if (appendMode && outputFile.isFile() && outputFile.length() > 0L) {
        try {
            def firstLine = outputFile.withReader('UTF-8') { r -> r.readLine() }
            if (firstLine != null && !firstLine.trim().isEmpty())
                finalHeaders = (firstLine.split(',') as List).collect { it.toString() }
        } catch (ignored) {
        }
    }
    // If appending to an existing file and new rows introduce headers not present
    // in the file, rewrite the existing file with merged headers and pad old rows.
    if (appendMode && outputFile.isFile() && outputFile.length() > 0L) {
        try {
            def existingFirstLine = outputFile.withReader('UTF-8') { r -> r.readLine() }
            if (existingFirstLine != null && !existingFirstLine.trim().isEmpty()) {
                def existingHeaders = (existingFirstLine.split(',') as List).collect { it.toString() }
                def newHeaders = rows.collectMany { it.keySet().collect { it.toString() } }.unique()
                def mergedHeaders = (existingHeaders + (newHeaders - existingHeaders))
                if (mergedHeaders.size() != existingHeaders.size()) {
                    // need to rewrite file with merged headers
                    def oldLines = outputFile.withReader('UTF-8') { r -> r.readLines() }
                    def temp = File.createTempFile(outputFile.name + '_merged_', '.csv', outputFile.parentFile)
                    temp.withWriter('UTF-8') { w ->
                        w.println mergedHeaders.join(',')
                        if (oldLines.size() > 1) {
                            def oldHeader = (oldLines[0].split(',') as List).collect { it.toString() }
                            def missing = mergedHeaders - oldHeader
                            oldLines.drop(1).each { line ->
                                def extra = missing.collect { '' }.join(',')
                                w.println line + (extra ? ',' + extra : '')
                            }
                        }
                    }
                    // Replace original file with merged version
                    if (!outputFile.delete())
                        throw new IOException("Failed to replace existing CSV: ${outputFile}")
                    if (!temp.renameTo(outputFile))
                        throw new IOException("Failed to move merged CSV into place: ${temp}")
                    finalHeaders = mergedHeaders
                } else {
                    finalHeaders = existingHeaders
                }
            }
        } catch (e) {
            // On any failure, fall back to naive append behavior
            logWriter("Warning: failed to merge CSV headers: ${e}")
        }
    }

    def fos = new FileOutputStream(outputFile, appendMode && outputFile.exists() && outputFile.length() > 0L)
    def osw = new OutputStreamWriter(fos, 'UTF-8')
    def writer = new PrintWriter(osw)
    try {
        if (!(appendMode && outputFile.isFile() && outputFile.length() > 0L))
            writer.println finalHeaders.join(',')

        rows.each { row ->
            def values = finalHeaders.collect { header ->
                def key = header.toString()
                def value = row.containsKey(key) ? row.get(key) : null
                if (value == null)
                    return ''
                def text = value.toString()
                if (text.contains(',') || text.contains('"') || text.contains('\n'))
                    return '"' + text.replace('"', '""') + '"'
                return text
            }
            writer.println values.join(',')
        }
    } finally {
        writer.flush()
        writer.close()
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
    logWriter('Warning: no project entry or last saved path available; image data was not saved.')
}

def applyNucPixelClassifier = { imageData ->
    if (imageData == null)
        throw new IllegalStateException('No image is open.')
    if (imageData.getHierarchy() == null)
        throw new IllegalStateException('No hierarchy is available for the current image.')

    def hierarchy = imageData.getHierarchy()
    def beforeAnnotations = new LinkedHashSet(hierarchy.getAnnotationObjects())
    def beforeDetections = new LinkedHashSet(hierarchy.getDetectionObjects())
    def existingNuc = new LinkedHashSet()
    existingNuc.addAll(hierarchy.getAnnotationObjects().findAll { obj -> obj?.getPathClass()?.getName() == nucObjectClassName })
    existingNuc.addAll(hierarchy.getDetectionObjects().findAll { obj -> obj?.getPathClass()?.getName() == nucObjectClassName })
    if (!existingNuc.isEmpty()) {
        hierarchy.removeObjects(new ArrayList<>(existingNuc), true)
        logWriter("Removed ${existingNuc.size()} existing '${nucObjectClassName}' object(s)")
    }

    logWriter("[NUC CLASSIFIER] inline classifier; classifier=${classifierName}; minAreaUm2=${minAreaUm2}; minHoleAreaUm2=${minHoleAreaUm2}")
    try {
        hierarchy.getSelectionModel().clearSelection()
        def beforeTotalAnnotations = hierarchy.getAnnotationObjects().size()
        def beforeTotalDetections = hierarchy.getDetectionObjects().size()
        createAnnotationsFromPixelClassifier(classifierName, minAreaUm2, minHoleAreaUm2)
        def afterTotalAnnotations = hierarchy.getAnnotationObjects().size()
        def afterTotalDetections = hierarchy.getDetectionObjects().size()
        logWriter("[NUC CLASSIFIER RESULT] annotations_before=${beforeTotalAnnotations} annotations_after=${afterTotalAnnotations} detections_before=${beforeTotalDetections} detections_after=${afterTotalDetections}")
    } catch (e) {
        logWriter("[NUC CLASSIFIER ERROR] ${e}")
    }

    def nucClass = getPathClass(nucObjectClassName)
    def newAnnotations = hierarchy.getAnnotationObjects().findAll { !beforeAnnotations.contains(it) }
    def newDetections = hierarchy.getDetectionObjects().findAll { !beforeDetections.contains(it) }
    def created = (newAnnotations + newDetections).unique()
    created.each { obj ->
        try {
            def current = obj.getPathClass()
            if (nucClass != null && (current == null || current.getName() != nucObjectClassName))
                obj.setPathClass(nucClass)
        } catch (ignored) {
        }
    }

    if (saveImageDataAfterClassifier)
        saveImageDataForImage(imageData)

    logWriter("[NUC OBJECTS] annotations=${newAnnotations.size()} detections=${newDetections.size()} total=${created.size()} class=${nucObjectClassName}")
}

// ---------------------------
// Main
// ---------------------------
def imageData = getCurrentImageData()
if (imageData == null)
    throw new IllegalStateException('No image is open.')

def qupathEntry = binding.hasVariable('entry') ? binding.getVariable('entry') : getProjectEntry()
def imageName = qupathEntry?.getImageName() ?: imageData.getServer()?.getMetadata()?.getName() ?: 'image'
def imageStem = getImageStem(imageData, qupathEntry)

logImageHeader(imageName, imageStem, outputCsvPath, logFilePath)
logWriter("Processing: ${imageName}")
logWriter("[CSV MODE] appendCsv=${appendCsv}; clearCsvAtRunStart=${clearCsvAtRunStart}; existedBeforeRun=${outputCsvFile.exists()}; sizeBeforeRun=${outputCsvFile.exists() ? outputCsvFile.length() : 0L}")

def annotationFile = new File(annotationInputDirectoryPath, imageStem + '.geojson')
if (annotationFile.isFile()) {
    def loaded = PathIO.readObjects(annotationFile) ?: []
    def toLoad = (annotationClassNamesToLoad ?: []).isEmpty() ? loaded.findAll { it.isAnnotation() } : loaded.findAll { it.isAnnotation() && it.getPathClass() != null && annotationClassNamesToLoad.contains(it.getPathClass().getName()) }
    if (toLoad && !toLoad.isEmpty()) {
        if (clearExistingAnnotationsBeforeLoad)
            imageData.getHierarchy().removeObjects(new ArrayList<>(imageData.getHierarchy().getAnnotationObjects()), true)
        imageData.getHierarchy().addObjects(toLoad)
        logWriter("[IMPORTED ANNOTATIONS] ${toLoad.size()} from ${annotationFile.name}")
    } else {
        logWriter("[IMPORTED ANNOTATIONS] 0 from ${annotationFile.name}")
    }
} else {
    logWriter("[MISSING ANNOTATION FILE] ${annotationFile.absolutePath}")
}

logGeometrySummary(imageData)

saveImageDataForImage(imageData)

logWriter("[APPLYING CLASSIFIER] ${classifierName}")
applyNucPixelClassifier(imageData)
logGeometrySummary(imageData)
saveImageDataForImage(imageData)

def distanceGeometrySnapshot = collectDistanceAnnotationGeometry(imageData)
if (distanceGeometrySnapshot != null) {
    def envelope = distanceGeometrySnapshot.getEnvelopeInternal()
    logWriter("[DISTANCE GEOMETRY SNAPSHOT] captured; type=${distanceGeometrySnapshot.getGeometryType()}; parts=${distanceGeometrySnapshot.getNumGeometries()}; area=${distanceGeometrySnapshot.getArea()}; bounds=[${envelope.getMinX()}, ${envelope.getMinY()} -> ${envelope.getMaxX()}, ${envelope.getMaxY()}]")
} else {
    logWriter('[DISTANCE GEOMETRY SNAPSHOT] missing')
}

if (clearDetectionsBeforeSpotiflow)
    clearDetectionsForCurrentImage(imageData)

logWriter('[DETECTING SPOTS]')
def spotiflow = Spotiflow.builder().setClassChannelName().nThreads(20).channels(*channels).cleanTempDir().clearChildObjectsBelongingToCurrentChannels().build()
def pathObjects = buildPathObjectsForImage(imageData)
logWriter("[SPOTIFLOW PATH OBJECTS] ${pathObjects?.size() ?: 0}")
spotiflow.detectObjects(imageData, qupathEntry?.getImageName() ?: imageData.getServer()?.getPath(), pathObjects)

def rows = []
appendSpotRows(imageData, imageName, rows, distanceGeometrySnapshot)
// Normalize map keys to plain strings to avoid GString/map-key mismatches
rows = rows.collect { r ->
    if (!(r instanceof Map)) return r
    def m = [:]
    r.each { k, v -> m[k.toString()] = v }
    return m
}
// Debug: show raw row samples and value types to diagnose distance counting
if (!rows.isEmpty()) {
    def sampleRows = rows.take(5).collect { r ->
        r.collect { k, v ->
            def vdesc = (v == null) ? 'null' : (v.getClass().getName() + ':' + v.toString())
            return "${k}=${vdesc}"
        }.join(', ')
    }
    logWriter("[RAW ROWS SAMPLE] ${sampleRows.join(' | ')}")
} else {
    logWriter('[RAW ROWS SAMPLE] none')
}
def rowsWithMissingMeta = countRowsWithMissingValues(rows, ['annotation_id', 'annotation_classification'])
// Compute finite-distance count with detailed debug (previous counter returned 0 unexpectedly)
def rowsWithFiniteDistances = 0
def rowsWithMissingDistance = 0
def debugFiniteChecks = []
rows.eachWithIndex { row, idx ->
    def px = row["distance_to_${distanceAnnotationClassName}_px"]
    def um = row["distance_to_${distanceAnnotationClassName}_um"]
    def isFinite = false
    def parseFinite = { v ->
        if (v == null) return false
        try {
            def d = (v instanceof Number) ? ((Number) v).doubleValue() : Double.parseDouble(v.toString())
            return !(Double.isNaN(d) || Double.isInfinite(d))
        } catch (ignored) {
            return false
        }
    }
    isFinite = parseFinite(px) || parseFinite(um)
    if (isFinite) rowsWithFiniteDistances++
    else rowsWithMissingDistance++
    if (idx < 5) {
        debugFiniteChecks << "idx=${idx}; px=${px} (${px?.getClass()?.getName()}); um=${um} (${um?.getClass()?.getName()}); finite=${isFinite}"
    }
}
if (!debugFiniteChecks.isEmpty()) logWriter("[FINITE CHECKS SAMPLE] ${debugFiniteChecks.join(' | ')}")
logWriter("[SPOTS FOUND] ${rows.size()} in ${imageName}")
logWriter("[ROW SUMMARY] missing_annotation_meta=${rowsWithMissingMeta}; finite_distance_rows=${rowsWithFiniteDistances}; missing_distance_rows=${rowsWithMissingDistance}")
if (rowsWithMissingDistance > 0) {
    def sampleMissing = rows.findAll { row ->
        def px = row["distance_to_${distanceAnnotationClassName}_px"]
        def um = row["distance_to_${distanceAnnotationClassName}_um"]
        !((px instanceof Number && Double.isFinite(px.doubleValue())) || (um instanceof Number && Double.isFinite(um.doubleValue())))
    }.take(5)
    logWriter("[MISSING DISTANCE SAMPLE] ${sampleMissing.collect { it.image_name + ':' + it.channel }.join(', ')}")
}
writeCsv(rows, outputCsvPath, appendCsv)
logWriter("[SAVING] ${outputCsvPath}")

if (save_detections) {
    saveImageDataForImage(imageData)
    logWriter("[IMAGE SAVED] ${imageName}")
}

logWriter('advanced_spotiflow complete')