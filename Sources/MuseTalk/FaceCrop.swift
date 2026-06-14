// Vision-based face crop for the MuseTalk talkingHead front-end.
//
// Replaces the upstream DWPose(+S3FD) crop path with Apple Vision: VNDetectFaceLandmarks gives
// both the 76-point face landmarks (primary crop) and a face boundingBox (fallback, in place of
// S3FD). Mirrors musetalk/utils/preprocessing.get_landmark_and_bbox: crop = (minX, upper, maxX,
// maxY) over the face landmarks, with `upper = noseY - (maxY - noseY)` (vertically centered on the
// nose). Validated against the dvisual DWPose golden crops.
#if canImport(Vision)
import CoreGraphics
import Foundation
import Vision

public struct FaceCrop {
    public struct Box: Equatable { public let x1, y1, x2, y2: Int }

    /// MuseTalk crop box from Vision landmarks (image pixel coords, top-left origin). nil if no face.
    public static func crop(cgImage: CGImage) -> Box? {
        let req = VNDetectFaceLandmarksRequest()
        try? VNImageRequestHandler(cgImage: cgImage, options: [:]).perform([req])
        guard let faces = req.results as? [VNFaceObservation], !faces.isEmpty else { return nil }
        // largest face
        let face = faces.max { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }!
        let (w, h) = (cgImage.width, cgImage.height)
        guard let lm = face.landmarks, let all = lm.allPoints else {
            return fallbackBox(face, width: w, height: h)
        }
        // Vision image coords are bottom-left origin → flip y to top-left (cv2 / golden convention).
        func toTL(_ region: VNFaceLandmarkRegion2D) -> [(Double, Double)] {
            region.pointsInImage(imageSize: CGSize(width: w, height: h))
                .map { (Double($0.x), Double(h) - Double($0.y)) }
        }
        let pts = toTL(all)
        let xs = pts.map(\.0), ys = pts.map(\.1)
        let (minX, maxX, maxY) = (xs.min()!, xs.max()!, ys.max()!)
        // nose reference (~iBUG point 29, the nose BRIDGE): noseCrest, not the full nose outline
        // (whose centroid sits ~25px too low at the nostrils → crop top 50px off). Bridge line centroid.
        let nosePts = toTL(lm.noseCrest ?? lm.nose ?? all)
        let noseY = nosePts.map(\.1).reduce(0, +) / Double(nosePts.count)
        let upper = max(0, noseY - (maxY - noseY))
        return Box(x1: Int(minX), y1: Int(upper), x2: Int(maxX), y2: Int(maxY))
    }

    /// Vision face boundingBox -> top-left pixel box (the S3FD-fallback replacement).
    public static func fallbackBox(_ face: VNFaceObservation, width w: Int, height h: Int) -> Box {
        let bb = face.boundingBox   // normalized, bottom-left origin
        return Box(x1: Int(bb.minX * Double(w)), y1: Int((1 - bb.maxY) * Double(h)),
                   x2: Int(bb.maxX * Double(w)), y2: Int((1 - bb.minY) * Double(h)))
    }
}
#endif
