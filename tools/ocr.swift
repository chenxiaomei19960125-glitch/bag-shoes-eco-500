
import Foundation
import Vision
import AppKit
let paths = try! String(contentsOfFile: CommandLine.arguments[1]).split(separator: "\n").map(String.init)
var out: [String: String] = [:]
for p in paths {
  guard let img = NSImage(contentsOfFile: p), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }
  let req = VNRecognizeTextRequest()
  req.recognitionLevel = .accurate
  req.recognitionLanguages = ["zh-Hans", "en-US"]
  req.usesLanguageCorrection = true
  try? VNImageRequestHandler(cgImage: cg).perform([req])
  let lines = (req.results ?? []).compactMap { o -> String? in
    guard let c = o.topCandidates(1).first, c.confidence > 0.4 else { return nil }
    return c.string }
  out[p] = lines.joined(separator: " / ")
}
let d = try! JSONSerialization.data(withJSONObject: out, options: [])
FileManager.default.createFile(atPath: CommandLine.arguments[2], contents: d)
