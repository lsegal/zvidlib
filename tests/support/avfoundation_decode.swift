// Opens an MP4 with AVFoundation and decodes every frame of its video track, for
// `tests/native_hevc_encoder.rs` (issue #495). AVFoundation is stricter about `hvc1`
// sample entries, `hvcC` contents and edit lists than ffmpeg is, so a file that ffmpeg
// reads can still be one QuickTime refuses.
//
// Usage: swift avfoundation_decode.swift <file.mp4>
//
// Prints `key=value` lines describing what AVFoundation saw and exits non-zero if the
// asset cannot be opened or read to the end. Whether what it saw is right is the
// calling test's decision.

import AVFoundation
import CoreMedia
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("avfoundation: \(message)\n".utf8))
    exit(1)
}

func fourCC(_ code: FourCharCode) -> String {
    let bytes = [24, 16, 8, 0].map { UInt8((code >> $0) & 0xff) }
    return String(decoding: bytes, as: UTF8.self)
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: swift avfoundation_decode.swift <file.mp4>")
}
let asset = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))

do {
    print("asset_playable=\(try await asset.load(.isPlayable))")
    print("asset_duration=\(String(format: "%.3f", try await asset.load(.duration).seconds))")
    let tracks = try await asset.loadTracks(withMediaType: .video)
    print("video_tracks=\(tracks.count)")
    guard let track = tracks.first else {
        fail("no video track")
    }
    print("track_playable=\(try await track.load(.isPlayable))")
    print("track_decodable=\(try await track.load(.isDecodable))")
    let size = try await track.load(.naturalSize)
    print("width=\(Int(size.width))")
    print("height=\(Int(size.height))")
    for description in try await track.load(.formatDescriptions) {
        print("codec=\(fourCC(CMFormatDescriptionGetMediaSubType(description)))")
    }
    let timeRange = try await track.load(.timeRange)
    print("track_start=\(String(format: "%.3f", timeRange.start.seconds))")
    print("track_duration=\(String(format: "%.3f", timeRange.duration.seconds))")

    // Decoding to pixel buffers is what makes this a decode rather than a demux: with
    // output settings, every sample goes through VideoToolbox's HEVC decoder.
    let reader = try AVAssetReader(asset: asset)
    let output = AVAssetReaderTrackOutput(
        track: track,
        outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        ])
    output.alwaysCopiesSampleData = false
    reader.add(output)
    guard reader.startReading() else {
        fail("startReading: \(reader.error.map { "\($0)" } ?? "unknown error")")
    }
    var decoded = 0
    while let buffer = output.copyNextSampleBuffer() {
        if CMSampleBufferGetImageBuffer(buffer) != nil {
            decoded += 1
        }
    }
    guard reader.status == .completed else {
        fail("reading stopped after \(decoded) frames: \(reader.error.map { "\($0)" } ?? "status \(reader.status.rawValue)")")
    }
    print("decoded_frames=\(decoded)")
} catch {
    fail("\(error)")
}
