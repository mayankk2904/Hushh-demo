"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Camera, X } from "lucide-react"

interface CameraCaptureProps {
  onCapture: (imageUrl: string) => void
  onCancel: () => void
}

export default function CameraCapture({ onCapture, onCancel }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const startCamera = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        })

        setStream(mediaStream)

        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
        }
      } catch (err) {
        console.error("Error accessing camera:", err)
        setError("Could not access camera. Please make sure you've granted camera permissions.")
      }
    }

    startCamera()

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current
      const canvas = canvasRef.current
      const context = canvas.getContext("2d")

      if (context) {
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight

        // Draw the current video frame on the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height)

        // Convert canvas to data URL
        const imageUrl = canvas.toDataURL("image/png")

        // Stop all video tracks
        if (stream) {
          stream.getTracks().forEach((track) => track.stop())
        }

        // Pass the image URL to the parent component
        onCapture(imageUrl)
      }
    }
  }

  return (
    <div className="relative">
      <div className="absolute top-2 right-2 z-10">
        <Button variant="outline" size="icon" className="bg-white/80 rounded-full h-8 w-8" onClick={onCancel}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      {error ? (
        <div className="p-6 text-center bg-red-50 rounded-lg">
          <p className="text-red-600">{error}</p>
          <Button variant="outline" className="mt-4" onClick={onCancel}>
            Go Back
          </Button>
        </div>
      ) : (
        <>
          <div className="rounded-lg overflow-hidden bg-black aspect-video relative">
            <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover" />
          </div>

          <canvas ref={canvasRef} className="hidden" />

          <div className="mt-4 flex justify-center">
            <Button
              className="bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white rounded-full px-6"
              onClick={captureImage}
            >
              <Camera className="w-4 h-4 mr-2" />
              Capture Photo
            </Button>
          </div>
        </>
      )}
    </div>
  )
}

