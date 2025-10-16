"use client"

import type React from "react"
import { useState, useRef } from "react"
import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, Camera, ArrowLeft, CheckCircle, X } from "lucide-react"
import CameraCapture from "@/components/camera-capture"

interface TrainingImage {
  url: string
  label: string
  file: File  // Added file property to store the original file
}

export default function TrainPage() {
  const [selectedImages, setSelectedImages] = useState<TrainingImage[]>([])
  const [currentLabel, setCurrentLabel] = useState("")
  const [isUploaded, setIsUploaded] = useState(false)
  const [isUsingCamera, setIsUsingCamera] = useState(false)
  const [error, setError] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files)
      
      // Check if adding these files would exceed reasonable limits
      if (selectedImages.length + files.length > 1000) {
        setError("Maximum 1000 images allowed per upload")
        return
      }
      
      const newImages = files.map(file => ({
        url: URL.createObjectURL(file),
        label: currentLabel,
        file: file  // Store the original file object
      }))
      
      setSelectedImages([...selectedImages, ...newImages])
      setIsUploaded(false)
      setError("")
    }
  }

  const handleCameraCapture = (imageUrl: string) => {
    if (selectedImages.length >= 1000) {
      setError("Maximum 1000 images allowed per upload")
      return
    }
    
    // Create a file object with a default name for camera captures
    const fileName = `camera-capture-${Date.now()}.jpg`
    fetch(imageUrl)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], fileName, { type: "image/jpeg" })
        setSelectedImages([...selectedImages, { 
          url: imageUrl, 
          label: currentLabel,
          file: file 
        }])
        setIsUploaded(false)
        setIsUsingCamera(false)
        setError("")
      })
  }

  const handleUploadClick = async () => {
    if (selectedImages.length < 20) {
      setError("Minimum 30 images are required for training")
      return
    }
    
    if (!currentLabel.trim()) {
      setError("Please enter a valid label")
      return
    }

    const formData = new FormData()
    formData.append("dir_name", currentLabel)

    try {
      // Use the original files with their original names
      for (let i = 0; i < selectedImages.length; i++) {
        formData.append("files", selectedImages[i].file)  // Using the original file
      }

      const res = await fetch("http://localhost:8000/upload-files", {
        method: "POST",
        body: formData,
      })

      if (res.ok) {
        setIsUploaded(true)
        setError("")
        console.log("Upload successful")
      } else {
        setError("Upload failed. Please try again.")
        console.error("Upload failed")
      }
    } catch (error) {
      setError("Error during upload. Please try again.")
      console.error("Error during upload", error)
    }
  }

  const removeImage = (index: number) => {
    const newImages = [...selectedImages]
    newImages.splice(index, 1)
    setSelectedImages(newImages)
    setIsUploaded(false)
    setError("")
  }

  return (
    <div className="min-h-screen bg-[#fad2ad]">
      {/* Header */}
      <header className="bg-[#0f0f1a] py-6">
        <div className="container mx-auto px-4">
          <nav className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Link href="/" className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gray-800 rounded-full flex items-center justify-center text-white">
                  <span className="text-xl"><img src="\1.jpeg" alt="TE" /></span>
                </div>
                <span className="font-bold text-white">TE CONNECTIVITY</span>
              </Link>
            </div>

            <div className="hidden md:flex items-center space-x-8 text-gray-300">
              <Link href="/" className="text-gray-400">
                Top
              </Link>
              <Link href="/upload" className="text-gray-400">
                Predict
              </Link>
              <Link href="/train" className="text-white font-medium">
                Upload Training Images
              </Link>
              <Link href="#" className="text-gray-400">
                Features
              </Link>
            </div>

            <div className="flex items-center gap-4">
              <Link href="#" className="text-gray-400">
                Manufacturer
              </Link>
              <Button className="bg-white text-black hover:bg-gray-100 rounded-full px-6">Login</Button>
            </div>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-2xl mx-auto">
          <Link href="/" className="flex items-center gap-2 text-gray-700 mb-6 hover:text-gray-900">
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>

          <Card className="bg-white shadow-xl rounded-xl overflow-hidden">
            <CardHeader>
              <CardTitle className="text-2xl font-bold text-center">Train Model with New Images</CardTitle>
            </CardHeader>
            <CardContent>
              {isUsingCamera ? (
                <CameraCapture onCapture={handleCameraCapture} onCancel={() => setIsUsingCamera(false)} />
              ) : (
                <div className="space-y-6">
                  {selectedImages.length > 0 ? (
                    <>
                      <div className="grid grid-cols-2 gap-4">
                        {selectedImages.map((image, index) => (
                          <div key={index} className="relative aspect-video rounded-lg overflow-hidden border-2 border-dashed border-gray-300">
                            <Image
                              src={image.url}
                              alt={`Selected image ${index + 1}`}
                              fill
                              className="object-contain"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white p-1 text-xs truncate">
                              {image.file.name}
                            </div>
                            <button
                              onClick={() => removeImage(index)}
                              className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 hover:bg-red-600"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                      <p className={`text-sm text-center ${selectedImages.length < 10 ? 'text-red-500' : 'text-green-600'}`}>
                        {selectedImages.length} of minimum 30 images selected
                      </p>
                    </>
                  ) : (
                    <div
                      className="aspect-video rounded-lg border-2 border-dashed border-gray-300 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className="w-12 h-12 text-gray-400 mb-2" />
                      <p className="text-gray-500">Click to upload images or drag and drop</p>
                      <p className="text-gray-400 text-sm">PNG, JPG, GIF up to 10MB each</p>
                      <p className="text-red-500 text-sm">Minimum 30 Images are required per Part</p>
                    </div>
                  )}

                  <div className="flex flex-col gap-4">
                    <div className="flex gap-4">
                      <Input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        multiple
                      />
                      <Button variant="outline" className="flex-1" onClick={() => fileInputRef.current?.click()}>
                        <Upload className="w-4 h-4 mr-2" />
                        Select Files
                      </Button>
                      <Button variant="outline" className="flex-1" onClick={() => setIsUsingCamera(true)}>
                        <Camera className="w-4 h-4 mr-2" />
                        Use Camera
                      </Button>
                    </div>

                    <div className="space-y-2">
                      <label htmlFor="label" className="text-sm font-medium text-gray-700">
                        Images Label (TE Part Number)
                      </label>
                      <Input
                        id="label"
                        placeholder="Enter TE part number (e.g., '1234567-H')"
                        value={currentLabel}
                        onChange={(e) => {
                          setCurrentLabel(e.target.value)
                          setError("")
                        }}
                      />
                    </div>

                    {error && (
                      <p className="text-red-500 text-sm text-center">{error}</p>
                    )}

                    {selectedImages.length > 0 && (
                      <Button
                        className="bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white"
                        onClick={handleUploadClick}
                        disabled={!currentLabel.trim() || isUploaded}
                      >
                        {isUploaded ? (
                          <>
                            <CheckCircle className="w-4 h-4 mr-2" />
                            Uploaded Successfully
                          </>
                        ) : (
                          `Upload ${selectedImages.length} Image${selectedImages.length > 1 ? 's' : ''} for Training`
                        )}
                      </Button>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
            {isUploaded && !isUsingCamera && (
              <CardFooter className="bg-gray-50 border-t">
                <div className="w-full text-center text-green-600">
                  <p className="font-medium">
                    {selectedImages.length} image{selectedImages.length > 1 ? 's' : ''} successfully uploaded for training with label: <span className="font-bold">{currentLabel}</span>
                  </p>
                </div>
              </CardFooter>
            )}
          </Card>
        </div>
      </main>
    </div>
  )
}