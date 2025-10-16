"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, Camera, ArrowLeft, ChevronDown } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import CameraCapture from "@/components/camera-capture"

interface ModelInfo {
  name: string;
  path: string;
  accuracy: number;
}

export default function UploadPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<{ classes: string[]; probabilities: number[] } | null>(null)
  const [attentionMap, setAttentionMap] = useState<string | null>(null)
  const [explanationText, setExplanationText] = useState<string | null>(null)
  const [isUsingCamera, setIsUsingCamera] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isExplaining, setIsExplaining] = useState(false)
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [isLoadingModels, setIsLoadingModels] = useState(true)
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const fileRef = useRef<File | null>(null)

  // Fetch models when component mounts
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/model-list")
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`)
        }
        const modelData = await response.json()
        
        // Transform the model data into ModelInfo array
        const formattedModels = Object.entries(modelData).map(([path, accuracy]) => {
          // Extract just the model name from the path
          const modelName = path.split('\\').pop()?.replace('.pth', '') || 'Unknown Model'
          return {
            name: modelName,
            path: path, // Store the full path
            accuracy: accuracy as number
          }
        })
        
        setModels(formattedModels)
        if (formattedModels.length > 0) {
          // Load the first model by default
          handleModelChange(formattedModels[0])
        }
        setIsLoadingModels(false)
      } catch (error) {
        console.error("Error fetching models:", error)
        setError("Failed to load model list")
        setIsLoadingModels(false)
      }
    }

    fetchModels()
  }, [])

  const handleModelChange = async (model: ModelInfo) => {
    setIsLoadingModel(true)
    setError(null)
    
    try {
      // Notify backend to load the selected model
      const response = await fetch("http://127.0.0.1:8000/load-model", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_path: model.path
        }),
      })

      if (!response.ok) {
        throw new Error(`Failed to load model: ${response.statusText}`)
      }

      const result = await response.json()
      if (result.success !== true) {
        throw new Error(result.message || "Model loading failed")
      }

      // Only update the selected model if loading was successful
      setSelectedModel(model)
      // Clear previous predictions when model changes
      setPrediction(null)
      setAttentionMap(null)
      setExplanationText(null)
    } catch (err) {
      console.error("Model loading error:", err)
      setError(`Failed to load model: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsLoadingModel(false)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      fileRef.current = file
      const imageUrl = URL.createObjectURL(file)
      setSelectedImage(imageUrl)
      setPrediction(null)
      setError(null)
      setAttentionMap(null)
      setExplanationText(null)
    }
  }

  const handleCameraCapture = (imageUrl: string) => {
    fetch(imageUrl)
      .then((res) => res.blob())
      .then((blob) => {
        const file = new File([blob], "captured_image.jpg", { type: "image/jpeg" })
        fileRef.current = file
        setSelectedImage(imageUrl)
        setPrediction(null)
        setError(null)
        setAttentionMap(null)
        setExplanationText(null)
        setIsUsingCamera(false)
      })
  }

  const handleExplainClick = async () => {
    if (!fileRef.current || !selectedModel) {
      setError("No image selected or no model chosen!")
      return
    }

    setIsExplaining(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", fileRef.current)
    formData.append("model", selectedModel.path)

    try {
      const response = await fetch("http://127.0.0.1:8000/explain", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`)
      }

      const result = await response.json()
      setAttentionMap(`data:image/png;base64,${result.explanation}`)
      if (result.textual_explanation) {
        setExplanationText(result.textual_explanation)
      }
      setError(null)
    } catch (err) {
      console.error("Explanation error:", err)
      setError("Failed to get explanation. Please try again.")
    } finally {
      setIsExplaining(false)
    }
  }

  const handlePredictClick = async () => {
    if (!fileRef.current || !selectedModel) {
      setError("No image selected or no model chosen!")
      return
    }

    const formData = new FormData()
    formData.append("file", fileRef.current)
    formData.append("model", selectedModel.path)

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`)
      }

      const result = await response.json()

      if (result.top_5_predictions && result.top_5_predictions.length > 0) {
        setPrediction({
          classes: result.top_5_predictions.map((p: any) => p.part_number),
          probabilities: result.top_5_predictions.map((p: any) => p.confidence),
        })
        setError(null)
      } else {
        setError("No predictions returned.")
      }
    } catch (err) {
      console.error("Prediction error:", err)
      setError("Failed to get prediction. Please try again.")
    }
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
              <Link href="/" className="text-gray-400">Top</Link>
              <Link href="/upload" className="text-white font-medium">Predict</Link>
              <Link href="/train" className="text-gray-400">Upload Training Images</Link>
              <Link href="#" className="text-gray-400">Features</Link>
            </div>

            <div className="flex items-center gap-4">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button 
                    variant="ghost" 
                    className="text-white hover:bg-orange-600/80 hover:text-white data-[state=open]:bg-orange-600/80 transition-colors duration-200"
                    disabled={isLoadingModel || isLoadingModels}
                  >
                    <span className="mr-2">
                      {isLoadingModel ? "Loading model..." : 
                       isLoadingModels ? "Loading models..." : 
                       selectedModel?.name || "Select a model"}
                    </span>
                    <ChevronDown className="w-4 h-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56 bg-[#0f0f1a] border-gray-700">
                  {models.map((model) => (
                    <DropdownMenuItem
                      key={model.path}
                      className="hover:bg-gray-700 focus:bg-gray-700"
                      onClick={() => handleModelChange(model)}
                      disabled={isLoadingModel}
                    >
                      <div className="flex flex-col">
                        <span className="text-white">{model.name}</span>
                        <span className="text-xs text-gray-400">
                          Accuracy: {model.accuracy.toFixed(1)}%
                        </span>
                      </div>
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
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
              <CardTitle className="text-2xl font-bold text-center">Upload Image for Prediction</CardTitle>
            </CardHeader>
            <CardContent>
              {isUsingCamera ? (
                <CameraCapture onCapture={handleCameraCapture} onCancel={() => setIsUsingCamera(false)} />
              ) : (
                <div className="space-y-6">
                  {selectedImage ? (
                    <div className="relative aspect-video rounded-lg overflow-hidden border-2 border-dashed border-gray-300 flex items-center justify-center">
                      <Image src={selectedImage || "/placeholder.svg"} alt="Selected image" fill className="object-contain" />
                    </div>
                  ) : (
                    <div
                      className="aspect-video rounded-lg border-2 border-dashed border-gray-300 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <Upload className="w-12 h-12 text-gray-400 mb-2" />
                      <p className="text-gray-500">Click to upload an image or drag and drop</p>
                      <p className="text-gray-400 text-sm">PNG, JPG, GIF up to 10MB</p>
                    </div>
                  )}

                  <div className="flex flex-col gap-4">
                    <div className="flex gap-4">
                      <Input type="file" accept="image/*" className="hidden" ref={fileInputRef} onChange={handleFileChange} />
                      <Button variant="outline" className="flex-1" onClick={() => fileInputRef.current?.click()}>
                        <Upload className="w-4 h-4 mr-2" />
                        Select File
                      </Button>
                      <Button variant="outline" className="flex-1" onClick={() => setIsUsingCamera(true)}>
                        <Camera className="w-4 h-4 mr-2" />
                        Use Camera
                      </Button>
                    </div>

                    {selectedImage && (
                      <Button 
                        className="bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white" 
                        onClick={handlePredictClick}
                        disabled={!selectedModel || isLoadingModel}
                      >
                        Predict Label ({selectedModel?.name || "No model selected"})
                      </Button>
                    )}
                  </div>
                  {error && <p className="text-red-500 text-center">{error}</p>}
                </div>
              )}
            </CardContent>

            {prediction && !isUsingCamera && (
              <CardFooter className="bg-gray-50 border-t flex flex-col gap-6">
                {/* Prediction Results */}
                <div className="w-full">
                  <div className="mb-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
                    <h3 className="text-lg font-semibold text-gray-800">
                      Predicted Part Number: 
                      <span className="text-indigo-600 ml-2">{prediction.classes[0]}</span>
                    </h3>
                    <p className="text-gray-600 mt-1">
                      Confidence: {(prediction.probabilities[0] * 100).toFixed(1)}%
                    </p>
                    <p className="text-sm text-gray-500 mt-2">
                      Model used: {selectedModel?.name}
                    </p>
                  </div>

                  <h3 className="font-medium text-gray-700 mb-3">Alternative Predictions:</h3>
                  <div className="p-4 bg-white border rounded-lg overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Part Number
                          </th>
                          <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Confidence
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {prediction.classes.slice(1).map((cls, index) => (
                          <tr key={index}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                              {cls}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-right">
                              {(prediction.probabilities[index + 1] * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Explanation Section */}
                <div className="w-full space-y-4">
                  <div className="flex justify-between items-center">
                    <h3 className="font-medium text-gray-700">Model Explanation</h3>
                    <Button 
                      onClick={handleExplainClick}
                      disabled={isExplaining || isLoadingModel}
                      variant="outline"
                      className="bg-blue-50 hover:bg-blue-100 text-blue-600"
                    >
                      {isExplaining ? 'Generating...' : 'Show Attention Map'}
                    </Button>
                  </div>

                  {attentionMap && (
                    <div className="relative aspect-video rounded-lg overflow-hidden border-2 border-gray-200 bg-gray-50">
                      <img
                        src={attentionMap}
                        alt="Attention heatmap"
                        className="object-contain w-full h-full"
                      />
                      <div className="absolute bottom-2 left-2 bg-black/50 text-white px-2 py-1 rounded text-sm">
                        Heatmap overlay showing model's focus areas
                      </div>
                    </div>
                  )}

                  {explanationText && (
                    <div className="p-4 bg-gray-100 border rounded-lg">
                      <p className="text-gray-700 text-sm">{explanationText}</p>
                    </div>
                  )}
                </div>
              </CardFooter>
            )}
          </Card>
        </div>
      </main>
    </div>
  )
}