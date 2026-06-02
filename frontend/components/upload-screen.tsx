"use client"

import { useState, useCallback } from "react"
import { Upload, Link, Twitter, Facebook, Instagram, Youtube, ArrowLeft, CheckCircle, Loader2, Shield, Lock, Clock, ChevronDown, ChevronUp, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { analyzeFile, type AnalyzeResponse } from "../src/api"

export interface SourceInfo {
  type: "file" | "url"
  platform?: string
  username?: string
  originalUrl?: string
  verified: boolean
  previewUrl?: string
  mediaType?: "image" | "video"
}

interface UploadScreenProps {
  mode: "file" | "url"
  initialUrl?: string
  onBack: () => void
  onAnalyze: (sourceInfo: SourceInfo, data?: AnalyzeResponse) => void
}

type ProcessingStep = {
  label: string
  icon: typeof Shield
  completed: boolean
}

export function UploadScreen({ mode, initialUrl = "", onBack, onAnalyze }: UploadScreenProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState(initialUrl)
  const [simulateSafeContent, setSimulateSafeContent] = useState(false)
  const [showSafeResult, setShowSafeResult] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [showSourceFields, setShowSourceFields] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Source traceability fields
  const [platform, setPlatform] = useState("")
  const [username, setUsername] = useState("")
  const [originalUrl, setOriginalUrl] = useState("")
  
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([
    { label: "Securing evidence...", icon: Lock, completed: false },
    { label: "SHA-256 Hash Generated", icon: Shield, completed: false },
    { label: "Timestamp Recorded", icon: Clock, completed: false },
    { label: "Evidence Locked", icon: CheckCircle, completed: false },
  ])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const droppedFile = e.dataTransfer.files[0]

    if (!droppedFile) return

    // 🚫 File size limit (20MB)
    if (droppedFile.size > 20 * 1024 * 1024) {
      alert("Please upload file under 20MB for smooth processing")
      return
    }

    setFile(droppedFile)
    setShowSourceFields(true)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]

    if (!selectedFile) return

    // 🚫 File size limit (20MB)
    if (selectedFile.size > 20 * 1024 * 1024) {
      alert("Please upload file under 20MB for smooth processing")
      return
    }

    setFile(selectedFile)
    setShowSourceFields(true)
  }

  const detectPlatformFromUrl = (inputUrl: string): { platform: string; username: string } => {
    try {
      const urlObj = new URL(inputUrl)
      const hostname = urlObj.hostname.toLowerCase()
      
      if (hostname.includes("twitter.com") || hostname.includes("x.com")) {
        const match = urlObj.pathname.match(/^\/([^/]+)/)
        return { platform: "X", username: match?.[1] || "" }
      }
      if (hostname.includes("instagram.com")) {
        const match = urlObj.pathname.match(/^\/([^/]+)/)
        return { platform: "Instagram", username: match?.[1] || "" }
      }
      if (hostname.includes("youtube.com") || hostname.includes("youtu.be")) {
        return { platform: "YouTube", username: "" }
      }
      if (hostname.includes("facebook.com") || hostname.includes("fb.com")) {
        return { platform: "Facebook", username: "" }
      }
      if (hostname.includes("whatsapp.com")) {
        return { platform: "WhatsApp", username: "" }
      }
      return { platform: "Other", username: "" }
    } catch {
      return { platform: "", username: "" }
    }
  }

  const startProcessing = async () => {
    setIsProcessing(true)
    setError(null)

    if (simulateSafeContent) {
      await new Promise(resolve => setTimeout(resolve, 800))
      setIsProcessing(false)
      setShowSafeResult(true)
      return
    }
    
    // Simulate processing steps
    for (let i = 0; i < processingSteps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 800))
      setProcessingSteps(prev => 
        prev.map((step, index) => 
          index === i ? { ...step, completed: true } : step
        )
      )
    }
    
    await new Promise(resolve => setTimeout(resolve, 500))
    
    // Call API if file mode and file exists
    if (mode === "file" && file) {
      setLoading(true)

      try {
        // Show warning for large files
        const fileSizeMB = file.size / 1024 / 1024;
        if (fileSizeMB > 100) {
          console.warn(`Large file detected (${fileSizeMB.toFixed(2)}MB). Processing may take a while...`);
        }

        const data = await analyzeFile(file)

        // Pass result forward
        const sourceInfo: SourceInfo = {
          type: "file",
          verified: false,
          previewUrl: URL.createObjectURL(file),
          mediaType: file.type.startsWith("video/") ? "video" : "image",
        }
        onAnalyze(sourceInfo, data)

      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Unknown error occurred";
        console.error("Analysis error:", err)
        setError(errorMsg)
        alert(`Error analyzing file: ${errorMsg}`)
        setIsProcessing(false)
      } finally {
        setLoading(false)
      }

      return
    }
    
    // Build source info for URL mode
    let sourceInfo: SourceInfo
    
    if (mode === "url") {
      const detected = detectPlatformFromUrl(url)
      sourceInfo = {
        type: "url",
        platform: detected.platform,
        username: detected.username,
        originalUrl: url,
        verified: true
      }
      
      try {
        setLoading(true);
        // Special handling for VibeStream demo links to extract the actual image
        let fetchUrl = url;
        if (url.includes('/post/')) {
          const postId = url.split('/').pop();
          let ext = '.png';
          if (postId === '5') ext = '.jpg';
          if (postId === '6') ext = '.jpeg';
          // Convert http://localhost:4000/post/1 to http://localhost:4000/images/1.png
          fetchUrl = url.replace(`/post/${postId}`, `/images/${postId}${ext}`);
        }
        
        const response = await fetch(fetchUrl);
        if (!response.ok) throw new Error("Failed to fetch image from URL");
        
        const blob = await response.blob();
        const filename = fetchUrl.split('/').pop() || 'downloaded_media';
        const fetchedFile = new File([blob], filename, { type: blob.type || 'image/png' });
        
        sourceInfo.previewUrl = URL.createObjectURL(fetchedFile);
        sourceInfo.mediaType = fetchedFile.type.startsWith("video/") ? "video" : "image";
        
        // Send to backend for analysis
        const data = await analyzeFile(fetchedFile);
        
        setIsProcessing(false);
        setLoading(false);
        onAnalyze(sourceInfo, data);
        return;
      } catch (err) {
        console.error("Failed to analyze URL:", err);
        alert(`Error analyzing URL: ${err instanceof Error ? err.message : "Unknown error"}`);
        setIsProcessing(false);
        setLoading(false);
        return;
      }
    } else {
      sourceInfo = {
        type: "file",
        platform: platform || undefined,
        username: username || undefined,
        originalUrl: originalUrl || undefined,
        verified: false
      }
    }
    
    setIsProcessing(false)
    onAnalyze(sourceInfo)
  }

  const canProceed = mode === "file" ? file !== null : url.trim().length > 0

  const socialPlatforms = [
    { icon: Twitter, label: "Twitter/X" },
    { icon: Facebook, label: "Facebook" },
    { icon: Instagram, label: "Instagram" },
    { icon: Youtube, label: "YouTube" },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-border px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={onBack}>
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Shield className="w-5 h-5 text-primary" />
            </div>
            <span className="font-semibold">BharatShield</span>
          </div>
        </div>
      </header>

      <main className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-2xl">
          {showSafeResult ? (
            <Card className="bg-card border-border animate-in fade-in slide-in-from-bottom-4 duration-500">
              <CardContent className="p-12 text-center space-y-6">
                <div className="mx-auto w-20 h-20 bg-verdict-real/10 rounded-full flex items-center justify-center mb-2">
                  <CheckCircle className="w-10 h-10 text-verdict-real" />
                </div>
                <h2 className="text-2xl font-bold text-verdict-real">No harmful or misleading content detected</h2>
                <p className="text-lg text-foreground">Further analysis not required</p>
                <div className="inline-block p-4 bg-secondary/50 rounded-lg border border-border mt-4">
                  <p className="text-sm text-muted-foreground">
                    Skipped full analysis via pre-filtering layer (demo simulation)
                  </p>
                </div>
                <div className="pt-6">
                  <Button 
                    variant="outline" 
                    onClick={() => {
                      setShowSafeResult(false)
                      setSimulateSafeContent(false)
                    }}
                    className="min-w-32"
                  >
                    Upload Another
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : !isProcessing && !loading ? (
            <Card className="bg-card border-border">
              <CardHeader className="text-center">
                <CardTitle className="text-2xl">
                  {mode === "file" ? "Upload Suspicious Media" : "Submit URL for Analysis"}
                </CardTitle>
                <CardDescription>
                  {mode === "file" 
                    ? "Drag and drop an image or video file, or click to browse"
                    : "Paste a link to the suspicious content"
                  }
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {mode === "file" ? (
                  <>
                    <div
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      className={`
                        relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200
                        ${isDragging 
                          ? "border-primary bg-primary/5" 
                          : file 
                            ? "border-verdict-real bg-verdict-real/5"
                            : "border-border hover:border-muted-foreground"
                        }
                      `}
                    >
                      <input
                        type="file"
                        accept="image/*,video/*"
                        onChange={handleFileSelect}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      />
                      {file ? (
                        <div className="space-y-3">
                          <div className="p-3 bg-verdict-real/10 rounded-full w-fit mx-auto">
                            <CheckCircle className="w-8 h-8 text-verdict-real" />
                          </div>
                          <p className="font-medium">{file.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                          </p>
                        </div>
                      ) : (
                        <div className="space-y-3">
                          <div className="p-3 bg-secondary rounded-full w-fit mx-auto">
                            <Upload className="w-8 h-8 text-muted-foreground" />
                          </div>
                          <p className="font-medium">Drop your file here</p>
                          <p className="text-sm text-muted-foreground">
                            Supports images and videos up to 500MB
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Source Info Fields - shown after file selection */}
                    {file && (
                      <div className="space-y-4 animate-in fade-in slide-in-from-top-2 duration-300">
                        {/* Info Box */}
                        <div className="flex items-start gap-3 p-4 bg-verdict-uncertain/10 border border-verdict-uncertain/30 rounded-lg">
                          <Info className="w-5 h-5 text-verdict-uncertain flex-shrink-0 mt-0.5" />
                          <div>
                            <p className="text-sm font-medium text-foreground">Source information not detected</p>
                            <p className="text-xs text-muted-foreground mt-1">
                              These details help improve traceability (optional)
                            </p>
                          </div>
                        </div>

                        {/* Collapsible Fields */}
                        <button
                          type="button"
                          onClick={() => setShowSourceFields(!showSourceFields)}
                          className="flex items-center justify-between w-full p-3 bg-secondary/50 rounded-lg hover:bg-secondary transition-colors"
                        >
                          <span className="text-sm text-muted-foreground">Add source details</span>
                          {showSourceFields ? (
                            <ChevronUp className="w-4 h-4 text-muted-foreground" />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-muted-foreground" />
                          )}
                        </button>

                        {showSourceFields && (
                          <div className="space-y-4 p-4 bg-secondary/30 rounded-lg animate-in fade-in slide-in-from-top-2 duration-200">
                            <div className="space-y-2">
                              <label className="text-sm text-muted-foreground">Platform</label>
                              <Select value={platform} onValueChange={setPlatform}>
                                <SelectTrigger className="bg-input border-border">
                                  <SelectValue placeholder="Select platform" />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="X">X (Twitter)</SelectItem>
                                  <SelectItem value="Instagram">Instagram</SelectItem>
                                  <SelectItem value="YouTube">YouTube</SelectItem>
                                  <SelectItem value="WhatsApp">WhatsApp</SelectItem>
                                  <SelectItem value="Facebook">Facebook</SelectItem>
                                  <SelectItem value="Other">Other</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>

                            <div className="space-y-2">
                              <label className="text-sm text-muted-foreground">Username / Handle</label>
                              <Input
                                type="text"
                                placeholder="@username"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="bg-input border-border"
                              />
                            </div>

                            <div className="space-y-2">
                              <label className="text-sm text-muted-foreground">Original link (if available)</label>
                              <Input
                                type="url"
                                placeholder="https://..."
                                value={originalUrl}
                                onChange={(e) => setOriginalUrl(e.target.value)}
                                className="bg-input border-border"
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Helper text for traceability */}
                    <p className="text-xs text-muted-foreground text-center">
                      For better traceability, we recommend submitting via URL or direct share.
                    </p>
                  </>
                ) : (
                  <>
                    <div className="space-y-4">
                      {initialUrl && (
                        <div className="flex items-center gap-2 text-sm text-primary bg-primary/10 p-2 rounded-md border border-primary/20 animate-in fade-in duration-200">
                          <Link className="w-4 h-4" />
                          <span className="font-medium">URL received from external platform</span>
                        </div>
                      )}
                      <div className="flex gap-3">
                        <div className="flex-1 relative">
                          <Link className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                          <Input
                            type="url"
                            placeholder="https://example.com/video..."
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            className="pl-10 bg-input border-border h-12"
                          />
                        </div>
                      </div>

                      {url && !initialUrl && (
                        <div className="flex items-center gap-2 text-sm text-verdict-real animate-in fade-in duration-200">
                          <CheckCircle className="w-4 h-4" />
                          <span>URL-based submission enables automatic source verification</span>
                        </div>
                      )}

                      <div className="space-y-3">
                        <p className="text-sm text-muted-foreground text-center">Or share from social media</p>
                        <div className="flex justify-center gap-3">
                          {socialPlatforms.map((platform) => (
                            <Button
                              key={platform.label}
                              variant="outline"
                              size="icon"
                              className="w-12 h-12 rounded-xl"
                              title={platform.label}
                            >
                              <platform.icon className="w-5 h-5" />
                            </Button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </>
                )}

                <Button 
                  onClick={startProcessing}
                  disabled={!canProceed || loading || isProcessing}
                  className="w-full h-12 bg-primary hover:bg-primary/90"
                >
                  {loading || isProcessing ? "Processing..." : "Begin Analysis"}
                </Button>

                <div className="flex items-center justify-center pt-2">
                  <button
                    type="button"
                    onClick={() => setSimulateSafeContent(!simulateSafeContent)}
                    className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors group"
                  >
                    <div className={`w-3.5 h-3.5 rounded-sm border flex items-center justify-center transition-colors ${simulateSafeContent ? 'bg-primary border-primary text-primary-foreground' : 'border-muted-foreground bg-transparent group-hover:border-foreground'}`}>
                      {simulateSafeContent && (
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="w-2.5 h-2.5">
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                      )}
                    </div>
                    Demo: Simulate Pre-Filter (Safe Content)
                  </button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="bg-card border-border">
              <CardContent className="p-12">
                <div className="text-center space-y-8">
                  {/* Processing Animation */}
                  <div className="relative w-24 h-24 mx-auto">
                    <div className="absolute inset-0 border-4 border-primary/20 rounded-full" />
                    <div className="absolute inset-0 border-4 border-transparent border-t-primary rounded-full animate-spin" />
                    <div className="absolute inset-4 bg-primary/10 rounded-full flex items-center justify-center">
                      <Shield className="w-8 h-8 text-primary" />
                    </div>
                  </div>

                  <div>
                    <h2 className="text-xl font-semibold mb-2">Processing Evidence</h2>
                    <p className="text-muted-foreground">Creating tamper-proof record</p>
                  </div>

                  {/* Processing Steps */}
                  <div className="space-y-4 max-w-sm mx-auto text-left">
                    {processingSteps.map((step, index) => (
                      <div 
                        key={step.label}
                        className={`flex items-center gap-4 transition-all duration-300 ${
                          step.completed ? "opacity-100" : "opacity-40"
                        }`}
                      >
                        <div className={`p-2 rounded-lg ${
                          step.completed ? "bg-verdict-real/10" : "bg-secondary"
                        }`}>
                          {step.completed ? (
                            <CheckCircle className="w-5 h-5 text-verdict-real" />
                          ) : index === processingSteps.findIndex(s => !s.completed) ? (
                            <Loader2 className="w-5 h-5 text-primary animate-spin" />
                          ) : (
                            <step.icon className="w-5 h-5 text-muted-foreground" />
                          )}
                        </div>
                        <span className={step.completed ? "text-foreground" : "text-muted-foreground"}>
                          {step.label}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  )
}
