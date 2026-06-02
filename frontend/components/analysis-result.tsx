"use client"

import { AlertTriangle, Shield, ArrowRight, Clock, Image as ImageIcon, Volume2, CheckCircle, AlertCircle, Download, MonitorPlay, Activity } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { SourceInfo } from "./upload-screen"

interface AnalysisResultProps {
  onContinue: () => void
  onBack: () => void
  sourceInfo?: SourceInfo
  analysisData?: any
}

export function AnalysisResult({ onContinue, onBack, sourceInfo, analysisData }: AnalysisResultProps) {
  // Default to mock data if no real data is provided
  const verdictData = analysisData ? {
    verdict: analysisData.deepfake_detection.label === "Fake" ? "SYNTHETIC MEDIA DETECTED" : "AUTHENTIC MEDIA",
    confidence: (analysisData.deepfake_detection.confidence * 100).toFixed(1),
    status: analysisData.deepfake_detection.label === "Fake" ? "fake" : "real" as "fake" | "real" | "uncertain",
  } : {
    verdict: "SYNTHETIC MEDIA DETECTED",
    confidence: "87.0",
    status: "fake" as "fake" | "real" | "uncertain",
  }

  // Determine which media URL to show
  const isVideo = sourceInfo?.mediaType === "video"
  const isFake = verdictData.status === "fake"
  
  // Display GradCAM if it's a fake image and backend provides it
  const displayUrl = (!isVideo && isFake && analysisData?.deepfake_detection?.gradcam_url) 
    ? `http://127.0.0.1:5001${analysisData.deepfake_detection.gradcam_url}`
    : sourceInfo?.previewUrl

  // Map real ML streams to issues
  const detectedIssues: any[] = []
  
  if (analysisData && analysisData.deepfake_detection.streams) {
    const streams = analysisData.deepfake_detection.streams
    
    if (streams.spatial_texture?.fake_prob > 0.5) {
      detectedIssues.push({
        icon: ImageIcon,
        title: "Spatial Inconsistency (CNN)",
        description: `Facial artifacts detected. Probability: ${(streams.spatial_texture.fake_prob * 100).toFixed(1)}%`,
        severity: streams.spatial_texture.fake_prob > 0.8 ? "high" : "medium",
      })
    }
    if (streams.frequency_domain?.fake_prob > 0.5) {
      detectedIssues.push({
        icon: MonitorPlay,
        title: "Frequency Domain Anomalies",
        description: `Invisible high-frequency noise patterns detected. Probability: ${(streams.frequency_domain.fake_prob * 100).toFixed(1)}%`,
        severity: streams.frequency_domain.fake_prob > 0.8 ? "high" : "medium",
      })
    }
    if (streams.temporal?.fake_prob > 0.5) {
      detectedIssues.push({
        icon: Clock,
        title: "Temporal Mismatch (Video)",
        description: `Inter-frame motion anomalies. Probability: ${(streams.temporal.fake_prob * 100).toFixed(1)}%`,
        severity: streams.temporal.fake_prob > 0.8 ? "high" : "medium",
      })
    }
    if (streams.audio?.available && streams.audio?.fake_prob > 0.5) {
      detectedIssues.push({
        icon: Volume2,
        title: "Audio Forensics (RawNet2)",
        description: `Synthetic audio signatures. Probability: ${(streams.audio.fake_prob * 100).toFixed(1)}%`,
        severity: streams.audio.fake_prob > 0.8 ? "high" : "medium",
      })
    }
    if (streams.rppg?.available && streams.rppg?.has_pulse === false) {
      detectedIssues.push({
        icon: AlertCircle,
        title: "Biological Liveness Failed",
        description: `No human pulse (rPPG) detected in the subject's face. Strong indicator of synthetic generation.`,
        severity: "high",
      })
    }
    if (analysisData.fact_check?.available && analysisData.fact_check?.harmful_score > 0.5) {
      detectedIssues.push({
        icon: AlertTriangle,
        title: `Semantic Harm: ${analysisData.fact_check.label}`,
        description: `Flags: ${analysisData.fact_check.flags?.join(", ")}`,
        severity: "high",
      })
    }
  } else if (isFake) {
    // Fallback Mock Issues
    detectedIssues.push({
      icon: ImageIcon,
      title: "Face inconsistency detected",
      description: "Facial features show unnatural blending artifacts",
      severity: "high",
    })
  }

  const getVerdictColor = (status: "fake" | "real" | "uncertain") => {
    switch (status) {
      case "fake": return "text-verdict-fake"
      case "real": return "text-verdict-real"
      case "uncertain": return "text-verdict-uncertain"
    }
  }

  const getVerdictBorder = (status: "fake" | "real" | "uncertain") => {
    switch (status) {
      case "fake": return "border-verdict-fake"
      case "real": return "border-verdict-real"
      case "uncertain": return "border-verdict-uncertain"
    }
  }

  // Source verification badge component
  const SourceBadge = () => {
    if (!sourceInfo) return null
    if (sourceInfo.verified && sourceInfo.type === "url") {
      return (
        <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-verdict-real/10 border border-verdict-real/30 rounded-md text-sm">
          <CheckCircle className="w-4 h-4 text-verdict-real" />
          <span className="text-verdict-real font-medium">Verified Source: {sourceInfo.platform}</span>
        </div>
      )
    }
    return (
      <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-secondary border border-border rounded-md text-sm">
        <AlertCircle className="w-4 h-4 text-muted-foreground" />
        <span className="text-muted-foreground font-medium">Source: Unverified Upload</span>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b border-border px-6 py-4 bg-card">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="outline" size="sm" onClick={onBack} className="border-border text-muted-foreground">
              ← Back
            </Button>
            <div className="flex items-center gap-3">
              <Shield className="w-5 h-5 text-primary" />
              <span className="font-bold tracking-tight text-foreground">BharatShield Forensics</span>
            </div>
          </div>
          <div className="text-sm text-muted-foreground font-mono">
            {analysisData?.submission_id ? `CASE ID: ${analysisData.submission_id.toUpperCase()}` : "DEMO MODE"}
          </div>
        </div>
      </header>

      <main className="flex-1 px-6 py-8">
        <div className="max-w-6xl mx-auto space-y-6">
          
          {/* Top Row: Verdict & Preview */}
          <div className="grid lg:grid-cols-2 gap-6">
            
            {/* Verdict Card */}
            <Card className={`border-l-4 ${getVerdictBorder(verdictData.status)} bg-card shadow-sm`}>
              <CardContent className="p-8 h-full flex flex-col justify-center space-y-6">
                <div>
                  <h2 className="text-sm font-semibold tracking-widest text-muted-foreground uppercase mb-2">Final Verdict</h2>
                  <h1 className={`text-3xl md:text-4xl font-extrabold tracking-tight ${getVerdictColor(verdictData.status)}`}>
                    {verdictData.verdict}
                  </h1>
                </div>
                
                <div className="flex items-baseline gap-3">
                  <span className="text-5xl font-black text-foreground">{verdictData.confidence}%</span>
                  <span className="text-lg font-medium text-muted-foreground">Confidence Level</span>
                </div>

                <div className="flex flex-col gap-3 pt-4 border-t border-border">
                  <SourceBadge />
                  {verdictData.status === "fake" && (
                    <div className="inline-flex items-center gap-2 px-3 py-2 bg-verdict-fake/10 border border-verdict-fake/20 rounded-md text-sm">
                      <AlertTriangle className="w-4 h-4 text-verdict-fake" />
                      <span className="text-verdict-fake font-semibold">Takedown Protocol Recommended</span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Media Preview Card */}
            <Card className="bg-card shadow-sm">
              <CardHeader className="pb-3 border-b border-border">
                <div className="flex justify-between items-center">
                  <CardTitle className="text-lg font-medium flex items-center gap-2">
                    <MonitorPlay className="w-5 h-5 text-muted-foreground" />
                    Forensic Media Preview
                  </CardTitle>
                  <span className="text-xs font-mono text-muted-foreground bg-secondary px-2 py-1 rounded">
                    {analysisData?.file?.name || "Uploaded Media"}
                  </span>
                </div>
              </CardHeader>
              <CardContent className="p-4">
                <div className="relative aspect-video bg-black rounded-md overflow-hidden flex items-center justify-center border border-border">
                  {isVideo && displayUrl ? (
                    <video 
                      src={displayUrl} 
                      controls 
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : displayUrl ? (
                    <img 
                      src={displayUrl} 
                      alt="Forensic Analysis" 
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : (
                    <div className="text-muted-foreground text-sm">Preview Unavailable</div>
                  )}
                  
                  {isFake && !isVideo && analysisData?.deepfake_detection?.gradcam_url && (
                    <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded backdrop-blur-sm border border-white/20">
                      GradCAM Heatmap Overlay
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Bottom Row: Stream Diagnostics */}
          <Card className="bg-card shadow-sm">
            <CardHeader className="border-b border-border">
              <CardTitle className="text-lg font-medium">Analysis Stream Diagnostics</CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <div className="space-y-6">
                {!analysisData?.deepfake_detection?.streams ? (
                  <div className="text-center text-muted-foreground p-4">
                    Detailed stream diagnostics are unavailable for this analysis.
                  </div>
                ) : (
                  <div className="grid md:grid-cols-2 gap-8">
                    {/* Stream Progress Bars */}
                    <div className="space-y-5">
                      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-2">Confidence by Modality</h3>
                      
                      {/* Spatial Stream */}
                      <div className="space-y-2">
                        <div className="flex justify-between items-center text-sm">
                          <div className="flex items-center gap-2">
                            <ImageIcon className="w-4 h-4 text-muted-foreground" />
                            <span className="font-medium">Spatial (Face & Artifacts)</span>
                          </div>
                          <span className="font-mono">
                            {analysisData.deepfake_detection.streams.spatial_texture?.fake_prob !== undefined && analysisData.deepfake_detection.streams.spatial_texture?.fake_prob !== null
                              ? `${(analysisData.deepfake_detection.streams.spatial_texture.fake_prob * 100).toFixed(1)}%`
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${analysisData.deepfake_detection.streams.spatial_texture?.fake_prob > 0.5 ? 'bg-verdict-fake' : 'bg-verdict-real'}`} 
                            style={{ width: `${(analysisData.deepfake_detection.streams.spatial_texture?.fake_prob || 0) * 100}%` }}
                          />
                        </div>
                      </div>

                      {/* Frequency Stream */}
                      <div className="space-y-2">
                        <div className="flex justify-between items-center text-sm">
                          <div className="flex items-center gap-2">
                            <MonitorPlay className="w-4 h-4 text-muted-foreground" />
                            <span className="font-medium">Frequency Domain (Noise)</span>
                          </div>
                          <span className="font-mono">
                            {analysisData.deepfake_detection.streams.frequency_domain?.fake_prob !== undefined && analysisData.deepfake_detection.streams.frequency_domain?.fake_prob !== null
                              ? `${(analysisData.deepfake_detection.streams.frequency_domain.fake_prob * 100).toFixed(1)}%`
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${analysisData.deepfake_detection.streams.frequency_domain?.fake_prob > 0.5 ? 'bg-verdict-fake' : 'bg-verdict-real'}`} 
                            style={{ width: `${(analysisData.deepfake_detection.streams.frequency_domain?.fake_prob || 0) * 100}%` }}
                          />
                        </div>
                      </div>

                      {/* Temporal Stream (if video and available) */}
                      {isVideo && analysisData.deepfake_detection.streams.temporal?.fake_prob !== null && (
                        <div className="space-y-2">
                          <div className="flex justify-between items-center text-sm">
                            <div className="flex items-center gap-2">
                              <Clock className="w-4 h-4 text-muted-foreground" />
                              <span className="font-medium">Temporal (Frame Consistency)</span>
                            </div>
                            <span className="font-mono">
                              {analysisData.deepfake_detection.streams.temporal?.fake_prob !== undefined
                                ? `${(analysisData.deepfake_detection.streams.temporal.fake_prob * 100).toFixed(1)}%`
                                : 'N/A'}
                            </span>
                          </div>
                          <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${analysisData.deepfake_detection.streams.temporal?.fake_prob > 0.5 ? 'bg-verdict-fake' : 'bg-verdict-real'}`} 
                              style={{ width: `${(analysisData.deepfake_detection.streams.temporal?.fake_prob || 0) * 100}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* Audio Stream (if available) */}
                      {analysisData.deepfake_detection.streams.audio?.available && analysisData.deepfake_detection.streams.audio?.fake_prob !== null && (
                        <div className="space-y-2">
                          <div className="flex justify-between items-center text-sm">
                            <div className="flex items-center gap-2">
                              <Volume2 className="w-4 h-4 text-muted-foreground" />
                              <span className="font-medium">Audio Forensics</span>
                            </div>
                            <span className="font-mono">
                              {analysisData.deepfake_detection.streams.audio?.fake_prob !== undefined
                                ? `${(analysisData.deepfake_detection.streams.audio.fake_prob * 100).toFixed(1)}%`
                                : 'N/A'}
                            </span>
                          </div>
                          <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${analysisData.deepfake_detection.streams.audio?.fake_prob > 0.5 ? 'bg-verdict-fake' : 'bg-verdict-real'}`} 
                              style={{ width: `${(analysisData.deepfake_detection.streams.audio?.fake_prob || 0) * 100}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* rPPG / Biological Liveness (if video and available) */}
                      {analysisData.deepfake_detection.streams.rppg?.available && (
                        <div className="space-y-2">
                          <div className="flex justify-between items-center text-sm">
                            <div className="flex items-center gap-2">
                              <Activity className="w-4 h-4 text-muted-foreground" />
                              <span className="font-medium">Biological Liveness (rPPG)</span>
                            </div>
                            <span className="font-mono">
                              {analysisData.deepfake_detection.streams.rppg.has_pulse === true
                                ? 'Pulse Detected (Real)'
                                : 'No Pulse (Fake)'}
                            </span>
                          </div>
                          <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full ${analysisData.deepfake_detection.streams.rppg.has_pulse === false ? 'bg-verdict-fake w-full' : 'bg-verdict-real w-full'}`} 
                            />
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Detected Issues / Interpretation */}
                    <div className="space-y-4 border-t md:border-t-0 md:border-l border-border md:pl-8 pt-6 md:pt-0">
                      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-2">Key Findings</h3>
                      {detectedIssues.length > 0 ? (
                        <div className="space-y-4">
                          {detectedIssues.map((issue, index) => (
                            <div key={index} className="flex gap-3">
                              <div className={`p-1.5 rounded mt-0.5 h-fit ${
                                issue.severity === "high" ? "bg-verdict-fake/10 text-verdict-fake" : "bg-verdict-uncertain/10 text-verdict-uncertain"
                              }`}>
                                <issue.icon className="w-4 h-4" />
                              </div>
                              <div>
                                <h4 className="text-sm font-semibold text-foreground">{issue.title}</h4>
                                <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                                  {issue.description}
                                </p>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="flex items-start gap-3">
                          <div className="p-1.5 rounded mt-0.5 bg-verdict-real/10 text-verdict-real h-fit">
                            <CheckCircle className="w-4 h-4" />
                          </div>
                          <div>
                            <h4 className="text-sm font-semibold text-foreground">No Anomalies Found</h4>
                            <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                              The ensemble model evaluated all available data streams and found no indications of synthetic manipulation. The media characteristics align with authentic capture profiles.
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Actions */}
          <div className="flex flex-col sm:flex-row justify-end gap-4 pt-4">
            {analysisData?.legal_report_url && (
                <a href={`http://127.0.0.1:5001${analysisData.legal_report_url}`} download target="_blank" rel="noopener noreferrer">
                    <Button 
                    size="lg" 
                    variant="outline"
                    className="w-full sm:w-auto gap-2 border-primary text-primary hover:bg-primary/10"
                    >
                    <Download className="w-5 h-5" />
                    Download Legal Report
                    </Button>
                </a>
            )}
            <Button 
              size="lg" 
              onClick={onContinue}
              className="w-full sm:w-auto gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
            >
              Take Action
              <ArrowRight className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </main>
    </div>
  )
}
