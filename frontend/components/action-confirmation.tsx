"use client"

import { CheckCircle, Shield, Copy, Home, FileText, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { useState } from "react"

interface ActionConfirmationProps {
  onStartOver: () => void
}

export function ActionConfirmation({ onStartOver }: ActionConfirmationProps) {
  const [copied, setCopied] = useState(false)

  const confirmationData = {
    trackingId: "BS-10234",
    actions: [
      { label: "Case forwarded to Cyber Cell", completed: true },
      { label: "Takedown notice generated", completed: true },
      { label: "Evidence secured", completed: true },
    ],
    timestamp: new Date().toLocaleString("en-IN", {
      dateStyle: "full",
      timeStyle: "short",
    }),
  }

  const copyTrackingId = () => {
    navigator.clipboard.writeText(confirmationData.trackingId)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-border px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-lg">
            <Shield className="w-5 h-5 text-primary" />
          </div>
          <span className="font-semibold">BharatShield</span>
        </div>
      </header>

      <main className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-lg">
          <Card className="bg-card border-border overflow-hidden">
            {/* Success Banner */}
            <div className="bg-verdict-real/10 p-8 text-center border-b border-verdict-real/20">
              <div className="w-20 h-20 rounded-full bg-verdict-real/20 flex items-center justify-center mx-auto mb-4">
                <CheckCircle className="w-10 h-10 text-verdict-real" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">
                Action Completed Successfully
              </h1>
              <p className="text-muted-foreground mt-2">
                Your report has been submitted and processed
              </p>
            </div>

            <CardContent className="p-6 space-y-6">
              {/* Completed Actions */}
              <div className="space-y-3">
                {confirmationData.actions.map((action, index) => (
                  <div 
                    key={index}
                    className="flex items-center gap-3 p-3 bg-verdict-real/5 rounded-lg border border-verdict-real/20"
                  >
                    <CheckCircle className="w-5 h-5 text-verdict-real shrink-0" />
                    <span className="text-sm">{action.label}</span>
                  </div>
                ))}
              </div>

              {/* Tracking ID */}
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-xs text-muted-foreground uppercase tracking-wide">
                      Tracking ID
                    </div>
                    <div className="text-xl font-mono font-bold mt-1">
                      {confirmationData.trackingId}
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={copyTrackingId}
                    className="shrink-0"
                  >
                    {copied ? (
                      <CheckCircle className="w-4 h-4 text-verdict-real" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground mt-3">
                  Save this ID to track the status of your report
                </p>
              </div>

              {/* Timestamp */}
              <div className="text-center text-sm text-muted-foreground">
                Submitted on {confirmationData.timestamp}
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col gap-3 pt-4">
                <Button 
                  size="lg" 
                  onClick={onStartOver}
                  className="w-full bg-primary hover:bg-primary/90 gap-2"
                >
                  <Home className="w-5 h-5" />
                  Start New Analysis
                </Button>
                <div className="grid grid-cols-2 gap-3">
                  <Button 
                    variant="outline" 
                    className="gap-2"
                  >
                    <FileText className="w-4 h-4" />
                    Download Report
                  </Button>
                  <Button 
                    variant="outline" 
                    className="gap-2"
                  >
                    <ExternalLink className="w-4 h-4" />
                    Track Status
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Support Link */}
          <p className="text-center text-sm text-muted-foreground mt-6">
            Need help? Contact{" "}
            <a href="#" className="text-primary hover:underline">
              support@bharatshield.gov.in
            </a>
          </p>
        </div>
      </main>
    </div>
  )
}
