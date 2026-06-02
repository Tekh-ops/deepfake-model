"use client"

import { Shield, Upload, Link, Play, ArrowRight, CheckCircle, Search, Scale, FileWarning } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import NextLink from "next/link"
import type { Role } from "@/components/login-page"

interface LandingPageProps {
  userRole: Role
  onUploadClick: () => void
  onUrlClick: () => void
  onDemoClick: () => void
  onHowItWorksClick: () => void
}

export function LandingPage({ userRole, onUploadClick, onUrlClick, onDemoClick, onHowItWorksClick }: LandingPageProps) {
  const steps = [
    { icon: Upload, label: "Upload", description: "Submit media" },
    { icon: Search, label: "Analyze", description: "AI verification" },
    { icon: FileWarning, label: "Verdict", description: "Get results" },
    { icon: Scale, label: "Action", description: "Take steps" },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-border px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col gap-3">
          {/* Row 1 */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <span className="text-xl font-semibold tracking-tight">BharatShield</span>
            </div>
            <nav className="hidden md:flex items-center gap-6">
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">About</a>
              <button onClick={onHowItWorksClick} className="text-sm text-muted-foreground hover:text-foreground transition-colors">How it Works</button>
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Resources</a>
            </nav>
          </div>
          
          {/* Row 2 */}
          <div className="flex items-center">
            <span className="text-xs text-muted-foreground/80 font-medium">Logged in as: {userRole}</span>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 py-16">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full border border-primary/20 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <Shield className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary">Trusted by Government Agencies</span>
          </div>

          {/* Main Heading */}
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-balance animate-in fade-in slide-in-from-bottom-4 duration-700">
            Detect Deepfakes.
            <span className="text-primary"> Preserve Evidence. </span>
            Enable Action.
          </h1>

          {/* Subtitle */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto text-balance animate-in fade-in slide-in-from-bottom-4 duration-700 delay-150">
            A unified platform that turns suspicious media into verified evidence and enables real-world response — all in one place.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
            <Button 
              size="lg" 
              onClick={onUploadClick}
              className="w-full sm:w-auto gap-2 bg-primary hover:bg-primary/90 text-primary-foreground"
            >
              <Upload className="w-5 h-5" />
              Upload File
            </Button>
            <Button 
              size="lg" 
              variant="outline" 
              onClick={onUrlClick}
              className="w-full sm:w-auto gap-2"
            >
              <Link className="w-5 h-5" />
              Paste URL
            </Button>
            <Button 
              size="lg" 
              variant="ghost" 
              onClick={onDemoClick}
              className="w-full sm:w-auto gap-2 text-muted-foreground hover:text-foreground"
            >
              <Play className="w-5 h-5" />
              View Demo Case
            </Button>
          </div>
        </div>

        {/* Visual Flow */}
        <div className="mt-20 w-full max-w-4xl">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {steps.map((step, index) => (
              <Card 
                key={step.label} 
                className="bg-card border-border hover:border-primary/50 transition-all duration-300 group"
              >
                <CardContent className="p-6 text-center">
                  <div className="flex flex-col items-center gap-3">
                    <div className="p-3 bg-secondary rounded-lg group-hover:bg-primary/10 transition-colors">
                      <step.icon className="w-6 h-6 text-muted-foreground group-hover:text-primary transition-colors" />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">0{index + 1}</span>
                      <span className="font-medium">{step.label}</span>
                    </div>
                    <p className="text-sm text-muted-foreground">{step.description}</p>
                  </div>
                  {index < steps.length - 1 && (
                    <div className="hidden md:block absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2">
                      <ArrowRight className="w-4 h-4 text-muted-foreground" />
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Trust Indicators */}
        <div className="mt-16 flex flex-wrap items-center justify-center gap-8 text-muted-foreground">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-verdict-real" />
            <span className="text-sm">End-to-end encryption</span>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-verdict-real" />
            <span className="text-sm">Evidence preservation</span>
          </div>
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-verdict-real" />
            <span className="text-sm">Legal compliance</span>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border px-6 py-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
          <p>BharatShield — Government of India Initiative</p>
          <div className="flex items-center gap-6">
            <a href="#" className="hover:text-foreground transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-foreground transition-colors">Terms of Service</a>
            <a href="#" className="hover:text-foreground transition-colors">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  )
}
