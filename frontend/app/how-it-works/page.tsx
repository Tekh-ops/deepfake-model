"use client"

import { 
  Shield, 
  Upload, 
  Link, 
  Share2, 
  Hash, 
  Clock, 
  Lock, 
  Eye, 
  AudioLines, 
  FileSearch, 
  Layers, 
  User, 
  Newspaper, 
  BadgeCheck, 
  Building2, 
  FileText, 
  Bell, 
  Send,
  ArrowDown,
  Sparkles,
  ArrowLeft
} from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import NextLink from "next/link"

export default function HowItWorksPage({ onBack }: { onBack?: () => void }) {
  const steps = [
    {
      number: 1,
      title: "Submit Suspicious Content",
      icon: Upload,
      content: [
        { icon: Upload, text: "Upload file" },
        { icon: Link, text: "Paste URL" },
        { icon: Share2, text: "Share from social media" },
      ],
      highlights: [
        { icon: Hash, text: "SHA-256 hash generated" },
        { icon: Clock, text: "Timestamp recorded" },
        { icon: Lock, text: "Evidence sealed" },
      ],
      note: "Evidence is secured at the moment of submission",
    },
    {
      number: 2,
      title: "Multi-Layer AI Analysis",
      icon: Eye,
      content: [
        { icon: Eye, text: "Visual analysis (face inconsistencies)" },
        { icon: AudioLines, text: "Audio analysis (voice mismatch)" },
        { icon: FileSearch, text: "Metadata analysis (file anomalies)" },
      ],
      output: "Likely Synthetic — 87% confidence",
      note: "With explainable highlights",
    },
    {
      number: 3,
      title: "Unified Risk Assessment",
      icon: Layers,
      content: [
        { icon: Layers, text: "Combines outputs from all analysis layers" },
        { icon: BadgeCheck, text: "Produces a single final verdict" },
      ],
    },
    {
      number: 4,
      title: "Role-Based Intelligence Delivery",
      icon: User,
      isHighlighted: true,
      roles: [
        { icon: User, role: "Citizen", description: "Simple verdict + report option" },
        { icon: Newspaper, role: "Journalist", description: "Detailed analysis report" },
        { icon: BadgeCheck, role: "Police", description: "Court-ready forensic report" },
        { icon: Building2, role: "Authority", description: "Takedown notice generation" },
      ],
      note: "Same analysis, different outputs based on user role",
    },
    {
      number: 5,
      title: "Automated Action & Response",
      icon: Send,
      content: [
        { icon: FileText, text: "Evidence package generated" },
        { icon: Bell, text: "Takedown notice created" },
        { icon: Send, text: "Case forwarded to authorities" },
      ],
      note: "From detection to enforcement — in one unified system",
    },
  ]

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="border-b border-border px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Shield className="w-6 h-6 text-primary" />
            </div>
            <span className="text-xl font-semibold tracking-tight">BharatShield</span>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            {onBack ? (
              <button onClick={onBack} className="text-sm text-muted-foreground hover:text-foreground transition-colors">Home</button>
            ) : (
              <NextLink href="/" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Home</NextLink>
            )}
            <NextLink href="/how-it-works" className="text-sm text-foreground font-medium">How it Works</NextLink>
            <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Resources</a>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="px-6 py-16 text-center">
        <div className="max-w-3xl mx-auto space-y-6">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-balance animate-in fade-in slide-in-from-bottom-4 duration-700">
            How BharatShield Works
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground text-balance animate-in fade-in slide-in-from-bottom-4 duration-700 delay-150">
            From suspicious content to verified action — in a seamless, legally compliant workflow.
          </p>
        </div>
      </section>

      {/* Vertical Timeline */}
      <main className="flex-1 px-6 pb-20">
        <div className="max-w-3xl mx-auto relative">
          {/* Vertical line */}
          <div className="absolute left-8 top-0 bottom-0 w-px bg-gradient-to-b from-primary/50 via-primary/30 to-primary/10 hidden md:block" />
          
          <div className="space-y-8">
            {steps.map((step, index) => (
              <div
                key={step.number}
                className={`relative animate-in fade-in slide-in-from-bottom-4 duration-500`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                {/* Step Card */}
                <Card 
                  className={`md:ml-20 border-border transition-all duration-300 ${
                    step.isHighlighted 
                      ? "border-primary/50 bg-card shadow-lg shadow-primary/10 animate-pulse-glow" 
                      : "bg-card hover:border-primary/30"
                  }`}
                >
                  {/* Step number badge - positioned on timeline */}
                  <div className="absolute left-0 top-6 w-16 hidden md:flex items-center justify-center">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm ${
                      step.isHighlighted 
                        ? "bg-primary text-primary-foreground" 
                        : "bg-secondary text-foreground border border-border"
                    }`}>
                      {step.number}
                    </div>
                  </div>

                  <CardContent className="p-6">
                    {/* Mobile step number */}
                    <div className="md:hidden flex items-center gap-3 mb-4">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-xs ${
                        step.isHighlighted 
                          ? "bg-primary text-primary-foreground" 
                          : "bg-secondary text-foreground border border-border"
                      }`}>
                        {step.number}
                      </div>
                    </div>

                    {/* Title with icon */}
                    <div className="flex items-center gap-3 mb-4">
                      <div className={`p-2 rounded-lg ${step.isHighlighted ? "bg-primary/20" : "bg-secondary"}`}>
                        <step.icon className={`w-5 h-5 ${step.isHighlighted ? "text-primary" : "text-muted-foreground"}`} />
                      </div>
                      <h2 className="text-xl font-semibold">{step.title}</h2>
                    </div>

                    {/* Content bullets */}
                    {step.content && (
                      <ul className="space-y-2 mb-4">
                        {step.content.map((item, i) => (
                          <li key={i} className="flex items-center gap-3 text-muted-foreground">
                            <item.icon className="w-4 h-4 text-primary/70" />
                            <span>{item.text}</span>
                          </li>
                        ))}
                      </ul>
                    )}

                    {/* Highlight box for step 1 */}
                    {step.highlights && (
                      <div className="bg-secondary/50 rounded-lg p-4 mb-4 border border-border">
                        <div className="flex flex-wrap gap-4">
                          {step.highlights.map((h, i) => (
                            <div key={i} className="flex items-center gap-2">
                              <h.icon className="w-4 h-4 text-verdict-real" />
                              <span className="text-sm text-foreground">{h.text}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Output box for step 2 */}
                    {step.output && (
                      <div className="bg-verdict-fake/10 border border-verdict-fake/30 rounded-lg px-4 py-3 mb-4">
                        <div className="flex items-center gap-2">
                          <Sparkles className="w-4 h-4 text-verdict-fake" />
                          <span className="font-medium text-verdict-fake">{step.output}</span>
                        </div>
                      </div>
                    )}

                    {/* Role cards for step 4 */}
                    {step.roles && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
                        {step.roles.map((r, i) => (
                          <div 
                            key={i} 
                            className="flex items-start gap-3 p-3 bg-secondary/50 rounded-lg border border-border hover:border-primary/30 transition-colors"
                          >
                            <div className="p-1.5 bg-primary/10 rounded">
                              <r.icon className="w-4 h-4 text-primary" />
                            </div>
                            <div>
                              <p className="font-medium text-sm">{r.role}</p>
                              <p className="text-xs text-muted-foreground">{r.description}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Note */}
                    {step.note && (
                      <p className="text-sm text-muted-foreground italic border-l-2 border-primary/30 pl-3">
                        {step.note}
                      </p>
                    )}
                  </CardContent>
                </Card>

                {/* Arrow connector */}
                {index < steps.length - 1 && (
                  <div className="flex justify-center py-4 md:ml-20">
                    <ArrowDown className="w-5 h-5 text-primary/50 animate-bounce" />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Back to home button */}
          <div className="mt-12 text-center">
            {onBack ? (
              <Button onClick={onBack} variant="outline" size="lg" className="gap-2">
                <ArrowLeft className="w-4 h-4" />
                Back to Home
              </Button>
            ) : (
              <Button asChild variant="outline" size="lg" className="gap-2">
                <NextLink href="/">
                  <ArrowLeft className="w-4 h-4" />
                  Back to Home
                </NextLink>
              </Button>
            )}
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
