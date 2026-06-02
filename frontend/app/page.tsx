"use client"

import { useState, useEffect, Suspense } from "react"
import { useSearchParams } from "next/navigation"
import { LandingPage } from "@/components/landing-page"
import { UploadScreen } from "@/components/upload-screen"
import { AnalysisResult } from "@/components/analysis-result"
import { RoleBasedOutput } from "@/components/role-based-output"
import { ActionConfirmation } from "@/components/action-confirmation"
import { LoginPage, type Role } from "@/components/login-page"
import HowItWorksPage from "./how-it-works/page"

type Screen = "landing" | "upload-file" | "upload-url" | "analysis" | "role-output" | "confirmation" | "how-it-works"

function MainApp() {
  const searchParams = useSearchParams()
  const sourceUrl = searchParams.get("sourceUrl")

  const [currentScreen, setCurrentScreen] = useState<Screen>("landing")
  const [userRole, setUserRole] = useState<Role | null>(null)
  const [initialUrl, setInitialUrl] = useState("")
  const [analysisData, setAnalysisData] = useState<any>(null)
  const [sourceInfo, setSourceInfo] = useState<any>(null)

  useEffect(() => {
    if (sourceUrl) {
      try {
        const decoded = decodeURIComponent(sourceUrl)
        setInitialUrl(decoded)
        setCurrentScreen("upload-url")
      } catch (e) {
        setInitialUrl(sourceUrl)
        setCurrentScreen("upload-url")
      }
    }
  }, [sourceUrl])

  const handleLogin = (role: Role) => {
    setUserRole(role)
  }

  const navigateTo = (screen: Screen) => {
    setCurrentScreen(screen)
  }

  const handleDemo = () => {
    // Skip to analysis screen for demo
    setCurrentScreen("analysis")
  }

  if (!userRole) {
    return <LoginPage onLogin={handleLogin} />
  }

  return (
    <div className="min-h-screen bg-background text-foreground relative">
      {currentScreen === "landing" && userRole && (
        <LandingPage
          userRole={userRole}
          onUploadClick={() => navigateTo("upload-file")}
          onUrlClick={() => navigateTo("upload-url")}
          onDemoClick={handleDemo}
          onHowItWorksClick={() => navigateTo("how-it-works")}
        />
      )}

      {currentScreen === "upload-file" && (
        <UploadScreen
          mode="file"
          onBack={() => navigateTo("landing")}
          onAnalyze={(sourceInfo, data) => {
            setSourceInfo(sourceInfo)
            if (data) setAnalysisData(data)
            navigateTo("analysis")
          }}
        />
      )}

      {currentScreen === "upload-url" && (
        <UploadScreen
          mode="url"
          initialUrl={initialUrl}
          onBack={() => navigateTo("landing")}
          onAnalyze={(sourceInfo, data) => {
            setSourceInfo(sourceInfo)
            if (data) setAnalysisData(data)
            navigateTo("analysis")
          }}
        />
      )}

      {currentScreen === "analysis" && (
        <AnalysisResult
          onContinue={() => navigateTo("role-output")}
          onBack={() => navigateTo("landing")}
          sourceInfo={sourceInfo}
          analysisData={analysisData}
        />
      )}

      {currentScreen === "role-output" && userRole && (
        <RoleBasedOutput
          userRole={userRole}
          onAction={() => navigateTo("confirmation")}
          onBack={() => navigateTo("analysis")}
        />
      )}

      {currentScreen === "confirmation" && (
        <ActionConfirmation
          onStartOver={() => navigateTo("landing")}
        />
      )}

      {currentScreen === "how-it-works" && (
        <HowItWorksPage onBack={() => navigateTo("landing")} />
      )}
    </div>
  )
}

export default function Home() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="animate-pulse flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
        </div>
      </div>
    }>
      <MainApp />
    </Suspense>
  )
}
