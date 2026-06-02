"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import Home from "../page";

function AnalyzeContent() {
  const searchParams = useSearchParams();
  const sourceUrl = searchParams.get("sourceUrl");
  
  let decodedUrl = "";
  if (sourceUrl) {
    try {
      decodedUrl = decodeURIComponent(sourceUrl);
    } catch (e) {
      decodedUrl = sourceUrl;
    }
  }

  return (
    <div className="flex flex-col min-h-screen bg-background">
      {/* Banner for External Source */}
      {decodedUrl ? (
        <div className="relative overflow-hidden bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 border-b border-border p-4 text-center z-50">
          <div className="relative z-10 max-w-4xl mx-auto flex flex-col items-center">
            <h2 className="text-lg font-semibold text-foreground tracking-tight flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
              </svg>
              Source: External Platform
            </h2>
            <div className="mt-3 inline-flex items-center gap-2 bg-background/80 px-4 py-2 rounded-full border shadow-sm">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Received URL:</span>
              <span className="text-sm font-mono text-primary break-all">{decodedUrl}</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-muted/50 border-b border-border p-3 text-center z-50">
          <p className="text-sm font-medium text-muted-foreground">No external source URL provided. Showing standard content.</p>
        </div>
      )}

      {/* Main BharatShield UI */}
      <div className="flex-1 w-full relative">
        <Home 
          initialScreen={decodedUrl ? "upload-url" : "landing"} 
          initialUrl={decodedUrl} 
        />
      </div>
    </div>
  );
}

export default function AnalyzePage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-background text-foreground">
        <div className="animate-pulse flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
          <p className="text-muted-foreground font-medium">Loading analysis engine...</p>
        </div>
      </div>
    }>
      <AnalyzeContent />
    </Suspense>
  );
}
